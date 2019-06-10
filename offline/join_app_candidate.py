import os
import sys
import pickle
import pandas as pd
from time import time


'''
app_result.pkl:
申请id | 职位id | 候选人id | 阶段 | 阶段类型 | 是否归档 | 筛选结果   

candidate.pkl:
candidate_id | title | summary

app_dict.pkl
application_id | job_id | result (这个字段无意义..) | stage | stage_type | archived | filter_result

app_candidate.pkl
application_id | job_id | candidate_id | candidate_title | candidate_summary |
stage | stage_type | archived | filter_result

该模块实现了对 app_result 和 candidate 两个表的 join 操作,
由于需要跟踪具体的招聘进展, 所以需要保留 app_result 后四个字段
方便日后修改规则.

整个文件的逻辑是:
首先构建 app_dict 词典, 以 candidate_id 为键, 其他所有 application 的字段为 value
构建过程是:
    如果已经被缓存则直接读取缓存
    如果没有缓存, 则从磁盘中加载 app_result.pkl, 然后逐行处理构建词典, 然后把结果缓存下来
然后多进程并行处理 (共 {MAX_PROC} 个), 每个进程独立打开一个 candidate.pkl 文件 (内含多个 batch)
每个进程只处理属于自己的 batch, 比如一共10个进程, 那么进程 0 只处理 batch 0, 10, 20, 30...
具体处理过程是:
    遍历 batch, 对每个 row, 根据其 candidate_id 获得 application 的对应字段
    将 row 的 title 和 summary 以及上述其他字段组合, 放入数组
    将数组转化为一个 DataFrame, 表示该 batch join 的结果, 存入文件系统
一共会输出 {MAX_PROC} 个文件, 即为 join 结果
'''


# 定义宏
BASE = '/data/chenghao'  # 基地址
APP_PATH = os.path.join(BASE, 'app_result.pkl')  # 申请记录文件
CANDIDATE_PATH = os.path.join(BASE, 'candidate.pkl')  # 候选人文件
APP_DICT_PATH = os.path.join(BASE, 'app_dict.pkl')  # 申请记录词典（以候选人id为键）
OUTPUT_PATH = os.path.join(BASE, 'app_candidate_%d.pkl')  # join后的输出文件
MAX_PROC = 8  # 进程数量


# app_dict 全部在内存, candidate 为一个 batch,
# 每次遍历 candidate 将 app_dict 的相应字段填充进去
# 这样总操作的复杂度为 O(m+n)
def join_app_candidate(app_dict, candidate, output_file):
    """
    join 申请记录和申请人数据
    :param app_dict: 申请记录词典, 以 candidate_id 为 key
    :param candidate: 申请人数据 (batch)
    :param output_file:  输出文件
    :return: 无返回
    """
    joined_array = []  # join后的结果

    for idx, row in candidate.iterrows():
        candidate_id = row['candidate_id']
        if candidate_id not in app_dict:
            continue
        title, summary = row['title'], row['summary']
        application_id, job_id, result, stage, stage_type, archived, filter_result = app_dict[candidate_id]
        joined_array.append(pd.DataFrame([[application_id, job_id, candidate_id, title, summary,
                                           stage, stage_type, archived, filter_result]],
                                         columns=['application_id', 'job_id', 'candidate_id', 'candidate_title',
                                                  'candidate_summary', 'stage', 'stage_type', 'archived',
                                                  'filter_result']))

    # 将结果存入磁盘
    if len(joined_array) == 0:
        return
    joined = pd.concat(joined_array, ignore_index=True)
    pickle.dump(joined, output_file)


def load_app():
    """
    从磁盘中将 app_result.pkl 加载到内存并合并成一个 DataFrame
    :return: 合并后的 DataFrame
    """
    print('Loading application data into memory...')
    app_array = []
    app_file = open(APP_PATH, 'rb')
    app_count = 0
    # 将 pickle 文件中的所有 DataFrame 整合到一起
    while True:
        try:
            app = pickle.load(app_file)
            app_count += len(app)
            app_array.append(app)
        except EOFError:
            break
    app = pd.concat(app_array, axis=1)
    app_file.close()
    print('Application loaded!')
    return app


def build_app_dict(app):
    """
    将 app DataFrame 转化为一个 dict, key 是 candidate_id
    :param app: DataFrame 文件
    :return: 以 candidate_id 为 key 的 dict
    """
    app_dict = {}
    count = 0
    total_len = len(app)
    for idx, row in app.iterrows():
        # 打印进度
        if (count + 1) % 1000 == 0 or (count + 1) == total_len:
            sys.stdout.write('\rBuilding progress %d/%d...' % (count + 1, total_len))
            sys.stdout.flush()
        count += 1
        application_id = row['申请id']
        candidate_id = row['候选人id']
        result = row['result']  # 这个字段不应该存在, 下次有时间删除
        job_id = row['职位id']
        stage = row['阶段']
        stage_type = row['阶段类型']
        archived = row['是否归档']
        filter_result = row['筛选结果']
        # 以上是字典的每一项
        app_dict[candidate_id] = (application_id, job_id, result, stage, stage_type, archived, filter_result)
    return app_dict


def run():
    # 构造 app dict
    # 首先判断 app dict 是否被缓存
    if os.path.exists(APP_DICT_PATH):
        # 如果 app dict 已经被缓存, 则直接读取
        with open(APP_DICT_PATH, 'rb') as f:
            app_dict = pickle.load(f)
        print('Application dict loaded!')
    else:
        # 没有被缓存. 从零开始构造 app dict, 大概需要两个小时
        app = load_app()
        print('Building application dict...')
        app_dict = build_app_dict(app)
        # 将 app_dict 缓存下来
        with open(APP_DICT_PATH, 'wb') as f:
            pickle.dump(app_dict, f)
        del app  # 删除 app 释放内存 (内存很缺)
        print('Application dict built!')

    # 开始 join candidate
    print('Start joining candidate...')
    candidate_file = open(CANDIDATE_PATH, 'rb')
    output_files = []
    for i in range(MAX_PROC):
        output_files.append(open(OUTPUT_PATH % i, 'wb'))

    batch_count = -1
    while True:
        try:
            batch_count += 1
            candidate = pickle.load(candidate_file)
            start = time()
            join_app_candidate(app_dict, candidate, output_files[batch_count % MAX_PROC])
            del candidate
            print('Batch_%d joined! Time cost: %.3f' % (batch_count, time()-start))
        except EOFError:
            break

    for i in range(MAX_PROC):
        output_files[i].close()
    candidate_file.close()
    print('All done!')


if __name__ == '__main__':
    run()
