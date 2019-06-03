import os
import sys
import pickle
import pandas as pd
from time import time
from multiprocessing import Process

'''
!!!
    使用该模块之前应当先执行 join_app_candidate.py
!!!

jobs.pkl:
id | title | job_description | job_class_1 | job_class_2 | job_class_3

app_candidate.pkl
application_id | job_id | candidate_id | candidate_title | candidate_summary |
stage | stage_type | archived | filter_result

app_joined.pkl


该模块实现了对 app_candidate 和 jobs 两个表的 join 操作.

整个文件的逻辑是:
首先构建 job_dict 词典 (装不下app_dict), 以 job_id 为键, 其他所有 job 的字段为 value
构建过程是:
    如果已经被缓存则直接读取缓存
    如果没有缓存, 则从磁盘中加载 jobs.pkl, 然后逐行处理构建词典, 然后把结果缓存下来
然后多进程并行处理 app_candidate.pkl, 具体处理过程是:
    遍历 batch, 对每个 row, 根据其 job_id 获得 job_dict 的对应字段
    将 row 的所有字段和 job_dict 的 title 和 summary 整合, 放入数组
    将数组转化为一个 DataFrame, 表示该 batch join 的结果, 存入文件系统
等待所有进程完成, 那么一共会输出 {MAX_PROC} 个文件, 即为 join 结果
'''

BASE = '/data/chenghao'
JOB_PATH = os.path.join(BASE, 'jobs_cls.pkl')
JOB_DICT_PATH = os.path.join(BASE, 'job_dict.pkl')
APP_PATH = os.path.join(BASE, 'app_class_%d.pkl')
OUTPUT_PATH = os.path.join(BASE, 'app_joined_%d.pkl')

MAX_PROC = 8


# job 全部在内存, app 为一个 batch,
# 每次遍历 app 将 job_dict 的title和job_description填充进去
def join_app_job(job_dict, app_batch, output_file):
    """
    根据 job_dict 将 app_batch 的信息填充完整, 并存入 output_file
    :param job_dict: 以 job_id 为键的词典
    :param app_batch: 申请记录的 DataFrame
    :param output_file: 输出文件
    :return: 无返回
    """
    joined_array = []

    for _, row in app_batch.iterrows():
        job_id = row['job_id']
        if job_id not in job_dict:
            continue
        application_id, candidate_id, c_title, c_summary, \
            stage, stage_type, archived, filter_result = row['application_id'], row['candidate_id'], \
            row['candidate_title'], row['candidate_summary'], row['stage'], row['stage_type'], \
            row['archived'], row['filter_result']
        j_title, j_desc, j_c1, j_c2, j_c3 = job_dict[job_id]
        joined_array.append(pd.DataFrame([[application_id, job_id, j_title, j_desc, j_c1, j_c2, j_c3,
                                           candidate_id, c_title, c_summary,
                                           stage, stage_type, archived, filter_result]],
                                         columns=['application_id', 'job_id', 'job_title', 'job_description',
                                                  'job_class_1', 'job_class_2', 'job_class_3', 'candidate_id',
                                                  'candidate_title', 'candidate_summary', 'stage',
                                                  'stage_type', 'archived', 'filter_result']))

    # 将结果存入磁盘
    if len(joined_array) == 0:
        return
    joined = pd.concat(joined_array, ignore_index=True)
    pickle.dump(joined, output_file)


def run_proc(job_dict, proc_id, input_path, output_path):
    """
    每个进程的任务, 和 join_app_candidate 不同, 每个进程独立处理一个文件, 所以比较方便
    :param job_dict: 包含了 JD 的 title 和 description
    :param proc_id: 进程的 id (自定义 id 而非 OS 的 pid)
    :param input_path: (app_candidate.pkl 路径)
    :param output_path: (app_joined.pkl 路径)
    :return: 无返回
    """
    input_file = open(input_path, 'rb')
    output_file = open(output_path, 'wb')
    batch_count = 0
    while True:
        try:
            start = time()
            app = pickle.load(input_file)
            join_app_job(job_dict, app, output_file)
            del app
            print('Process [%d]: Batch_%d joined! Time cost: %.3f' % (proc_id, batch_count, time()-start))
            batch_count += 1
        except EOFError:
            break

    input_file.close()
    output_file.close()
    print('Process [%d]: Finished!' % proc_id)


def load_jobs():
    """
    加载 jobs.pkl (存储的是 JD 的 DataFrame)
    :return: jobs.pkl
    """
    print('Loading job data into memory...')
    job_file = open(JOB_PATH, 'rb')
    jobs = pickle.load(job_file)
    job_file.close()
    print('Jobs loaded!')
    return jobs


def build_job_dict(jobs):
    """
    构建 job_dict
    :param jobs: jobs.pkl DataFrame
    :return: job_dict
    """
    job_dict = {}
    count = 0
    total_len = len(jobs)
    for idx, row in jobs.iterrows():
        if (count + 1) % 1000 == 0 or (count + 1) == total_len:
            sys.stdout.write('\rProcessing %d/%d...' % (count + 1, total_len))
            sys.stdout.flush()
        count += 1
        job_id = row['id']
        title = row['title']
        desc = row['job_description']
        j_c1, j_c2, j_c3 = row['class_1'], row['class_2'], row['class_3']
        job_dict[job_id] = (title, desc, j_c1, j_c2, j_c3)
    return job_dict


def run():
    # 构建 job_dict
    if os.path.exists(JOB_DICT_PATH):
        # 如果有缓存则直接读取缓存
        with open(JOB_DICT_PATH, 'rb') as f:
            job_dict = pickle.load(f)
        print('Job dict loaded!')
    else:
        # 否则从头开始构建
        jobs = load_jobs()
        print('Building job dict...')
        job_dict = build_job_dict(jobs)
        del jobs
        with open(JOB_DICT_PATH, 'wb') as f:
            pickle.dump(job_dict, f)
        print('Job dict built!')

    # 多进程 join, 因为 app 文件是离散的, 所以最好进程数和 app 文件数一致
    print('Start multi processing')
    processes = []
    for i in range(MAX_PROC):
        p = Process(target=run_proc, args=(job_dict, i, APP_PATH % i, OUTPUT_PATH % i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('All done!')


if __name__ == '__main__':
    run()
