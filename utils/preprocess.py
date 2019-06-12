import pickle
import numpy as np
from utils.macros import *
from utils.embedding import Embedding

"""
    将原始数据转化为向量化后的可直接喂给网络的数据
    app_joined -> app_train
    目前原始数据文件位于 /data/wangchenghao/data/app_joined_%d.pkl
    目标文件位于同一文件夹下
"""


def text2vec(app):
    """
    将 app 的 candidate_summary 和 job_description 字段向量化
    """
    total_len = len(app)
    print('Converting words into vec')
    summary = []
    description = []
    count = 0
    for idx, row in app.iterrows():
        count += 1
        summary.append(embedding.sentence2vec(row['candidate_summary']))
        description.append(embedding.sentence2vec(row['job_description']))
        if count % 1000 == 0 or count == total_len:
            sys.stdout.write('\rProcessing %d / %d' % (count, total_len))
            sys.stdout.flush()
    app['candidate_summary'] = np.asarray(summary)
    app['job_description'] = np.asarray(description)
    print('\nDone!')


def run():
    # 目前只使用第一个训练数据
    current_train_file = 0
    raw_data_path = DATA_PATH % current_train_file
    output_data_path = TRAIN_PATH % current_train_file

    output_file = open(output_data_path, 'wb')

    with open(raw_data_path, 'rb') as f:
        while True:
            try:
                app = pickle.load(f)
                app = app[['candidate_summary', 'job_description', 'job_class_1']]
                app = app[app['candidate_summary'].str.len() > 1]
                app = app[app['job_description'].str.len() > 1]
                app.dropna(inplace=True)
                print('Data loaded ', len(app))
                app.reset_index(drop=True, inplace=True)
                # 生成负样本
                print('Shuffling data...')
                total_len = len(app)
                half_len = total_len // 2
                labels = []
                jds = []

                for idx, row in app.iterrows():
                    labels.append(1)
                    jds.append(row['job_description'])

                for idx, row in app.iterrows():
                    other = idx + half_len
                    if other >= total_len:
                        break
                    if app.iloc[idx, 2] != app.iloc[other, 2]:
                        jds[idx] = app.iloc[other, 1]
                        jds[other] = app.iloc[idx, 1]
                        labels[idx] = 0
                        labels[other] = 0
                app['label'] = labels
                app['job_description'] = jds
                print('Done!')

                app_batches = [app.iloc[i:i+TRAIN_SIZE] for i in range(0, len(app), TRAIN_SIZE)]

                # 解除引用
                del app
                batch_len = len(app_batches)
                for i in range(batch_len):
                    app_batch = app_batches[0]
                    text2vec(app_batch)
                    print(app_batch.iloc[0])
                    pickle.dump(app_batch, output_file)
                    del app_batch
                    del app_batches[0]

            except EOFError:
                break

    output_file.close()
    print('All done!')


if __name__ == '__main__':
    # 加载腾讯词向量
    embedding = Embedding()
    run()
