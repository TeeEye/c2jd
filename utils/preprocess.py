import sys
sys.path.append("..")

import pickle
import numpy as np
from utils.macros import *
from utils.embedding import Embedding

"""
    将原始数据转化为向量化后的可直接喂给网络的数据
    app_joined -> app_train
"""

def shuffle_data(app):
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
            print(app.iloc[idx, 2], app.iloc[other, 2])
            jds[idx] = app.iloc[other, 1]
            jds[other] = app.iloc[idx, 1]
            labels[idx] = 0
            labels[other] = 0
        print(jds[idx])
        print(jds[other])
    app['label'] = labels
    del app['job_description']
    app['job_description'] = jds
    print('Done!')


def text2vec(app):
    total_len = len(app)
    print('Converting words into vec')
    summary = []
    description = []
    count = 0
    for idx, row in app.iterrows():
        count += 1
        summary.append(embedding.sentence2vec(row[0]))
        description.append(embedding.sentence2vec(row[1]))
        if count % 1000 == 0:
            sys.stdout.write('\rProcessing %d / %d' % (count, total_len))
            sys.stdout.flush()
    del app['candidate_summary']
    del app['job_description']
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
                shuffle_data(app)
                print(app.head(10))
                text2vec(app)
                pickle.dump(app, output_file)
                # for offset in range(0, len(app), TRAIN_SIZE):
                #     app_batch = app.iloc[offset:offset+TRAIN_SIZE]
                #
                del app
                break
            except EOFError:
                break

    output_file.close()
    print('All done!')


if __name__ == '__main__':
    embedding = Embedding()
    run()
