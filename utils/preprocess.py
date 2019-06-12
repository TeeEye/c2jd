import sys
sys.path.append("..")

import pickle
import numpy as np
from utils.macros import *
from utils.embedding import Embedding


def shuffle_data(app):
    # 生成负样本
    print('Shuffling data...')
    total_len = len(app)
    half_len = total_len // 2
    labels = []
    jds = []
    for _ in range(total_len):
        labels.append(1)
        jds.append([''])
    for idx, row in app.iterrows():
        if idx >= half_len:
            break
        other = idx + half_len
        if idx % 2 == 0 or app.iloc[idx, 2] == app.iloc[other, 2]:
            jds[idx] = app.iloc[idx, 1]
            jds[other] = app.iloc[other, 1]
        else:
            jds[idx] = app.iloc[other, 1]
            jds[other] = app.iloc[idx, 1]
            labels[idx] = 0
            labels[other] = 0
    app['label'] = labels
    del app['job_description']
    app['job_description'] = jds
    print('Done!')


def text2vec(app):
    total_len = len(app)
    print('Converting words into vec')
    summary = []
    description = []
    for idx, row in app.iterrows():
        summary.append(embedding.sentence2vec(row[0]))
        description.append(embedding.sentence2vec(row[1]))
        if idx % 1000 == 0:
            sys.stdout.write('\rProcessing %d / %d' % (idx, total_len))
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
        app_arr = []
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
                for offset in range(0, len(app), TRAIN_SIZE):
                    app_batch = app.iloc[offset:offset+TRAIN_SIZE]
                    text2vec(app_batch)
                    pickle.dump(app_batch, output_file)
                del app
            except EOFError:
                break

    output_file.close()
    print('All done!')


if __name__ == '__main__':
    embedding = Embedding()
    run()
