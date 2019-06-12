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
    for a in range(0, total_len - 1, 2):
        b = (a + total_len // 2) % total_len
        row_a = app.iloc[a]
        row_b = app.iloc[b]
        cls_a = row_a['job_class_1']
        cls_b = row_b['job_class_1']
        if cls_a == cls_b:
            continue
        tmp_summary = row_a['candidate_summary']
        tmp_description = row_a['job_description']
        app.iloc[a, 0] = row_b[0]
        app.iloc[a, 1] = row_b[1]
        app.iloc[a, 2] = cls_b
        app.iloc[a, 3] = 0
        app.iloc[b, 0] = tmp_summary
        app.iloc[b, 1] = tmp_description
        app.iloc[b, 2] = cls_a
        app.iloc[b, 3] = 0
    print('\nDone!')


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
                print('Data loaded ', len(app))
                app = app[['candidate_summary', 'job_description', 'job_class_1']]
                app = app[app['candidate_summary'].str.len() > 1]
                app = app[app['job_description'].str.len() > 1]
                app.dropna(inplace=True)
                app['label'] = 1
                app.reset_index(drop=True, inplace=True)
                for offset in range(0, len(app), TRAIN_SIZE):
                    app_batch = app.iloc[offset:offset+TRAIN_SIZE]
                    shuffle_data(app_batch)
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
