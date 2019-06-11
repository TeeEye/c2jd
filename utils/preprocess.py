import sys
sys.path.append("..")

import pickle
import pandas as pd
import numpy as np
from utils.macros import *
from utils.embedding import Embedding


if __name__ == '__main__':
    # 目前只使用第一个训练数据
    current_train_file = 0
    raw_data_path = DATA_PATH % current_train_file
    train_data_path = TRAIN_PATH % current_train_file

    with open(EMBEDDING_PATH, 'rb') as f:
        tencent = pickle.load(f)

    # 先将所有数据合并
    current_count = 0
    with open(raw_data_path, 'rb') as f:
        app_arr = []
        while current_count < TRAIN_SIZE:
            try:
                app = pickle.load(f)
                app = app[['candidate_summary', 'job_description', 'job_class_1']]
                app = app[app['candidate_summary'].str.len() > 1]
                app = app[app['job_description'].str.len() > 1]
                app.dropna(inplace=True)
                app_arr.append(app)
                current_count += len(app)
                print('Data loaded ', len(app))
            except EOFError:
                break
        print('All data loaded!')
    print('Start data cleaning...')
    app = pd.concat(app_arr, ignore_index=True)
    app = app.head(TRAIN_SIZE)
    app['label'] = 1
    total_len = len(app)
    app.reset_index(drop=True, inplace=True)
    print('Data cleaning finished!')

    for a in range(0, total_len-1, 2):
        b = (a + total_len//2) % total_len
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
        if a % 1000 == 0:
            sys.stdout.write('\rProcessing %d / %d' % (a, total_len))
            sys.stdout.flush()
    print('\nDone!')
    print('Converting words into vecs')
    summary = []
    description = []
    embedding = Embedding()
    for idx, row in app.iterrows():
        summary.append(embedding.sentence2vec(row[0]))
        description.append(embedding.sentence2vec(row[1]))
        if idx % 1000 == 0:
            sys.stdout.write('\rProcessing %d / %d' % (idx, total_len))
            sys.stdout.flush()
    del embedding
    print('memory released!')
    app['candidate_summary'] = np.array(summary)
    app['job_description'] = np.array(description)
    print('\nDone!')
    print('Saving final data...')
    with open(train_data_path, 'wb') as f:
        pickle.dump(app, f)
    print('Done!')
