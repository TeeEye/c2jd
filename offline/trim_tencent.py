import sys
sys.path.append('..')
from utils.macros import *
import pickle
from collections import defaultdict
import jieba

def run():
    print('Loading tencent embedding...')
    tencent = {}
    with open(EMBEDDING_PATH, 'rb') as f:
        while True:
            try:
                dic = pickle.load(f)
                for k, v, in dic.items():
                    tencent[k] = v
                del dic
            except EOFError:
                break
    print('Embedding initiated!')
    word_count = defaultdict(int)
    for i in range(8):
        print('Processing batch %d' % i)
        with open(DATA_PATH % i, 'rb') as f:
            while True:
                try:
                    app = pickle.load(f)
                    print('App batch loaded')
                    for _, row in app.iterrows():
                        for word in jieba.cut(row['job_description']):
                            if word in tencent:
                                word_count[word] += 1
                        for word in jieba.cut(row['candidate_summary']):
                            if word in tencent:
                                word_count[word] += 1
                    print('App batch processed')
                    del app
                except EOFError:
                    break

    print('Filtering...')
    for key in tencent.keys():
        if key not in word_count or word_count[key] < MIN_TENCENT_FREQ:
            del tencent[key]

    with open(WORD_COUNT_PATH, 'wb') as f:
        pickle.dump(tencent, f)

    print('All done!')


if __name__ == '__main__':
    run()
