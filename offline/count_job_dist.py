import pickle
from multiprocessing import Process

MAX_PROC = 8

'''
app_joined.pkl
| candidate_title | candidate_summary | job_title | job_summary | job_class_1 | job_class_2 | job_class_3 |
'''


def run_proc(proc_id, input_path, output_path):
    input_file = open(input_path, 'rb')
    output_file = open(output_path, 'wb')
    count_dict = {}
    while True:
        try:
            app = pickle.load(input_file)
            for idx, row in app.iterrows():
                cls = row['job_class_1']
                if cls is None or len(cls) == 0:
                    continue
                if cls in count_dict.keys():
                    count_dict[cls] += 1
                else:
                    count_dict[cls] = 1
            del app
            print('proc [%d]: batch processed!' % proc_id)
        except EOFError:
            break

    input_file.close()
    output_file.close()
    print('Process %d finished!' % proc_id)


if __name__ == '__main__':
    print('Start multi processing')
    input_path = '/data/chenghao/app_joined_%d.pkl'
    output_path = '/data/chenghao/job_count_%d.pkl'
    processes = []
    for i in range(MAX_PROC):
        p = Process(target=run_proc, args=(i, input_path % i, output_path % i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('All done!')
