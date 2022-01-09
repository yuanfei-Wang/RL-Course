import torch.multiprocessing as mp
import numpy as np
from mcts_main import worker
import time

def stat(results,workers,trial):
    final_results = np.zeros(6)
    for res in results:
        final_results += res
    final_results /= (workers*trial)
    return final_results


def start(workers):
    processes = []
    manager = mp.Manager()
    results = manager.list()
    gamma = 1
    iter_time = 0.005
    d = 10
    render = False
    trial = 2
    ntuple = True
    if ntuple:
        c = 0.001
    else:
        c = 100
    for rank in range(workers):
        p = mp.Process(target=worker, args=(rank,results,trial,gamma,c,iter_time,d,ntuple,render))
        results.append([])
        p.start()
        processes.append(p)
        time.sleep(0.1)
    
    for p in processes:
        time.sleep(0.1)
        p.join()

    final_results = stat(results,workers,trial)
    print(final_results)

if __name__=='__main__':
    start(50)
