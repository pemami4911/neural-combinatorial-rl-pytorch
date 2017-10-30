import subprocess
import numpy as np
import sys
from math import floor, log10


if __name__ == '__main__':
    exp_i = sys.argv[1]
    #rand_seed = int(sys.argv[2])

    #np.random.seed(rand_seed)
  

    exps = [4]
    #num = np.arange(1, 9)
    num = [2]

    num_trials = 10
    
    seeds = [123, 343]

    for i in range(num_trials):
    #for rs in seeds:
        """
        for exp in exps:
            for n in num:
            
                lr = n * (1./(10 ** exp))
                subprocess.call(["./tune_hyper.sh", str(lr), str(rs), exp_i])
        """
        lr = np.random.normal(2e-4, 1e-5)
        subprocess.call(["./tune_hyper.sh", str(lr), '4911', exp_i])
