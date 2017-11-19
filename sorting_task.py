# Generate sorting data and store in .txt
# Define the reward function 

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange, tqdm
import os
import sys


def reward(sample_solution, USE_CUDA=False):
    """
    The reward for the sorting task is defined as the
    length of the longest sorted consecutive subsequence.

    Input sequences must all be the same length.

    Example: 

    input       | output
    ====================
    [1 4 3 5 2] | [5 1 2 3 4]
    
    The output gets a reward of 4/5, or 0.8

    The range is [1/sourceL, 1]

    Args:
        sample_solution: list of len sourceL of [batch_size]
        Tensors
    Returns:
        [batch_size] containing trajectory rewards
    """
    batch_size = sample_solution[0].size(0)
    sourceL = len(sample_solution)

    longest = Variable(torch.ones(batch_size), requires_grad=False)
    current = Variable(torch.ones(batch_size), requires_grad=False)

    if USE_CUDA:
        longest = longest.cuda()
        current = current.cuda()

    for i in range(1, sourceL):
        # compare solution[i-1] < solution[i] 
        res = torch.lt(sample_solution[i-1], sample_solution[i]) 
        # if res[i,j] == 1, increment length of current sorted subsequence
        current += res.float()  
        # else, reset current to 1
        current[torch.eq(res, 0)] = 1
        #current[torch.eq(res, 0)] -= 1
        # if, for any, current > longest, update longest
        mask = torch.gt(current, longest)
        longest[mask] = current[mask]
    return -torch.div(longest, sourceL)

def create_dataset(
        train_size,
        val_size,
        #test_size,
        data_dir,
        data_len,
        seed=None):

    if seed is not None:
        torch.manual_seed(seed)
    
    train_task = 'sorting-size-{}-len-{}-train.txt'.format(train_size, data_len)
    val_task = 'sorting-size-{}-len-{}-val.txt'.format(val_size, data_len)
    #test_task = 'sorting-size-{}-len-{}-test.txt'.format(test_size, data_len)
    
    train_fname = os.path.join(data_dir, train_task)
    val_fname = os.path.join(data_dir, val_task)

    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    else:
        if os.path.exists(train_fname) and os.path.exists(val_fname):
            return train_fname, val_fname
    
    train_set = open(os.path.join(data_dir, train_task), 'w')
    val_set = open(os.path.join(data_dir, val_task), 'w') 
    #test_set = open(os.path.join(data_dir, test_task), 'w')
    
    def to_string(tensor):
        """
        Convert a a torch.LongTensor 
        of size data_len to a string 
        of integers separated by whitespace
        and ending in a newline character
        """
        line = ''
        for j in range(data_len-1):
            line += '{} '.format(tensor[j])
        line += str(tensor[-1]) + '\n'
        return line
    
    print('Creating training data set for {}...'.format(train_task))
    
    # Generate a training set of size train_size
    for i in trange(train_size):
        x = torch.randperm(data_len)
        train_set.write(to_string(x))

    print('Creating validation data set for {}...'.format(val_task))
    
    for i in trange(val_size):
        x = torch.randperm(data_len)
        val_set.write(to_string(x))

#    print('Creating test data set for {}...'.format(test_task))
#
#    for i in trange(test_size):
#        x = torch.randperm(data_len)
#        test_set.write(to_string(x))

    train_set.close()
    val_set.close()
#    test_set.close()
    return train_fname, val_fname

class SortingDataset(Dataset):

    def __init__(self, dataset_fname):
        super(SortingDataset, self).__init__()
       
        print('Loading training data into memory')
        self.data_set = []
        with open(dataset_fname, 'r') as dset:
            lines = dset.readlines()
            for next_line in tqdm(lines):
                toks = next_line.split()
                sample = torch.zeros(1, len(toks)).long()
                for idx, tok in enumerate(toks):
                    sample[0, idx] = int(tok)
                self.data_set.append(sample)
        
        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

if __name__ == '__main__':
    if int(sys.argv[1]) == 0:
        #sample = Variable(torch.Tensor([[3, 2, 1, 4, 5], [2, 3, 5, 1, 4]])) 
        sample = [Variable(torch.Tensor([3,2])), Variable(torch.Tensor([2,3])), Variable(torch.Tensor([1,5])),
                Variable(torch.Tensor([4, 1])), Variable(torch.Tensor([5, 4]))]
        answer = torch.Tensor([3/5., 3/5])

        res = reward(sample)

        print('Expected answer: {}, Actual answer: {}'.format(answer, res.data))
        """
        sample = Variable(torch.Tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])) 
        answer = torch.Tensor([1., 1/5])

        res = reward(sample)

        print('Expected answer: {}, Actual answer: {}'.format(answer, res.data))
        
        sample = Variable(torch.Tensor([[1, 2, 5, 4, 3], [4, 1, 2, 3, 5]])) 
        answer = torch.Tensor([3/5., 4/5])

        res = reward(sample)

        print('Expected answer: {}, Actual answer: {}'.format(answer, res.data))
        """
    elif int(sys.argv[1]) == 1:
        create_sorting_dataset(1000, 100, 'data', 10, 123)
    elif int(sys.argv[1]) == 2:

        sorting_data = SortingDataset('data', 'sorting-size-1000-len-10-train.txt',
            'sorting-size-100-len-10-val.txt')
        
        for i in range(len(sorting_data)):
            print(sorting_data[i])
