# This file is to process the datasets download directly from CondTSC. 
# We re-classify train, val and test datasets with each dataset has identical class distribution of samples. e.g. # the ratio of class 1: # class 2 in train set is equal to that in val/test set.

import torch
import os
import argparse
import numpy as np
from pyts.datasets import load_gunpoint

parser = argparse.ArgumentParser(description='raw_data processing')
parser.add_argument('--dataset', type=str, default='har', help='har; electric; fault_a; insect; sleep')

args = parser.parse_args()

if args.dataset in ['gunpoint']:
    data = load_gunpoint(return_X_y=True)
    x_train, x_test, y_train, y_test = [torch.from_numpy(i) for i in data]
    x_train, x_test = [item.unsqueeze(1) for item in [x_train, x_test]]
    sp_point = int(x_test.shape[0] * 0.5) 
    x_val, x_test = x_test[:sp_point], x_test[sp_point:]
    y_val, y_test = y_test[:sp_point], y_test[sp_point:]
    train, val, test = {}, {}, {}
    train['labels'], train['samples']= y_train, x_train
    val['labels'], val['samples']= y_val, x_val
    test['labels'], test['samples']= y_test, x_test
else:
    train = torch.load('./raw_data/{}/train.pt'.format(args.dataset), weights_only=True)
    test = torch.load('./raw_data/{}/test.pt'.format(args.dataset), weights_only=True)
    val = torch.load('./raw_data/{}/val.pt'.format(args.dataset), weights_only=True)
train_num = len(train['labels'])
test_num = len(test['labels'])
val_num = len(val['labels'])
# we follow this spilt ratio to re-classify har, electric and sleep datasets.
ratio = {'train': train_num/(train_num+test_num+val_num),
         'test':test_num/(train_num+test_num+val_num),}

all_x = torch.cat([train['samples'], test['samples'], val['samples']])
all_y = torch.cat([train['labels'], test['labels'], val['labels']])

all_data = [(i, int(j.item())) for (i, j) in zip(all_x, all_y)]

classes = list(set([y for (x,y) in all_data]))
classes.sort()
class2idx = {item[1]: item[0] for item in enumerate(classes)}
all_data_dict = dict(zip(range(len(classes)), [[] for i in range(len(classes))]))
for x, c in all_data:
    c_idx = class2idx[c]
    all_data_dict[c_idx].append(x)

for k, v in all_data_dict.items():
    all_data_dict[k] = torch.stack(all_data_dict[k]).numpy()
    print("class {} has {} samples".format(k, all_data_dict[k].shape[0]))
    samples_num = all_data_dict[k].shape[0]
    nums = [int(ratio['train'] * samples_num), int(ratio['test'] * samples_num)]
    nums.append(int(samples_num - sum(nums)))
    nums = dict(zip(['train', 'test', 'val'], nums))
    
    st = 0
    for stage in ['train', 'test', 'val']:
        folder_path = "{}/{}/{}".format('data', args.dataset, stage)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = "{}/{:05d}".format(folder_path, k)
        x = all_data_dict[k][st: st + nums[stage],...]
        print("class {}, stage {}, from {} to {}, has {} samples of {} class samples".format(k, stage, st, st + nums[stage], nums[stage], len(all_data_dict[k])))
        st += nums[stage]
        np.save(file_path, x)
    print('train:test:val divede by ratio: {:2f}: {:2f}: {:2f}, saved as {} classes.'.format(
        ratio['train'], ratio['test'], 1.0 - sum(ratio.values()), len(classes)))
    