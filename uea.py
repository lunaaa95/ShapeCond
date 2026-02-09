# This file is to process the datasets download from uea repo.
# from aeon.datasets import load_classification
import numpy as np
import os
from sktime.datasets import load_UCR_UEA_dataset

# split ratio: ["Tiselac": 8:1:1]. 
# Considering FacesUCR is relatively smaller dataset compared to other datasets, we reduce the split ratio to ["FacesUCR":6:2:2] to allow more samples in val/test datasets for more robust evaluation (despite this, val/test only contains 400+ samples).
dname = "FacesUCR"
# Load the dataset
X_train_raw, y_train = load_UCR_UEA_dataset(dname, split="train", return_X_y=True)  
X_test_raw, y_test = load_UCR_UEA_dataset(dname, split="test", return_X_y=True) 

def dataframe_to_2darray(df):
    num_samples, chs = df.shape
    ret = np.zeros((num_samples, chs, len(df.iloc[0, 0])), dtype=np.float32)
    for i in range(chs):
        num_samples = df.shape[0]
        num_timesteps = len(df.iloc[0, 0])
        array_2d = np.empty((num_samples, num_timesteps), dtype=np.float32)
        for j in range(num_samples):
            array_2d[j, :] = df.iloc[j, i]
        ret[:,i,:] = array_2d
    return ret

X_train_processed = dataframe_to_2darray(X_train_raw)
X_test_processed = dataframe_to_2darray(X_test_raw) 

X = np.concatenate([X_train_processed, X_test_processed], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

classes = list(set(y_train))
classes.sort()
num_classes = len(classes)

data_c = []
for i in classes:
    data_ci = X[y == i] 
    print(f"Class {i} has {len(data_ci)} samples")
    data_c.append(data_ci)

train = [i[: int(0.6 * len(i))] for i in data_c]
test = [i[int(0.6 * len(i)): int(0.8 * len(i))] for i in data_c]
val = [i[int(0.8 * len(i)): ] for i in data_c]

all = {'train': train, 'test': test, 'val': val}

r1 = 'data/' + dname

classes = [str(i).zfill(5) for i in range(num_classes)]
for na, data in all.items():
    r2 = r1 + '/' + na
    os.makedirs(r2, exist_ok=True)
    for i in range(num_classes):
        file_path = r2 + '/' + classes[i] + '.npy'
        np.save(file_path, data[i])