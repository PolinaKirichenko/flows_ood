import numpy as np
import matplotlib.pyplot as plt
import os.path
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle


class MINIBOONE(Dataset):
    num_classes = 2
    class_weights = None    
    def __init__(self, root=os.path.expanduser('~/datasets/UCI/miniboone/'), train=True, remake=False,
            class_idx=0, transform_idx=0):
        super().__init__()
        self.class_idx = class_idx

        if not os.path.exists(root + 'dataset.npz') or remake:
            X, Y = load_data(root + 'MiniBooNE_PID.txt')
            X0, Y0 = X[Y == 0], Y[Y == 0]
            X1, Y1 = X[Y == 1], Y[Y == 1]
            X0_train, X0_test, _, _ = train_test_split(X0, Y0, test_size=.1, random_state=0)
            X1_train, X1_test, _, _ = train_test_split(X1, Y1, test_size=.1, random_state=0)
            # normalization for each class individually
            mean0, std0 = X0_train.mean(axis=0), X0_train.std(axis=0)
            mean1, std1 = X1_train.mean(axis=0), X1_train.std(axis=0)

            np.savez(root + 'dataset.npz',
                     train0=X0_train, test0=X0_test, mean0=mean0, std0=std0,
                     train1=X1_train, test1=X1_test, mean1=mean1, std1=std1)

        data = np.load(root + 'dataset.npz')
        X_train, X_test = data['train'+str(class_idx)], data['test'+str(class_idx)]
        X_ = X_train if train else X_test
        X_normalized = (X_ - data['mean'+str(transform_idx)]) / data['std'+str(transform_idx)]
        self.X = torch.from_numpy(X_normalized).float()
        self.dim = self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[idx], self.class_idx

    def __len__(self):
        return self.X.shape[0]

    # def show_histograms(self, split, vars):
    #     data_split = getattr(self, split, None)
    #     if data_split is None:
    #         raise ValueError('Invalid data split')
    #     util.plot_hist_marginals(data_split.x[:, vars])
    #     plt.show()


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    # print("got here")
    data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    nsignal = int(data.iloc[0][0])
    nbackground = int(data.iloc[0][1])
    print("{} signal, {} background".format(nsignal, nbackground))
    minimum = min(nsignal, nbackground)
    # the signal events come first, followed by the background events
    labels = np.concatenate((np.ones(minimum), np.zeros(minimum)))
    data = data.iloc[1:].values
    data = np.concatenate((data[:minimum], data[nsignal : nsignal+minimum]))
    # print("got here")
    # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    # labels = labels[~indices]
    # i = 0
    # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.items())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # print(features_to_remove)
    # print(np.array([i for i in range(data.shape[1]) if i not in features_to_remove]))
    # print(data.shape)
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)
    return data, labels


# def load_data_normalised(root_path):
#     data, labels = load_data(root_path)
#     # Data normalization will be performed on each class separately
#     # data = (data - data.mean(axis=0)) / data.std(axis=0)
#     return data, labels
