import os
import numpy as np
import torch
from torch.utils.data import Dataset


training_data = []
window_size = 250
overlapping_rate = 0.5
sample_ = []
label_ = []


class LoadDataset_from_numpy():
    def __init__(self, X_train, y_train):
        super(LoadDataset_from_numpy, self).__init__()
        self.len = len(X_train)
        self.x_data = torch.from_numpy(X_train)
        self.x_data = self.x_data.to(torch.float32)
        # print(self.x_data.size())
        self.y_data = torch.from_numpy(y_train)
        self.y_data = self.y_data.to(torch.long)
        # print(self.y_data.size())

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_data, training_label, subject_data, subject_label, batch_size):
    train_dataset = LoadDataset_from_numpy(training_data, training_label)
    test_dataset = LoadDataset_from_numpy(subject_data, subject_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader

