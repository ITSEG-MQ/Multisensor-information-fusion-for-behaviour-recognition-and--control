import os
import numpy as np
from sklearn.model_selection import train_test_split


def data_preparation():
    window_size = 200
    overlapping_rate = 0.5
    sample_ = []
    label_ = []
    root_path = "./raw_data"
    files = os.listdir(root_path)
    for fi in files:
        if fi == "idle":
            label = np.array(float(0))
        elif fi == "stand":
            label = np.array(float(1))
        elif fi == "walk":
            label = np.array(float(2))
        filepath = os.path.join(root_path, fi)
        files = os.listdir(filepath)
        for file in files:
            sample = np.loadtxt(filepath + "/" + file, comments="%", delimiter=",", skiprows=5,
                                usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
            sample = sample.transpose(1, 0)

            for i in range(np.shape(sample)[1]):
                if i * overlapping_rate * window_size + window_size < np.shape(sample)[1]:
                    x = int(i * overlapping_rate * window_size)
                    y = int(i * overlapping_rate * window_size + window_size)
                    sample_.append(sample[:, x:y])
                    label_.append(label)

    x = np.array(sample_)
    y = np.array(label_)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
    return X_train, X_test, y_train, y_test


