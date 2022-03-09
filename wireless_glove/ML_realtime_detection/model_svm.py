import os
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle


def feature_extraction(x):
    data_feature = []
    data_ave = np.mean(x, 1)
    data_std = np.std(x, 1)
    data_max = np.max(x, 1)
    data_min = np.min(x, 1)
    data_energy = np.sum(np.abs(x), 1)
    data_max_min = data_max - data_min
    list_feature = [data_ave, data_std, data_max, data_min, data_energy, data_max_min]
    data_feature.append(list_feature)

    data_feature = np.array(data_feature)
    return data_feature


training_data = []
window_size = 2
overlapping_rate = 0.5
data_feature = []

root_path = "./data"
folders = os.listdir(root_path)
for folder in folders:
    label = np.array(folder).reshape(1, -1)
    filepath = os.path.join(root_path, folder)
    files = os.listdir(filepath)
    for file in files:
        txtfilepath = os.path.join(filepath, file)
        txtfiles = os.listdir(txtfilepath)
        for txtfile in txtfiles: 
            sample = np.loadtxt(txtfilepath + '/' + txtfile, delimiter=",")
            
            for i in range(np.shape(sample)[1]):
                if i*overlapping_rate*window_size+window_size < np.shape(sample)[1]:
                    x=int(i*overlapping_rate*window_size)
                    y=int(i*overlapping_rate*window_size+window_size)
                    temp.append(sample[:, x:y])
            temp = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1]*np.shape(temp)[2]))
            
            feature = feature_extraction(sample).reshape(1, -1)
            print(np.shape(feature))
            feature_label = np.concatenate((feature, label), axis=1)
            training_data.append(feature_label)
print(np.shape(training_data))
training_data = np.array(training_data).reshape(132, -1)
print("training data size:", np.shape(training_data))
X = training_data[:, 0:-1]
y = training_data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=13)

scores = ['precision', 'recall']

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# clf = GridSearchCV(SVC(), tuned_parameters, cv=5)

# clf.fit(X_train, y_train)



for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)

    clf.fit(X_train, y_train)
    filename = 'svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    print("Best parameters set found on development set:")
    print()

    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)

    print(classification_report(y_true, y_pred))

    print()

