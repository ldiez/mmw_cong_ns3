from sklearn import metrics
from sklearn.metrics import f1_score
import itertools
from logging.handlers import RotatingFileHandler
import coloredlogs
import logging
from pandas.plotting import scatter_matrix
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix

# create logger with 'spam_application'
log = logging.getLogger("["+os.path.basename(__file__)+"] ")
log.setLevel(logging.DEBUG)

prettyFormatter = coloredlogs.ColoredFormatter(
    fmt='%(asctime)s [[%(filename)s | %(lineno)s]] %(levelname)s - %(message)s', datefmt="%H:%M:%S")
plainFormatter = logging.Formatter(
    '%(asctime)s [[%(filename)s | %(lineno)s]] %(levelname)s - %(message)s', '%H:%M:%S')
TQDM_DISABLE = False

h = logging.StreamHandler()
h.setLevel(logging.INFO)
h.setFormatter(prettyFormatter)
log.addHandler(h)

bws = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
dists = [30]  # [10, 20, 30, 40, 50]
qth = 5e5
inFolder = './traces_default/'
outFolder = './cluster_out/'


def FindOutliers(data, m=1):
    ncols = len(data[0])
    f_std = []
    f_mean = []
    for i in range(0, ncols):
        f_std.append(np.std(data[:, i]))
        f_mean.append(np.mean(data[:, i]))
    rms = []
    for i in range(0, len(data)):
        if (sum(abs(data[i, :] - f_mean) > m*np.asarray(f_std)) > 0):
            rms.append(i)
    return rms


def MinMaxScale(dataset):
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    return dataset


def PrepareData(X, Y):
    # outliers = FindOutliers(X, m=1)
    # X = np.delete(X, outliers, 0)
    # Y = np.delete(Y, outliers, 0)
    X = MinMaxScale(X)
    return X, Y


def FixLegendEntries(plt):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def PlotPred(Y, y_pred, bw, dist):
    fig = plt.figure()
    xx = range(0, len(Y))
    plt.plot(xx, Y, 'ro', label="Actual")
    plt.plot(xx, y_pred, 'b.', label="Pred")
    FixLegendEntries(plt)
    fig.savefig(outFolder + 'AAA_SvmPred_{}Mbps_{:04d}.png'.format(bw, dist))
    plt.close(fig)


def PlotClusters(X, Y, y_pred, bw, dist):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 1)
    auxCong = X[y_pred == 1, :]
    auxNoCong = X[y_pred == 0, :]

    ax.plot(auxCong[:, 0], auxCong[:, 1], 'r.', label="W/Cong")
    ax.plot(auxNoCong[:, 0], auxNoCong[:, 1], 'b.', label="Wo/Cong")

    ax = fig.add_subplot(2, 1, 2)
    auxCong = X[Y == 1, :]
    auxNoCong = X[Y == 0, :]

    ax.plot(auxCong[:, 0], auxCong[:, 1], 'r.', label="W/Cong")
    ax.plot(auxNoCong[:, 0], auxNoCong[:, 1], 'b.', label="Wo/Cong")
    FixLegendEntries(plt)
    fig.savefig(outFolder + 'AAA_SvmClusters_{}Mbps_{:04d}.png'.format(bw, dist))
    plt.close(fig)


def PlotConfusionMatrix(y_true, y_pred):
    labels = ['W/Cong', 'Wo/Cong']
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float)

    cm[0, :] = cm[0, :] / sum(cm[0, :])
    cm[1, :] = cm[1, :] / sum(cm[1, :])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(outFolder + 'AAA_SvmConfmat_{}Mbps_{:04d}.png'.format(bw, dist))
    plt.close(fig)


def ChoseLabel(y_pred, Y):
    y_pred_inv = np.logical_not(y_pred)
    asIs = float(sum(y_pred == Y))/len(Y)
    inv = float(sum(y_pred_inv == Y))/len(Y)
    if (inv > asIs):
        return y_pred_inv
    return y_pred


def getMetrics(Y, y_pred):
    cm = confusion_matrix(Y, y_pred)
    cm = cm.astype(float)
    recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    precission = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / \
        (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    return [precission, recall, accuracy]


def random_with_condition(Y, n):
    idxs1 = np.where(Y == 1.0)[0]
    idxs1_aux = random.sample(np.arange(idxs1.size), n)
    idxs0 = np.where(Y == 0.0)[0]
    idxs0_aux = random.sample(np.arange(idxs0.size), n)
    aux = np.concatenate((idxs0[idxs0_aux], idxs1[idxs1_aux]))
    np.random.shuffle(aux)
    return aux


def DoCluster(data, bw, dist):
    clf = svm.SVC(random_state=0, gamma='auto')
    X = data[:, [0, 1]]
    Y = data[:, 2]
    X, Y = PrepareData(X, Y)
    # idxs = random.sample(np.arange(Y.size), int(0.1 * (Y.size)))
    idxs = random_with_condition(Y, int(0.025*Y.size))
    svc = clf.fit(X[idxs, :], Y[idxs])
    y_pred = svc.predict(X)

    report = metrics.classification_report(Y, y_pred, labels=[0, 1], target_names=[
        'No congestion', 'Congestion'])
    print(report)

    # PlotPred(Y, y_pred, bw, dist)
    # PlotClusters(X, Y, y_pred, bw, dist)
    # PlotConfusionMatrix(Y, y_pred)
    return getMetrics(Y, y_pred)
    # return [1, 2, 3]


def PlotArray(X, bw, dist):
    fig = plt.figure()
    plt.plot(X, 'b.')
    fig.savefig(outFolder + 'cong-nocong_{}Mbps_{:04d}.png'.format(bw, dist))
    plt.close(fig)


def LoadData(bw, dist):
    fn = "static_{}Mbps_{:04d}.txt".format(bw, dist)
    dpath = inFolder + fn
    dataset = pd.read_csv(dpath, delimiter=',', names=[
        'seq', 'time', 'buff', 'del', 'iat'])
    dataset['cong'] = [1 if x > qth else 0 for x in dataset['buff']]
    return dataset[['del', 'iat', 'cong']].to_numpy()


if __name__ == '__main__':
    for dist in dists:
        res = np.zeros((len(bws), 3))
        for idx, bw in enumerate(bws):
            # for bw, dist in itertools.product(bws, dists):
            print('bw {}'.format(bw))
            ds = LoadData(bw, dist)
            ret = DoCluster(ds, bw, dist)
            res[idx, :] = ret
        np.savetxt('svmMetrics.txt', res, delimiter=' ')
        # try:
        #     DoCluster(ds, bw, dist)
        # except ValueError:
        #     print('No Classification for {}Mbps & {}m'.format(bw, dist))
