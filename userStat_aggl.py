import pandas as pd
import psycopg2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import glob
from scipy import stats
from sklearn.decomposition import PCA
import gapkmean


# -*- coding: utf-8 -*-
import os
import sys
import copy
import time

# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
#
# https://en.wikipedia.org/wiki/Variation_of_information

from math import log

def variation_of_information(X, Y):

    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    return abs(sigma)

# VI = 3.322 (maximum VI is log(N) = log(10) = 3.322)




def fileLoader(path, wait):
    time.sleep(int(wait))
    print('now we start')

    frame_clean = pd.read_csv(path)
    colDropped = ['serial', 'username', 'timeframe', 'noEdits']
    print('dataset loaded')

    resultsKmeans = {}

    for n in range(2,9):
        label_array = []
        resultsAll = []
        for num in range(1, 10):
            labelSample = []
            frame_sample = frame_clean.sample(frac=0.8)
            kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_sample.drop(colDropped, axis=1))
            labels = kmeans.labels_
            frame_sample['labels'] = labels
            # frameTest = np.array(frame_sample.loc[frame_sample['labels'] == 0,]['noEdits'],
            #                       frame_sample.loc[frame_sample['labels'] == 1,]['noEdits'])
            # F, p = stats.f_oneway(frame_sample.loc[frame_sample['labels'] == 0,]['noBatchEdits'],
            #                       frame_sample.loc[frame_sample['labels'] == 1,]['noBatchEdits'], frame_sample.loc[frame_sample['labels'] == 2,]['noBatchEdits'],
            #                       frame_sample.loc[frame_sample['labels'] == 3,]['noBatchEdits'])
            for g in range(0, n):
                listSerials= frame_sample['serial'].loc[frame_sample['labels'] == g]
                labelSample.append(list(listSerials))
            label_array.append(labelSample)
        for i in label_array:
            for j in label_array:
                IV = variation_of_information(i, j)
                resultsAll.append(IV)
        resultsKmeans[str(n)] = resultsAll

    # kAvg = {}
    # for key in resultsKmeans:
    #     listres = resultsKmeans[key]
    #     res = np.mean(listres)
    #     rstd = np.std(listres)
    #     kAvg[key] = (res, rstd)

    print('VI computed')

    # with open('kmeansAvg.txt', 'w') as f:
    #     f.write(str(kAvg))
    #     f.close()

    resultSscore ={}
    for n in range(2, 9):
        resultsAll = []
        for num in range(1, 15):
            labelSample = []
            kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_clean.drop(colDropped, axis=1))
            labels = kmeans.labels_
            try:
                sscore = metrics.silhouette_score(frame_clean.drop(colDropped, axis=1), labels, sample_size=20000, metric='euclidean')
            except ValueError:
                sscore = 'NA'
        # print(n, sscore)
            resultsAll.append(sscore)
        resultSscore[str(n)] = resultsAll

    with open('kmeansscore.txt', 'w') as f:
        f.write(str(resultSscore))
        f.close()

    print('sscore done')

    X = np.array(frame_clean.drop(colDropped, axis=1))
    gaps, s_k, K = gapkmean.gap_statistic(X, refs=None, B=150, K=range(2, 9), N_init=10)
    bestKValue, things = gapkmean.find_optimal_k(gaps, s_k, K)

    with open('allResults_new.txt', 'w') as f:
        f.write('VI scores')
        f.write('\n')
        for key in resultsKmeans:
            listres = resultsKmeans[key]
            f.write('{'+str(key) + ':'+str(listres)+'},')
        f.write('\n')
        f.write('silhouette scores')
        f.write('\n')
        f.write(str(resultSscore))
        f.write('\n')
        f.write('gaps: ' + str(gaps))
        f.write('\n')
        f.write(str(s_k))
        f.write('\n')
        f.write('best K: ' + str(bestKValue))
        f.write('\n')
        f.write(str(things))
        f.close()
    print('gap done')



# pca = PCA(n_components=2)
# pca.fit(frame_clean.drop('serial'))
# frame_pca = pca.fit_transform(frame_clean.drop('serial'))
# kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_pca)
# print(pca.explained_variance_ratio_)

def main():
    # create_table()
    # path = '/Users/alessandro/Documents/PhD/userstats'
    path = sys.argv[1]
    waitTime = sys.argv[2]
    fileLoader(path, waitTime)


if __name__ == "__main__":
    main()
