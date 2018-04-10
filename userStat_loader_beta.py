import pandas as pd
# import psycopg2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import glob
from scipy import stats
# from sklearn.decomposition import PCA
from pyitlib import discrete_random_variable as drv
import string
import matplotlib
import matplotlib.ticker as ticker

# matplotlib.use('WX')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates

# from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
# from gap_statistic import OptimalK
# from sklearn.datasets.samples_generator import make_blobs
# import random

# -*- coding: utf-8 -*-
import os
import sys
import copy

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




def fileLoader(path):
    allFiles = glob.glob(path + "/WDuserstats_last*")
    # frame = pd.DataFrame()
    list_ = []

    #bots
    bot_list_file = path + '/bot_list.csv'
    bot_list = pd.read_csv(bot_list_file)

    # admin
    admin_list_file = path + '/admin_list.csv'
    admin_list = pd.read_csv(admin_list_file)
    admin_list.start_date = pd.to_datetime(admin_list.start_date)
    admin_list.end_date = pd.to_datetime(admin_list.end_date)

    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)

        list_.append(df)
    frame = pd.concat(list_)
    frame.columns = ['username', 'noEdits', 'noItems', 'noOntoEdits', 'noPropEdits', 'noCommEdits', 'noTaxoEdits',
                  'noBatchEdits', 'minTime', 'timeframe', 'userAge']
    frame = frame.set_index('username')
    frame = frame.drop(['minTime'], axis=1)
    frame['editNorm'] = frame['noEdits']
    colN = ['editNorm', 'noTaxoEdits', 'noOntoEdits', 'noPropEdits', 'noCommEdits', 'timeframe']
    normaliser = lambda x: x / x.sum()
    frame_norm = frame[colN].groupby('timeframe').transform(normaliser)
    frame_norm['timeframe'] = frame['timeframe']
    frame_norm['noItems'] = frame['noEdits'] / frame['noItems']
    frame_norm['userAge'] = frame['userAge'] / 360
    frame_norm['noBatchEdits'] = frame['noBatchEdits'] / frame['noEdits']
    frame_norm['noEdits'] = frame['noEdits']
    # frame_norm = frame_norm.loc[frame_norm['noEdits'] >= 5,]
    frame_norm.reset_index(inplace=True)
    frame_norm['admin'] = False
    frame_norm['admin'].loc[frame_norm['username'].isin(admin_list['user_name']),] = True
    frame_anon = frame_norm.loc[frame_norm['username'].str.match(
        r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))',
        case=False),]
    frame_bots = frame_norm.loc[frame_norm['username'].isin(bot_list['bot_name']),]

    frame_norm = frame_norm.loc[~frame_norm['username'].isin(bot_list['bot_name']),]

    frame_norm = frame_norm.loc[~frame_norm['username'].str.match(r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))', case=False),]

    frame_norm = frame_norm.loc[~frame_norm['username'].isin(bot_list['bot_name']),]
    # frame_norm.drop('noEdits', axis=1, inplace=True)

    # frame_norm = frame_norm.set_index('username')

    # zscore = lambda x: (x - x.mean()) / x.std()

    # colZ = ['noEdits', 'noOntoEdits', 'noPropEdits', 'noCommEdits', 'userAge',  'timeframe']
    # frame_norm = frame[colZ].groupby('timeframe').transform(zscore)
    frame_norm = frame_norm.loc[frame_norm['timeframe'] > '2013-02-01',]
    frame_clean = frame_norm[frame_norm.notnull()]
    frame_clean = frame_clean.replace([np.inf, -np.inf], np.nan)
    frame_clean = frame_clean.fillna(0)
    frame_clean['serial'] = range(1, len(frame_clean) + 1)
    # frame_clean.set_index('timeframe', inplace=True)
    # frame_clean.index = frame_clean['serial']
    colDropped = ['noEdits', 'serial', 'username', 'timeframe']
    print('dataset loaded')

    kmeans = KMeans(n_clusters=4, n_init=10, n_jobs=-1).fit(frame_clean.drop(colDropped, axis=1))
    labels = kmeans.labels_
    frame_clean['labels'] = labels
    frame_all = pd.concat([frame_anon, frame_bots, frame_clean])
    frame_all['labels'].loc[frame_all['username'].str.match(
        r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))',
        case=False),] = 4
    frame_all['labels'].loc[frame_all['username'].isin(bot_list['bot_name']),] = 5
    frame_patterns = frame_all[['timeframe', 'labels', 'noEdits']]
    frame_patterns = frame_patterns.groupby(['timeframe', 'labels']).agg({'noEdits': 'sum'})
    frame_pcts = frame_patterns.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    frame_pcts.reset_index(inplace=True)
    frame_pcts['timeframe'] = pd.to_datetime(frame_pcts['timeframe'])
    frame_pcts = frame_pcts.loc[frame_pcts['timeframe'] > '2013-02-01',]
    print('all done')


###graph
    f3 = plt.figure(figsize=(10, 6))
    font = {'size': 12}

    matplotlib.rc('font', **font)

    ax5 = plt.subplot(111)
    ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 0,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 0,], '--')
    ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 1,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 1,], '-.')
    ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 2,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 2,], ':')
    ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 3,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 3,], '-')
    ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 4,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 4,], '-',  marker='x', markevery=0.05)
    ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 5,],
             frame_pcts['noEdits'].loc[frame_pcts['labels'] == 5,], '-', marker='^', markevery=0.05)
    ax5.grid(color='gray', linestyle='--', linewidth=.5)
    ax5.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Anonymous users', 'Bots'], loc='center left')
    ax5.set_ylabel('User activity along time (in%)')

    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # to get a tick every 15 minutes
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))  # optional formatting

    f3.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    plt.savefig('clusterUsers.eps', format='eps', transparent=True)
    print('also the graph')


    resultsKmeans = {}

    for n in range(2,9):
        label_array = []
        resultsAll = []
        for num in range(1, 15):
            labelSample = []
            frame_sample = frame_clean.sample(frac=0.8)
            kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_sample.drop(colDropped, axis = 1))
            labels = kmeans.labels_
            frame_sample['labels'] = labels
            for g in range(0, n):
                listSerials= frame_sample['serial'].loc[frame_sample['labels'] == g]
                labelSample.append(list(listSerials))
            label_array.append(labelSample)

        for i in label_array:
            for j in label_array:
                IV = variation_of_information(i, j)
                resultsAll.append(IV)
        resultsKmeans[str(n)] = resultsAll

    kAvg = {}
    for key in resultsKmeans:
        listres = resultsKmeans[key]
        res = np.mean(listres)
        rstd = np.std(listres)
        kAvg[key] = (res, rstd)

    print('VI computed')

    with open('kmeansAvg.txt', 'w') as f:
        f.write(str(kAvg))
        f.close()

    resultSscore ={}
    for n in range(2, 9):
        resultsAll = []
        for num in range(1, 6):
            labelSample = []
            kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_clean.drop(colDropped, axis=1))
            labels = kmeans.labels_
            sscore = metrics.silhouette_score(frame_clean.drop('serial'), labels, sample_size=10000, metric='euclidean')
        # print(n, sscore)
            resultsAll.append(sscore)
        resultSscore[str(n)] = resultsAll

    with open('kmeansscore.txt', 'w') as f:
        f.write(str(resultSscore))
        f.close()

    print('all done')


from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

resultsAmui = {}

for n in range(2,9):
    label_array = []
    resultsAll = []
    for num in range(1, 4):
        labelSample = []
        frame_sample = frame_clean.sample(frac=0.5)
        kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_sample.drop('serial'))
        labels = kmeans.labels_
        for g in range(0, n):
            labelSample.append(list(labels))
        label_array.append(labelSample)
    for i in label_array:
        for j in label_array:
            amui = adjusted_mutual_info_score(i, j)
            resultsAll.append(amui)
    resultsAmui[str(n)] = resultsAll



# elbow method
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(frame_clean.drop(colDropped, axis=1))
    kmeanModel.fit(frame_clean.drop('serial'))
    distortions.append(sum(np.min(cdist(frame_clean.drop(colDropped, axis=1), kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / frame_clean.drop(colDropped, axis=1).shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


#gap statistic

frame_clean['admin'] = frame_clean['admin']*1
X = frame_clean.drop('serial').as_matrix()
optimalK = OptimalK(parallel_backend='joblib')
n_clusters = optimalK(X, cluster_array=np.arange(1, 10))
print('Optimal clusters: ', n_clusters)

optimalK.gap_df.head(10)

gapDf = pd.DataFrame({'n_clusters':list(range(1,11)), 'gap_value':list(coso)})

# plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
plt.plot(range(1,11), coso, linewidth=3)

plt.scatter(gapDf[gapDf.n_clusters == n_clusters].n_clusters,
            gapDf[gapDf.n_clusters == n_clusters].gap_value, s=250, c='r')
# plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
#             optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()



import scipy
import scipy.cluster.vq
import scipy.spatial.distance

dst = scipy.spatial.distance.euclidean


def gap(data, refs=None, nrefs=20, ks=range(1, 11)):
    """
    Compute the Gap statistic for an nxm dataset in data.
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.
    Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape = data.shape
    if refs == None:


        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops - bots))

        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i] * dists + bots
    else:
        rands = refs

    gaps = scipy.zeros((len(ks),))
    for (i, k) in enumerate(ks):
        (kmc, kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc, kml) = scipy.cluster.vq.kmeans2(rands[:, :, j], k)
            refdisps[j] = sum([dst(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])
        # gaps[i] = scipy.log(scipy.mean(refdisps)) - scipy.log(disp)
        gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
    return gaps

#
# def cluster_points(X, mu):
#     clusters = {}
#     for x in X:
#         bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
#                          for i in enumerate(mu)], key=lambda t: t[1])[0]
#         try:
#             clusters[bestmukey].append(x)
#         except KeyError:
#             clusters[bestmukey] = [x]
#     return clusters
#
#
# def reevaluate_centers(mu, clusters):
#     newmu = []
#     keys = sorted(clusters.keys())
#     for k in keys:
#         newmu.append(np.mean(clusters[k], axis=0))
#     return newmu
#
#
# def has_converged(mu, oldmu):
#     return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
#
#
# def find_centers(X, K):
#     # Initialize to K random centers
#     oldmu = random.sample(list(X), K)
#     mu = random.sample(list(X), K)
#     while not has_converged(mu, oldmu):
#         oldmu = mu
#         # Assign all points in X to clusters
#         clusters = cluster_points(X, mu)
#         # Reevaluate centers
#         mu = reevaluate_centers(oldmu, clusters)
#     return (mu, clusters)
#
# def bounding_box(X):
#     xmin, xmax = min(X, key=lambda a: a[0])[0], max(X, key=lambda a: a[0])[0]
#     ymin, ymax = min(X, key=lambda a: a[1])[1], max(X, key=lambda a: a[1])[1]
#     return (xmin, xmax), (ymin, ymax)
#
#
#
# from numpy import zeros
#
# def gap_statistic(X):
#     (xmin, xmax), (ymin, ymax) = bounding_box(X)
#     # Dispersion for real distribution
#     ks = range(1, 10)
#     Wks = zeros(len(ks))
#     Wkbs = zeros(len(ks))
#     sk = zeros(len(ks))
#     for indk, k in enumerate(ks):
#         mu, clusters = find_centers(X, k)
#         Wks[indk] = np.log(Wk(mu, clusters))
#         # Create B reference datasets
#         B = 10
#         BWkbs = zeros(B)
#         for i in range(B):
#             Xb = []
#             for n in range(len(X)):
#                 Xb.append([random.uniform(xmin, xmax),
#                            random.uniform(ymin, ymax)])
#             Xb = np.array(Xb)
#             mu, clusters = find_centers(Xb, k)
#             BWkbs[i] = np.log(Wk(mu, clusters))
#         Wkbs[indk] = sum(BWkbs) / B
#         sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk]) ** 2) / B)
#     sk = sk * np.sqrt(1 + 1 / B)
#     return (ks, Wks, Wkbs, sk)
#
# ks, logWks, logWkbs, sk = gap_statistic(X)
# frameTest = np.array(frame_sample.loc[frame_sample['labels'] == 0,]['noEdits'],
#                      frame_sample.loc[frame_sample['labels'] == 1,]['noEdits'])
#
anovaDict ={}
for col in frame_clean.drop(['serial']).columns:
    F, p = stats.f_oneway(frame_clean.loc[frame_clean['labels'] == 0,][col],
                      frame_clean.loc[frame_clean['labels'] == 1,][col], frame_clean.loc[frame_clean['labels'] == 2,][col],
                          frame_clean.loc[frame_clean['labels'] == 3,][col])
    anovaDict[col] = {}
    anovaDict[col]['F'] = F
    anovaDict[col]['p'] = p

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

tukeyDict ={}
for col in frame_clean.drop(['serial']).columns:
    mc = MultiComparison(frame_clean[col], frame_clean['labels'])
    result = mc.tukeyhsd()
    print(col)
    print(result)
    print(mc.groupsunique)

# pca = PCA(n_components=2)
# pca.fit(frame_clean.drop('serial'))
# frame_pca = pca.fit_transform(frame_clean.drop('serial'))
# kmeans = KMeans(n_clusters=n, n_init=10, n_jobs=-1).fit(frame_pca)
# print(pca.explained_variance_ratio_)

def main():
    # create_table()
    path = '/Users/alessandro/Documents/PhD/userstats'
    # path = sys.argv[1]
    fileLoader(path)


if __name__ == "__main__":
    main()
