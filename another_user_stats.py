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
# from pyitlib import discrete_random_variable as drv
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
    allFiles = glob.glob(path + "/WDuserstats-*")
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

    frame = frame.drop(['minTime'], axis=1)
    frame['editNorm'] = frame['noEdits']
    frame_anon = frame.loc[frame['username'].str.match(
        r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))',
        case=False),]
    frame_bots = frame.loc[frame['username'].isin(bot_list['bot_name']),]

    frame = frame.loc[~frame['username'].isin(bot_list['bot_name']),]

    frame = frame.loc[~frame['username'].str.match(
        r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))',
        case=False),]

    frame = frame.loc[~frame['username'].isin(bot_list['bot_name']),]
    frame = frame.set_index('username')
    colN = ['editNorm', 'noCommEdits', 'timeframe']
    normaliser = lambda x: x / x.sum()
    frame_norm = frame[colN].groupby('timeframe').transform(normaliser)
    frame_norm['timeframe'] = frame['timeframe']
    frame_norm['noItems'] = frame['noEdits'] / frame['noItems']
    frame_norm['userAge'] = frame['userAge'] / 360
    frame_norm['noBatchEdits'] = frame['noBatchEdits'] / frame['noEdits']
    frame_norm['noTaxoEdits'] = frame['noTaxoEdits'] / frame['noEdits']
    frame_norm['noOntoEdits'] = frame['noOntoEdits'] / frame['noEdits']
    frame_norm['noPropEdits'] = frame['noPropEdits'] / frame['noEdits']
    frame_norm['noEdits'] = frame['noEdits']
    # frame_norm = frame_norm.loc[frame_norm['noEdits'] >= 5,]
    frame_norm.reset_index(inplace=True)
    frame_norm['admin'] = False
    frame_norm['admin'].loc[frame_norm['username'].isin(admin_list['user_name']),] = True

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

    kmeans = KMeans(n_clusters=2, n_init=10, n_jobs=-1).fit(frame_clean.drop(colDropped, axis=1))
    labels = kmeans.labels_
    frame_clean['labels'] = labels
    frame_all = pd.concat([frame_anon, frame_bots, frame_clean])
    frame_all['normAll'] = frame_all['noEdits']
    colZ = ['normAll', 'timeframe']
    frame_norm_all = frame_all[colZ].groupby('timeframe').transform(normaliser)
    frame_all['normAll'] = frame_norm_all['normAll']


    frame_all['labels'].loc[frame_all['username'].str.match(
        r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))',
        case=False),] = 2
    frame_all['labels'].loc[frame_all['username'].isin(bot_list['bot_name']),] = 3
    frame_patterns = frame_all[['timeframe', 'labels', 'noEdits']]
    frame_patterns = frame_patterns.groupby(['timeframe', 'labels']).agg({'noEdits': 'sum'})
    frame_pcts = frame_patterns.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    frame_pcts.reset_index(inplace=True)
    frame_pcts['timeframe'] = pd.to_datetime(frame_pcts['timeframe'])
    frame_pcts = frame_pcts.loc[frame_pcts['timeframe'] > '2013-02-01',]
    frame_all.to_csv('frameAll.csv', index=False)
    print('all done')


###graph
    # f3 = plt.figure(figsize=(10, 6))
    # font = {'size': 12}
    #
    # matplotlib.rc('font', **font)
    #
    # ax5 = plt.subplot(111)
    # ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 0,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 0,], '--')
    # ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 1,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 1,], '-.')
    # ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 2,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 2,], ':')
    # ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 3,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 3,], '-')
    # ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 4,], frame_pcts['noEdits'].loc[frame_pcts['labels'] == 4,], '-',  marker='x', markevery=0.05)
    # ax5.plot(frame_pcts['timeframe'].loc[frame_pcts['labels'] == 5,],
    #          frame_pcts['noEdits'].loc[frame_pcts['labels'] == 5,], '-', marker='^', markevery=0.05)
    # ax5.grid(color='gray', linestyle='--', linewidth=.5)
    # ax5.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Anonymous users', 'Bots'], loc='center left')
    # ax5.set_ylabel('User activity along time (in%)')
    #
    # ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # to get a tick every 15 minutes
    # ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))  # optional formatting
    #
    # f3.autofmt_xdate()
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('clusterUsers.eps', format='eps', transparent=True)
    # print('also the graph')


def main():
    # create_table()
    # path = '/Users/alessandro/Documents/PhD/userstats'
    path = sys.argv[1]
    fileLoader(path)


if __name__ == "__main__":
    main()
