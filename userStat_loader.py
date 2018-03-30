import pandas as pd
import psycopg2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import glob

# -*- coding: utf-8 -*-
import os
import sys
import copy

path = '/Users/alessandro/Documents/PhD/userstats'



def fileLoader(path):
    allFiles = glob.glob(path + "/WDuserstats*")
    frame = pd.DataFrame()
    list_ = []

    #bots
    bot_list_file = path + '/bot_list.csv'
    bot_list = pd.read_csv(bot_list_file)

    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)

        list_.append(df)
    frame = pd.concat(list_)
    frame.columns = ['username', 'noEdits', 'noItems', 'noOntoEdits', 'noPropEdits', 'noCommEdits', 'noTaxoEdits',
                  'noBatchEdits', 'minTime', 'timeframe', 'userAge']
    frame = frame.loc[~frame['username'].str.match(r'([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))', case=False),]

    frame = frame.loc[~frame['username'].isin(bot_list['bot_name']),]
    frame = frame.set_index('username')
    frame = frame.drop(['minTime'], axis=1)
    frame_grouped = frame.groupby('timeframe')
    zscore = lambda x: (x - x.mean()) / x.std()
    frame_norm = frame.groupby('timeframe').transform(zscore)
    frame_clean = frame_norm[frame_norm.notnull()]
    frame_clean = frame_clean.replace([np.inf, -np.inf], np.nan)
    frame_clean = frame_clean.fillna(0)


    for n in range(2,9):
        kmeans = KMeans(n_clusters=n, random_state=32).fit(frame_sample)
        labels = kmeans.labels_
        sscore = metrics.silhouette_score(frame_sample, labels, metric='euclidean')
        print(n, sscore)



def main():
    # create_table()
    fileLoader()


if __name__ == "__main__":
    main()
