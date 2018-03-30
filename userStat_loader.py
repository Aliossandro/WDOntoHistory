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
    frame['noEdits'] = frame['noEdits'].apply(lambda x: (x - x.mean()) / x.std())
    frame = frame.drop(['minTime', 'timeframe'], axis=1)

    frame = (frame - frame.mean()) / (frame.std())
    frame = frame.set_index('username')
    # frame_norm = (frame - frame.mean()) / (frame.std())
    kmeans = KMeans(n_clusters=3, random_state=32).fit(frame_norm)
    labels = kmeans.labels_
    metrics.silhouette_score(frame_sample, labels, metric='euclidean')



def main():
    # create_table()
    fileLoader()


if __name__ == "__main__":
    main()
