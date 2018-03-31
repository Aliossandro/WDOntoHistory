import pandas as pd
import psycopg2
import pickle
import numpy as np
import json
from collections import defaultdict
# counterS = 0
# global counterS
# global valGlob
# from sqlalchemy import create_engine

# -*- coding: utf-8 -*-
import os
import sys
import copy

def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]
    seen.append(v)
    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths

def depthCalculator(fileName):
    dfClean = pd.read_csv(fileName)
    dfClean.drop(['statementid', 'ts', 'revid'], axis = 1, inplace=True)

    dfClean['statvalue'] = dfClean['statvalue'].apply(lambda ni: str(ni))
    dfClean['itemid'] = dfClean['itemid'].apply(lambda nu: str(nu))
    subClasses = list(dfClean['itemid'].loc[dfClean['statproperty'] == "P279",].unique())
    classesList = list(dfClean['statvalue'].unique())
    # rootClasses = [x for x in classesList if x not in subClasses]
    rootClasses = list(set(classesList) - set(subClasses))
    instanceOf = list(dfClean['statvalue'].loc[dfClean['statproperty'] == 'P31',].unique())
    # instanceOf = [k for k in instanceOf if k not in rootClasses]
    instanceOf = list(set(instanceOf) - set(rootClasses))
    leafClasses = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (~dfClean['itemid'].isin(dfClean['statvalue'])),].unique())
    shallowClasses = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (~dfClean['itemid'].isin(dfClean['statvalue'])) & (dfClean['statvalue'].isin(rootClasses)),].unique())
    # firstSub = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (dfClean['statvalue'].isin(rootClasses)),].unique())
    # twoDepth = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (~dfClean['itemid'].isin(dfClean['statvalue'])) & (~dfClean['statvalue'].isin(firstSub)),].unique())
    # deepClasses = list(set(twoDepth) - set(shallowClasses))
    # leafClasses = set(leafClasses + instanceOf)
    classesList += subClasses
    # childless classes; reduces computation time for avgDepth
    superClasses = list(dfClean['statvalue'].loc[dfClean['statproperty'] == "P279",].unique())
    childLessClasses = list(set(rootClasses) - set(superClasses))

    ###remember to add childLessClasses and shallowClasses


    ### Explicit depth
    # bibi = dfClean.groupby(['itemid', 'statproperty'])['statvalue'].unique()
    bibi = dfClean.loc[dfClean.statproperty == 'P279', ].groupby('itemid')['statvalue'].unique()

    #compute depth only for leaf classes whose hierarchy is deeper than 1
    deepClasses = list(set(leafClasses) - set(shallowClasses))
    fertileRoots = list(set(rootClasses) - set(childLessClasses))

    shallowDepth = [1] * len(shallowClasses)
    childlessDepth = [0] * len(childLessClasses)

    uniqueSuperClasses = bibi.to_frame()
    uniqueSuperClasses.reset_index(inplace=True)
    # uniqueSuperClasses = uniquePerClass.loc[uniquePerClass['statproperty'] == 'P279',]

    if len(uniqueSuperClasses.index) != 0:
        # uniqueSuperClasses.drop('statproperty', axis=1, inplace=True)
        uniqueSuperClasses['statvalue'] = uniqueSuperClasses['statvalue'].apply(lambda c: c.tolist())
        uniqueDict = uniqueSuperClasses.set_index('itemid').T.to_dict('list')

        for key in uniqueDict.keys():
            uniqueDict[key] = uniqueDict[key][0]

        classesDefaultDict = defaultdict(str, uniqueDict)
        allPaths = [p for ps in [DFS(classesDefaultDict, n) for n in set(deepClasses)] for p in ps]
        allPaths = [p for p in allPaths if p[len(p)-1] not in set(fertileRoots)]

        # allPaths = []
        # for cla in deepClasses:
        #     for clo in list(set(rootClasses) - set(childLessClasses)):
        #         pathLength = find_all_paths(uniqueDict, cla, clo)
        #         allPaths += pathLength
        # allPaths = [len(path) for path in allPaths]
        lenList = [len(p) for p in allPaths if p[len(p)-1] not in set(fertileRoots)]

        dictStats[date]['maxDepth'] = max(lenList)
        dictStats[date]['avgDepth'] = np.asscalar(np.mean(lenList))
        dictStats[date]['medianDepth'] = np.asscalar(np.median(lenList))
        dictStats[date]['quantileDepth'] = (np.asscalar(np.percentile(lenList, 25)), np.asscalar(np.percentile(lenList, 50)),
         np.asscalar(np.percentile(lenList, 75)))
    else:
        dictStats[date]['maxDepth'] = 0
        dictStats[date]['avgDepth'] = 0
        dictStats[date]['medianDepth'] = 0
        dictStats[date]['quantileDepth'] = (0,0,0)
    print('depth done')



def main():
    # create_table()
    queryexecutor()


if __name__ == "__main__":
    main()