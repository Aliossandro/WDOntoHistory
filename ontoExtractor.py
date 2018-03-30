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

# fileName = '/Users/alessandro/Documents/PhD/OntoHistory/WDTaxo_October2014.csv'

# def find_all_paths(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return [path]
#     if start not in graph.keys():
#         return []
#     paths = []
#     for node in graph[start]:
#         if node not in path:
#             newpaths = find_all_paths(graph, node, end, path)
#             for newpath in newpaths:
#                 paths.append(newpath)
#     return paths
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

# def get_max_rows(df):
#     B_maxes = df.groupby(['statementId', 'statValue']).revId.transform(min) == df['revId']
#     return df[B_maxes]

# connection parameters
def get_db_params():
    params = {
        'database': 'wikidb',
        'user': 'postgres',
        'password': 'postSonny175',
        'host': 'localhost',
        'port': '5432'
    }
    conn = psycopg2.connect(**params)
    return conn



# create table
def create_table():
    ###statement table query
    query_table = """CREATE TABLE IF NOT EXISTS tempData AS (SELECT p.itemId, p.revId, (p.timestamp::timestamp) AS tS, t.statementId, t.statProperty, t.statvalue FROM
(SELECT itemId, revId, timestamp FROM revisionData_201710) p, (SELECT revId, statementId, statProperty, statvalue FROM statementsData_201710 WHERE statProperty = 'P279' OR statProperty = 'P31') t
WHERE p.revId = t.revId)"""

    queryStatData = """CREATE TABLE IF NOT EXISTS statementDated AS (SELECT p.itemid, p.statproperty, p.statvalue, p.statementid, p.revid, t.timestamp, t.username
    FROM statementsData_201710 p LEFT JOIN revisionData_201710 t ON p.revid::int = t.revid::int);"""


    conn = None

    try:
        conn = get_db_params()
        cur = conn.cursor()
        cur.execute(query_table)
        cur.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    conn = None

    try:
        conn = get_db_params()
        cur = conn.cursor()
        cur.execute(queryStatData)
        cur.close()
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()



def queryexecutor():
    dictStats = {}
    conn = get_db_params()
    # cur = conn.cursor()

    for i in range(13, 15):
        for j in range(1, 10):
            date = "20" + str(i) + "-0" + str(j) + "-01"
            print(date)

            try:
                dictStats[date] = {}

                query = """
                    SELECT * FROM tempData WHERE tS < '""" + date + """ 00:00:00';
                """

                # print(query)
                df = pd.DataFrame()
                for chunk in pd.read_sql(query, con=conn, chunksize=100000):
                    df = df.append(chunk)

                if len(df.index) != 0:
                    df = df[df['statvalue'] != 'deleted']
                    df = df[df['statvalue'] != 'novalue']
                    df = df[df['statvalue'] != 'somevalue']
                    idx = df.groupby(['statementid'])['revid'].transform(max) == df['revid']
                    dfClean = df[idx]
                    fileName = "WDHierarchy-" + date + ".csv"
                    dfClean.to_csv(fileName, index=False)

                    # unique P279 and P31
                    uniqueClasses = dfClean['statvalue'].nunique()
                    dictStats[date]['uniqueClasses'] = uniqueClasses
                    uniqueAll = dfClean.groupby('statproperty')['statvalue'].nunique()
                    try:
                        dictStats[date]['P279'] = np.asscalar(uniqueAll['P279'])
                    except:
                        dictStats[date]['P279'] = 0
                    dictStats[date]['P31'] = np.asscalar(uniqueAll['P31'])
                    print('unique done')


                    ### No. classes
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
                    firstSub = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (dfClean['statvalue'].isin(rootClasses)),].unique())
                    twoDepth = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (~dfClean['itemid'].isin(dfClean['statvalue'])) & (~dfClean['statvalue'].isin(firstSub)),].unique())
                    deepClasses = list(set(twoDepth) - set(shallowClasses))
                    # leafClasses = set(leafClasses + instanceOf)
                    classesList += subClasses
                    dictStats[date]['noClasses'] = len(set(classesList))
                    # childless classes; reduces computation time for avgDepth
                    superClasses = list(dfClean['statvalue'].loc[dfClean['statproperty'] == "P279",].unique())
                    childLessClasses = list(set(rootClasses) - set(superClasses))
                    dictStats[date]['childLessClasses'] = len(set(childLessClasses))

                    ### No. root classes
                    dictStats[date]['noRoot'] = len(set(rootClasses))

                    ### No. leaf classes
                    dictStats[date]['noLeaf'] = len((leafClasses))
                    print('no classes done')

                    ### Avg. population metric and class richness
                    # Mean, median, and 0.25-0.75 quantiles no. instances per class
                    classCount = dfClean.groupby('statproperty')['statvalue'].value_counts()
                    classCountNew = classCount['P31'].to_dict()

                    dictStats[date]['classesWInstances'] = len(classCountNew)

                    for cl in classesList:
                        if cl not in classCountNew.keys():
                            classCountNew[cl] = 0

                    dictStats[date]['cRichness'] = len(classCountNew)/len(set(classesList))
                    instanceList = [classCountNew[l] for l in classCountNew.keys()]
                    dictStats[date]['avgPop'] = np.asscalar(np.mean(instanceList))
                    dictStats[date]['medianPop'] = np.asscalar(np.median(instanceList))
                    dictStats[date]['quantilePop'] = (np.asscalar(np.percentile(instanceList, 25)), np.asscalar(np.percentile(instanceList, 50)), np.asscalar(np.percentile(instanceList, 75)))
                    print('avg population done')

                    ### inheritance richness
                    try:
                        classCountSub = classCount['P279'].to_dict()
                    except:
                        classCountSub = {}

                    for cl in classesList:
                        if cl not in classCountSub.keys():
                            classCountSub[cl] = 0

                    inheritanceList = [classCountSub[z] for z in classCountSub.keys()]
                    dictStats[date]['iRichness'] = np.asscalar(np.mean(inheritanceList))
                    dictStats[date]['medianInheritance'] = np.asscalar(np.median(inheritanceList))
                    dictStats[date]['quantileInheritance'] = (np.asscalar(np.percentile(inheritanceList, 25)), np.asscalar(np.percentile(inheritanceList, 50)), np.asscalar(np.percentile(inheritanceList, 75)))
                    print('inheritance done')

                    ### Explicit depth
                    # bibi = dfClean.groupby(['itemid', 'statproperty'])['statvalue'].unique()
                    bibi = dfClean.loc[dfClean.statproperty == 'P279', ].groupby('itemid')['statvalue'].unique()

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
                        allPaths = [p for ps in [DFS(classesDefaultDict, n) for n in set(leafClasses)] for p in ps]

                        # allPaths = []
                        # for cla in deepClasses:
                        #     for clo in list(set(rootClasses) - set(childLessClasses)):
                        #         pathLength = find_all_paths(uniqueDict, cla, clo)
                        #         allPaths += pathLength
                        # allPaths = [len(path) for path in allPaths]

                        dictStats[date]['maxDepth'] = max(len(p) for p in allPaths)
                        dictStats[date]['avgDepth'] = np.asscalar(np.mean(len(p) for p in allPaths))
                        dictStats[date]['medianDepth'] = np.asscalar(np.median(len(p) for p in allPaths))
                        dictStats[date]['quantileDepth'] = (np.asscalar(np.percentile((len(p) for p in allPaths), 25)), np.asscalar(np.percentile((len(p) for p in allPaths), 50)),
                         np.asscalar(np.percentile((len(p) for p in allPaths), 75)))
                    else:
                        dictStats[date]['maxDepth'] = 0
                        dictStats[date]['avgDepth'] = 0
                        dictStats[date]['medianDepth'] = 0
                        dictStats[date]['quantileDepth'] = (0,0,0)
                    print('depth done')


                    ### Relationship richness
                    try:
                        queryRich = """
                                            SELECT itemid, statproperty, statvalue, statementid, revid, timestamp FROM statementDated WHERE  timestamp < '""" + date + """ 00:00:00'
                                            AND ((itemid IN (SELECT DISTINCT itemId FROM tempData WHERE statproperty != 'P31' WHERE  timestamp < '""" + date + """ 00:00:00'))
                                            OR (itemid IN (SELECT DISTINCT statvalue FROM tempData WHERE  timestamp < '""" + date + """ 00:00:00')));
                                        """
                        # print(query)
                        dfRich = pd.DataFrame()
                        for chunk in pd.read_sql(queryRich, con=conn, chunksize=10000):
                            dfRich = dfRich.append(chunk)

                        dfRich = dfRich.loc[dfRich['statvalue'] != 'deleted',]
                        idx = dfRich.groupby(['statementid'])['revid'].transform(max) == dfRich['revid']
                        dfRichClean = dfRich[idx]
                        richAll = dfRichClean.groupby('statproperty')['statvalue'].nunique()
                        dictStats[date]['relRichness'] = (richAll.sum() - richAll['P279'])/richAll.sum()
                    except:
                        dictStats[date]['relRichness'] = 'NA'
                else:
                    dictStats[date]['P279'] = 0
                    dictStats[date]['P31'] = 0
                    dictStats[date]['relRichness'] = 0
                    dictStats[date]['maxDepth'] = 0
                    dictStats[date]['avgDepth'] = 0
                    dictStats[date]['medianDepth'] = 0
                    dictStats[date]['quantileDepth'] = 0
                    dictStats[date]['iRichness'] = 0
                    dictStats[date]['medianInheritance'] = 0
                    dictStats[date]['quantileInheritance'] = 0
                    dictStats[date]['cRichness'] = 0
                    dictStats[date]['avgPop'] = 0
                    dictStats[date]['medianPop'] = 0
                    dictStats[date]['quantilePop'] = 0
                    dictStats[date]['classesWInstances'] = 0
                    dictStats[date]['noClasses'] = 0
                    ### No. root classes
                    dictStats[date]['noRoot'] = 0
                    ### No. leaf classes
                    dictStats[date]['noLeaf'] = 0

            except Exception as e:
                print(e, "no df available")

            # try:
            #     query2 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId !~* 'P[0-9]{1,}') AS fs;"""
            #     # print(query)
            #     dfIndiv = pd.DataFrame()
            #     for chunk in pd.read_sql(query2, con=conn, chunksize=500000):
            #         dfIndiv = dfIndiv.append(chunk)
            #
            #     fileName = "WDIndiv-" + date + ".csv"
            #     dfIndiv.to_csv(fileName, index=False)
            # except Exception as e:
            #     print(e, "no df available")

            try:
                query3 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId ~* 'P[0-9]{1,}') AS fs;"""
                # print(query)
                dfProp = pd.DataFrame()
                for chunk in pd.read_sql(query3, con=conn, chunksize=500000):
                    dfProp = dfProp.append(chunk)

                if len(dfProp.index) != 0:
                    fileName = "WDProp-" + date + ".csv"
                    dfProp.to_csv(fileName, index=False)

                    ### No. properties
                    dictStats[date]['noProps'] = dfProp['itemid'].nunique()
                else:
                    dictStats[date]['noProps'] = 0

                ### No. statements per property
                try:
                    queryProps = """
                    SELECT statproperty, COUNT(*) AS propuse FROM (SELECT * FROM statementDated WHERE  timestamp < '""" + date + """ 00:00:00') AS moo GROUP BY statproperty;
                    """
                    dfPropUse = pd.read_sql(queryProps, con=conn)
                    if len(dfPropUse.index) != 0:
                        fileName = "WDPropUse-" + date + ".csv"
                        dfPropUse.to_csv(fileName, index=False)

                        propUseCount = list(dfPropUse['propuse'])

                        dictStats[date]['noPropUseAvg'] = np.asscalar(np.mean(propUseCount))
                        dictStats[date]['noPropUseMedian'] = np.asscalar(np.median(propUseCount))
                        dictStats[date]['noPropUseMax'] = max(propUseCount)
                        dictStats[date]['noPropUseMin'] = min(propUseCount)
                        dictStats[date]['noPropUseQuant'] = (np.asscalar(np.percentile(propUseCount, 25)), np.asscalar(np.percentile(propUseCount, 50)),
                         np.asscalar(np.percentile(propUseCount, 75)))
                    else:
                        dictStats[date]['noPropUseAvg'] = 0
                        dictStats[date]['noPropUseMedian'] = 0
                        dictStats[date]['noPropUseMax'] = 0
                        dictStats[date]['noPropUseMin'] = 0
                        dictStats[date]['noPropUseQuant'] = 0

                except Exception as e:
                    print("propuse not available")

            except Exception as e:
                print(e, "no df available")

            with open('WDataStats_1.txt', 'w') as myfile:
                myfile.write(json.dumps(dictStats))

                myfile.close()


        for j in range(10, 13):
            date = "20" + str(i) + "-" + str(j) + "-01"
            print(date)
            try:
                dictStats[date] = {}
                query = """
                                SELECT * FROM tempData WHERE tS < '""" + date + """ 00:00:00';
                            """
                df = pd.DataFrame()
                for chunk in pd.read_sql(query, con=conn, chunksize=50000):
                    df = df.append(chunk)

                if len(df.index) != 0:
                    df = df[df['statvalue'] != 'deleted']
                    idx = df.groupby(['statementid'])['revid'].transform(max) == df['revid']
                    dfClean = df[idx]
                    fileName = "WDHierarchy-" + date + ".csv"
                    dfClean.to_csv(fileName, index=False)

                    # unique P279 and P31
                    uniqueClasses = dfClean['statvalue'].nunique()
                    dictStats[date]['uniqueClasses'] = uniqueClasses
                    uniqueAll = dfClean.groupby('statproperty')['statvalue'].nunique()
                    try:
                        dictStats[date]['P279'] = np.asscalar(uniqueAll['P279'])
                    except:
                        dictStats[date]['P279'] = 0
                    dictStats[date]['P31'] = np.asscalar(uniqueAll['P31'])

                    ### No. classes
                    dfClean['statvalue'] = dfClean['statvalue'].apply(lambda ni: str(ni))
                    dfClean['itemid'] = dfClean['itemid'].apply(lambda nu: str(nu))
                    subClasses = list(dfClean['itemid'].loc[dfClean['statproperty'] == "P279",].unique())
                    classesList = list(dfClean['statvalue'].unique())
                    # rootClasses = [x for x in classesList if x not in subClasses]
                    rootClasses = list(set(classesList) - set(subClasses))
                    instanceOf = list(dfClean['statvalue'].loc[dfClean['statproperty'] == 'P31',].unique())
                    # instanceOf = [k for k in instanceOf if k not in rootClasses]
                    instanceOf = list(set(instanceOf) - set(rootClasses))
                    leafClasses = list(dfClean['itemid'].loc[(dfClean['statproperty'] == 'P279') & (
                        ~dfClean['itemid'].isin(dfClean['statvalue'])),].unique())
                    # leafClasses = set(leafClasses + instanceOf)
                    classesList += subClasses
                    dictStats[date]['noClasses'] = len(set(classesList))

                    ### No. root classes
                    dictStats[date]['noRoot'] = len(set(rootClasses))

                    ### No. leaf classes
                    dictStats[date]['noLeaf'] = len((leafClasses))

                    ### Avg. population metric and class richness
                    # Mean, median, and 0.25-0.75 quantiles no. instances per class
                    classCount = dfClean.groupby('statproperty')['statvalue'].value_counts()
                    classCountNew = classCount['P31'].to_dict()

                    dictStats[date]['classesWInstances'] = len(classCountNew)

                    for cl in classesList:
                        if cl not in classCountNew.keys():
                            classCountNew[cl] = 0

                    dictStats[date]['cRichness'] = len(classCountNew) / len(set(classesList))
                    instanceList = [classCountNew[l] for l in classCountNew.keys()]
                    dictStats[date]['avgPop'] = np.asscalar(np.mean(instanceList))
                    dictStats[date]['medianPop'] = np.asscalar(np.median(instanceList))
                    dictStats[date]['quantilePop'] = (
                    np.asscalar(np.percentile(instanceList, 25)), np.asscalar(np.percentile(instanceList, 50)), np.asscalar(np.percentile(instanceList, 75)))

                    ### inheritance richness
                    classCountSub = classCount['P279'].to_dict()

                    for cl in classesList:
                        if cl not in classCountSub.keys():
                            classCountSub[cl] = 0

                    inheritanceList = [classCountSub[z] for z in classCountSub.keys()]
                    dictStats[date]['iRichness'] = np.asscalar(np.mean(inheritanceList))
                    dictStats[date]['medianInheritance'] = np.asscalar(np.median(inheritanceList))
                    dictStats[date]['quantileInheritance'] = (
                    np.asscalar(np.percentile(inheritanceList, 25)), np.asscalar(np.percentile(inheritanceList, 50)),
                    np.asscalar(np.percentile(inheritanceList, 75)))

                    ### Explicit depth
                    # bibi = dfClean.groupby(['itemid', 'statproperty'])['statvalue'].unique()
                    bibi = dfClean.loc[dfClean.statproperty == 'P279', ].groupby('itemid')['statvalue'].unique()

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
                        allPaths = [p for ps in [DFS(classesDefaultDict, n) for n in set(leafClasses)] for p in ps]

                        # allPaths = []
                        # for cla in deepClasses:
                        #     for clo in list(set(rootClasses) - set(childLessClasses)):
                        #         pathLength = find_all_paths(uniqueDict, cla, clo)
                        #         allPaths += pathLength
                        # allPaths = [len(path) for path in allPaths]

                        dictStats[date]['maxDepth'] = max(len(p) for p in allPaths)
                        dictStats[date]['avgDepth'] = np.asscalar(np.mean(len(p) for p in allPaths))
                        dictStats[date]['medianDepth'] = np.asscalar(np.median(len(p) for p in allPaths))
                        dictStats[date]['quantileDepth'] = (np.asscalar(np.percentile((len(p) for p in allPaths), 25)), np.asscalar(np.percentile((len(p) for p in allPaths), 50)),
                         np.asscalar(np.percentile((len(p) for p in allPaths), 75)))
                    else:
                        dictStats[date]['maxDepth'] = 0
                        dictStats[date]['avgDepth'] = 0
                        dictStats[date]['medianDepth'] = 0
                        dictStats[date]['quantileDepth'] = (0,0,0)

                    ### Relationship richness
                    try:
                        queryRich = """
                                                            SELECT itemid, statproperty, statvalue, statementid, revid, timestamp FROM statementDated WHERE  timestamp < '""" + date + """ 00:00:00'
                                                            AND ((itemid IN (SELECT itemId FROM tempData WHERE statproperty != 'P31' AND timestamp < '""" + date + """ 00:00:00'))
                                                            OR (itemid IN (SELECT statvalue FROM tempData WHERE  timestamp < '""" + date + """ 00:00:00')));
                                                        """
                        # print(query)
                        dfRich = pd.DataFrame()
                        for chunk in pd.read_sql(queryRich, con=conn, chunksize=10000):
                            dfRich = dfRich.append(chunk)

                        dfRich = dfRich[dfRich['statvalue'] != 'deleted']
                        idx = dfRich.groupby(['statementid'])['revid'].transform(max) == dfRich['revid']
                        dfRichClean = dfRich[idx]
                        richAll = dfRichClean.groupby('statproperty')['statvalue'].nunique()
                        dictStats[date]['relRichness'] = (richAll.sum() - richAll['P279']) / richAll.sum()
                    except:
                        dictStats[date]['relRichness'] = 'NA'
                else:
                    dictStats[date]['P279'] = 0
                    dictStats[date]['P31'] = 0
                    dictStats[date]['relRichness'] = 0
                    dictStats[date]['maxDepth'] = 0
                    dictStats[date]['avgDepth'] = 0
                    dictStats[date]['medianDepth'] = 0
                    dictStats[date]['quantileDepth'] = 0
                    dictStats[date]['iRichness'] = 0
                    dictStats[date]['medianInheritance'] = 0
                    dictStats[date]['quantileInheritance'] = 0
                    dictStats[date]['cRichness'] = 0
                    dictStats[date]['avgPop'] = 0
                    dictStats[date]['medianPop'] = 0
                    dictStats[date]['quantilePop'] = 0
                    dictStats[date]['classesWInstances'] = 0
                    dictStats[date]['noClasses'] = 0
                    ### No. root classes
                    dictStats[date]['noRoot'] = 0
                    ### No. leaf classes
                    dictStats[date]['noLeaf'] = 0

            except Exception as e:
                print(e, "no df available")

            # try:
            #     query2 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId !~* 'P[0-9]{1,}') AS fs;"""
            #     # print(query)
            #     dfIndiv = pd.DataFrame()
            #     for chunk in pd.read_sql(query2, con=conn, chunksize=500000):
            #         dfIndiv = dfIndiv.append(chunk)
            #
            #     fileName = "WDIndiv-" + date + ".csv"
            #     dfIndiv.to_csv(fileName, index=False)
            # except Exception as e:
            #     print(e, "no df available")

            try:
                query3 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId ~* 'P[0-9]{1,}') AS fs;"""
                # print(query)
                dfProp = pd.DataFrame()
                for chunk in pd.read_sql(query3, con=conn, chunksize=500000):
                    dfProp = dfProp.append(chunk)

                if len(dfProp.index) != 0:
                    fileName = "WDProp-" + date + ".csv"
                    dfProp.to_csv(fileName, index=False)

                    ### No. properties
                    dictStats[date]['noProps'] = dfProp['itemid'].nunique()
                else:
                    dictStats[date]['noProps'] = 0

                ### No. statements per property
                try:
                    queryProps = """
                                    SELECT statproperty, COUNT(*) AS propuse FROM (SELECT * FROM statementDated WHERE  timestamp < '""" + date + """ 00:00:00') AS moo GROUP BY statproperty;
                                    """
                    dfPropUse = pd.read_sql(queryProps, con=conn)
                    if len(dfPropUse.index) != 0:
                        fileName = "WDPropUse-" + date + ".csv"
                        dfPropUse.to_csv(fileName, index=False)

                        propUseCount = list(dfPropUse['propuse'])

                        dictStats[date]['noPropUseAvg'] = np.asscalar(np.mean(propUseCount))
                        dictStats[date]['noPropUseMedian'] = np.asscalar(np.median(propUseCount))
                        dictStats[date]['noPropUseMax'] = max(propUseCount)
                        dictStats[date]['noPropUseMin'] = min(propUseCount)
                        dictStats[date]['noPropUseQuant'] = (np.asscalar(np.percentile(propUseCount, 25)), np.asscalar(np.percentile(propUseCount, 50)),
                         np.asscalar(np.percentile(propUseCount, 75)))
                    else:
                        dictStats[date]['noPropUseAvg'] = 0
                        dictStats[date]['noPropUseMedian'] = 0
                        dictStats[date]['noPropUseMax'] = 0
                        dictStats[date]['noPropUseMin'] = 0
                        dictStats[date]['noPropUseQuant'] = 0

                except Exception as e:
                    print("propuse not available")

            except Exception as e:
                print(e, "no df available")

            with open('WDataStats_1.txt', 'w') as myfile:
                myfile.write(json.dumps(dictStats))
                myfile.close()



    # try:
    #     pickle_out = open("WDdata_1.pickle", "wb")
    #     pickle.dump(dictStats, pickle_out)
    #     pickle_out.close()
    # except:
    #     print("suca")

def main():
    # create_table()
    queryexecutor()


if __name__ == "__main__":
    main()
