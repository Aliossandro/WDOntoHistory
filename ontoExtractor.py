import pandas as pd
import psycopg2
import pickle
import numpy as np
# counterS = 0
# global counterS
# global valGlob
# from sqlalchemy import create_engine

# -*- coding: utf-8 -*-
import os
import sys
import copy

# fileName = '/Users/alessandro/Documents/PhD/OntoHistory/WDTaxo_October2014.csv'

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph.keys():
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
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

    queryStatData = """CREATE TABLE IF NOT EXISTS statementDated AS (SELECT p.itemid, p.statproperty, p.statvalue, p.statementid, p.revid, t.timestamp
    FROM statementsData_201710 p LEFT JOIN revisionData_201710 t ON p.revid = t.revid);"""


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

    for i in range(13, 18):
        for j in range(1, 10):
            date = "20" + str(i) + "-0" + str(j) + "-01"

            try:
                dictStats[date] = {}

                query = """
                    SELECT * FROM tempData WHERE tS < '""" + date + """ 00:00:00';
                """

                # print(query)
                df = pd.DataFrame()
                for chunk in pd.read_sql(query, con=conn, chunksize=500000):
                    df = df.append(chunk)


                df = df[df['statvalue'] != 'deleted']
                idx = df.groupby(['statementid'])['revid'].transform(max) == df['revid']
                dfClean = df[idx]
                fileName = "WDHierarchy-" + date + ".csv"
                dfClean.to_csv(fileName, index=False)

                # unique P279 and P31
                uniqueClasses = dfClean['statvalue'].nunique()
                dictStats[date]['uniqueClasses'] = uniqueClasses
                uniqueAll = dfClean.groupby('statproperty')['statvalue'].nunique()
                dictStats[date]['P279'] = uniqueAll['P279']
                dictStats[date]['P31'] = uniqueAll['P31']


                ### No. classes
                dfClean['statvalue'] = dfClean['statvalue'].apply(lambda x: str(x))
                dfClean['itemid'] = dfClean['itemid'].apply(lambda x: str(x))
                subClasses = list(dfClean['itemid'][dfClean['statproperty'] == "P279"].unique())
                classesList = list(dfClean['statvalue'].unique())
                rootClasses = [i for i in classesList if i not in subClasses]
                instanceOf = list(dfClean['statvalue'][dfClean['statproperty'] == 'P31'].unique())
                instanceOf = [j for j in instanceOf if j not in rootClasses]
                leafClasses = list(dfClean['itemid'][(dfClean['statproperty'] == 'P279') & (~dfClean['itemid'].isin(dfClean['statvalue']))].unique())
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

                dictStats[date]['cRichness'] = classesWInstances/noClasses
                instanceList = [classCountNew[k] for k in classCountNew.keys()]
                dictStats[date]['avgPop'] = np.mean(instanceList)
                dictStats[date]['medianPop'] = np.median(instanceList)
                dictStats[date]['quantilePop'] = (np.percentile(instanceList, 25), np.percentile(instanceList, 50), np.percentile(instanceList, 75))

                ### inheritance richness
                classCountSub = classCount['P279'].to_dict()

                for cl in classesList:
                    if cl not in classCountSub.keys():
                        classCountSub[cl] = 0

                inheritanceList = [classCountSub[z] for z in classCountSub.keys()]
                dictStats[date]['iRichness'] = np.mean(inheritanceList)
                dictStats[date]['medianInheritance'] = np.median(inheritanceList)
                dictStats[date]['quantileInheritance'] = (np.percentile(inheritanceList, 25), np.percentile(inheritanceList, 50), np.percentile(inheritanceList, 75))

                ### Explicit depth
                x = dfClean.groupby(['itemid', 'statproperty'])['statvalue'].unique()

                uniquePerClass = x.to_frame()
                uniquePerClass.reset_index(inplace=True)
                uniqueSuperClasses = uniquePerClass[uniquePerClass['statproperty'] == 'P279']
                uniqueSuperClasses.drop('statproperty', axis=1, inplace=True)
                uniqueSuperClasses['statvalue'] = uniqueSuperClasses['statvalue'].apply(lambda c: c.tolist())
                uniqueDict = uniqueSuperClasses.set_index('itemid').T.to_dict('list')

                for key in uniqueDict.keys():
                    uniqueDict[key] = uniqueDict[key][0]

                allPaths = []
                for i in leafClasses:
                    for j in rootClasses:
                        pathLength = find_all_paths(uniqueDict, i, j)
                        allPaths += pathLength

                allPaths = [len(i) for i in allPaths]
                dictStats[date]['maxDepth'] = max(allPaths)
                dictStats[date]['avgDepth'] = np.mean(allPaths)
                dictStats[date]['medianDepth'] = np.median(allPaths)
                dictStats[date]['quantileDepth'] = (np.percentile(allPaths, 25), np.percentile(allPaths, 50),
                 np.percentile(allPaths, 75))

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

                    dfRich = dfRich[dfRich['statvalue'] != 'deleted']
                    idx = dfRich.groupby(['statementid'])['revid'].transform(max) == df['revid']
                    dfRichClean = dfRich[idx]
                    richAll = dfRichClean.groupby('statproperty')['statvalue'].nunique()
                    dictStats[date]['relRichness'] = (richAll.sum() - richAll['P279'])/richAll.sum()
                except:
                    dictStats[date]['relRichness'] = 'NA'

            except:
                print("no df available")

            try:
                query2 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId !~* 'P[0-9]{1,}') AS fs;"""
                # print(query)
                dfIndiv = pd.DataFrame()
                for chunk in pd.read_sql(query2, con=conn, chunksize=500000):
                    dfIndiv = dfIndiv.append(chunk)

                fileName = "WDIndiv-" + date + ".csv"
                dfIndiv.to_csv(fileName, index=False)
            except:
                print("no df available")

            try:
                query3 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId ~* 'P[0-9]{1,}') AS fs;"""
                # print(query)
                dfProp = pd.DataFrame()
                for chunk in pd.read_sql(query3, con=conn, chunksize=500000):
                    dfProp = dfProp.append(chunk)

                fileName = "WDProp-" + date + ".csv"
                dfProp.to_csv(fileName, index=False)

                ### No. properties
                dictStats[date]['noProps'] = dfProp['itemid'].nunique()

                ### No. statements per property
                try:
                    queryProps = """
                    SELECT statproperty, COUNT(*) AS propuse FROM (SELECT * FROM statementDated WHERE  timestamp < '""" + date + """ 00:00:00') AS moo GROUP BY statproperty;
                    """
                    dfPropUse = pd.read_sql(queryProps, con=conn)
                    fileName = "WDPropUse-" + date + ".csv"
                    dfPropUse.to_csv(fileName, index=False)

                    propUseCount = list(dfPropUse['propuse'])

                    dictStats[date]['noPropUseAvg'] = np.mean(propUseCount)
                    dictStats[date]['noPropUseMedian'] = np.median(propUseCount)
                    dictStats[date]['noPropUseMax'] = max(propUseCount)
                    dictStats[date]['noPropUseMin'] = min(propUseCount)
                    dictStats[date]['noPropUseQuant'] = (np.percentile(propUseCount, 25), np.percentile(propUseCount, 50),
                     np.percentile(propUseCount, 75))
                except:
                    print("propuse not available")

            except:
                print("no df available")


        for j in range(10, 13):
            date = "20" + str(i) + "-" + str(j) + "-01"
            try:
                dictStats[date] = {}
                query = """
                                SELECT * FROM tempData WHERE tS < '""" + date + """ 00:00:00';
                            """
                df = pd.DataFrame()
                for chunk in pd.read_sql(query, con=conn, chunksize=50000):
                    df = df.append(chunk)

                df = df[df['statvalue'] != 'deleted']
                idx = df.groupby(['statementid'])['revid'].transform(max) == df['revid']
                dfClean = df[idx]
                fileName = "WDHierarchy-" + date + ".csv"
                dfClean.to_csv(fileName, index=False)

                # unique P279 and P31
                uniqueClasses = dfClean['statvalue'].nunique()
                dictStats[date]['uniqueClasses'] = uniqueClasses
                uniqueAll = dfClean.groupby('statproperty')['statvalue'].nunique()
                dictStats[date]['P279'] = uniqueAll['P279']
                dictStats[date]['P31'] = uniqueAll['P31']

                ### No. classes
                dfClean['statvalue'] = dfClean['statvalue'].apply(lambda x: str(x))
                dfClean['itemid'] = dfClean['itemid'].apply(lambda x: str(x))
                subClasses = list(dfClean['itemid'][dfClean['statproperty'] == "P279"].unique())
                classesList = list(dfClean['statvalue'].unique())
                rootClasses = [i for i in classesList if i not in subClasses]
                instanceOf = list(dfClean['statvalue'][dfClean['statproperty'] == 'P31'].unique())
                instanceOf = [j for j in instanceOf if j not in rootClasses]
                leafClasses = list(dfClean['itemid'][(dfClean['statproperty'] == 'P279') & (
                    ~dfClean['itemid'].isin(dfClean['statvalue']))].unique())
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

                dictStats[date]['cRichness'] = classesWInstances / noClasses
                instanceList = [classCountNew[k] for k in classCountNew.keys()]
                dictStats[date]['avgPop'] = np.mean(instanceList)
                dictStats[date]['medianPop'] = np.median(instanceList)
                dictStats[date]['quantilePop'] = (
                np.percentile(instanceList, 25), np.percentile(instanceList, 50), np.percentile(instanceList, 75))

                ### inheritance richness
                classCountSub = classCount['P279'].to_dict()

                for cl in classesList:
                    if cl not in classCountSub.keys():
                        classCountSub[cl] = 0

                inheritanceList = [classCountSub[z] for z in classCountSub.keys()]
                dictStats[date]['iRichness'] = np.mean(inheritanceList)
                dictStats[date]['medianInheritance'] = np.median(inheritanceList)
                dictStats[date]['quantileInheritance'] = (
                np.percentile(inheritanceList, 25), np.percentile(inheritanceList, 50),
                np.percentile(inheritanceList, 75))

                ### Explicit depth
                x = dfClean.groupby(['itemid', 'statproperty'])['statvalue'].unique()

                uniquePerClass = x.to_frame()
                uniquePerClass.reset_index(inplace=True)
                uniqueSuperClasses = uniquePerClass[uniquePerClass['statproperty'] == 'P279']
                uniqueSuperClasses.drop('statproperty', axis=1, inplace=True)
                uniqueSuperClasses['statvalue'] = uniqueSuperClasses['statvalue'].apply(lambda c: c.tolist())
                uniqueDict = uniqueSuperClasses.set_index('itemid').T.to_dict('list')

                for key in uniqueDict.keys():
                    uniqueDict[key] = uniqueDict[key][0]

                allPaths = []
                for i in leafClasses:
                    for j in rootClasses:
                        pathLength = find_all_paths(uniqueDict, i, j)
                        allPaths += pathLength

                allPaths = [len(i) for i in allPaths]
                dictStats[date]['maxDepth'] = max(allPaths)
                dictStats[date]['avgDepth'] = np.mean(allPaths)
                dictStats[date]['medianDepth'] = np.median(allPaths)
                dictStats[date]['quantileDepth'] = (np.percentile(allPaths, 25), np.percentile(allPaths, 50),
                                                    np.percentile(allPaths, 75))

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

                    dfRich = dfRich[dfRich['statvalue'] != 'deleted']
                    idx = dfRich.groupby(['statementid'])['revid'].transform(max) == df['revid']
                    dfRichClean = dfRich[idx]
                    richAll = dfRichClean.groupby('statproperty')['statvalue'].nunique()
                    dictStats[date]['relRichness'] = (richAll.sum() - richAll['P279']) / richAll.sum()
                except:
                    dictStats[date]['relRichness'] = 'NA'

            except:
                print("no df available")

            try:
                query2 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId !~* 'P[0-9]{1,}') AS fs;"""
                # print(query)
                dfIndiv = pd.DataFrame()
                for chunk in pd.read_sql(query2, con=conn, chunksize=500000):
                    dfIndiv = dfIndiv.append(chunk)

                fileName = "WDIndiv-" + date + ".csv"
                dfIndiv.to_csv(fileName, index=False)
            except:
                print("no df available")

            try:
                query3 = """ SELECT DISTINCT itemId FROM (SELECT itemId, (timestamp::timestamp) FROM revisionData_201710 WHERE timestamp < '""" + date + """ 00:00:00' AND itemId ~* 'P[0-9]{1,}') AS fs;"""
                # print(query)
                dfProp = pd.DataFrame()
                for chunk in pd.read_sql(query3, con=conn, chunksize=500000):
                    dfProp = dfProp.append(chunk)

                fileName = "WDProp-" + date + ".csv"
                dfProp.to_csv(fileName, index=False)
                ### No. properties
                dictStats[date]['noProps'] = dfProp['itemid'].nunique()

                ### No. statements per property
                try:
                    queryProps = """
                                    SELECT statproperty, COUNT(*) AS propuse FROM (SELECT * FROM statementDated WHERE  timestamp < '""" + date + """ 00:00:00') AS moo GROUP BY statproperty;
                                    """
                    dfPropUse = pd.read_sql(queryProps, con=conn)
                    fileName = "WDPropUse-" + date + ".csv"
                    dfPropUse.to_csv(fileName, index=False)

                    propUseCount = list(dfPropUse['propuse'])

                    dictStats[date]['noPropUseAvg'] = np.mean(propUseCount)
                    dictStats[date]['noPropUseMedian'] = np.median(propUseCount)
                    dictStats[date]['noPropUseMax'] = max(propUseCount)
                    dictStats[date]['noPropUseMin'] = min(propUseCount)
                    dictStats[date]['noPropUseQuant'] = (
                    np.percentile(propUseCount, 25), np.percentile(propUseCount, 50),
                    np.percentile(propUseCount, 75))
                except:
                    print("propuse not available")

            except:
                print("no df available")



    try:
        pickle_out = open("WDdata.pickle", "wb")
        pickle.dump(dictStats, pickle_out)
        pickle_out.close()
    except:
        print("suca")

def main():
    create_table()
    queryexecutor()


if __name__ == "__main__":
    main()