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
    # dictStats = {}
    # conn = get_db_params()
    # cur = conn.cursor()

    for i in range(13, 18):
        for j in range(1, 10):
            date = "20" + str(i) + "-0" + str(j) + "-01"
            if j == 1:
                yr = i-1
                datePrev = "20" + str(yr) + "-12-01"
            else:
                datePrev = "20" + str(i) + "-0" + str(j-1) + "-01"

            print(date)

            try:

                queryStart = """
                SELECT * INTO timetable_temp FROM revisionData_201710 WHERE (timestamp > '"""+ datePrev + """ 00:00:00' AND  timestamp < '"""+ date + """ 00:00:00');
                """

                conn = get_db_params()
                cur = conn.cursor()
                cur.execute(queryStart)
                cur.close()
                conn.commit()

                queryBig = """
                    WITH revTempo AS (SELECT itemid, revid, timestamp, username FROM timetable_temp
                    WHERE (username NOT IN (SELECT bot_name FROM bot_list)
                    AND username !~ '([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))'))
SELECT username, COUNT(*) AS noEdits, COUNT(DISTINCT itemid) AS itemDiv, (COUNT(*)/COUNT(DISTINCT itemid)::float) AS editRatio
FROM revTempo
GROUP BY username;
                """
                #EXTRACT(EPOCH FROM ('2016-10-01 00:00:00'::timestamp - MIN(timestamp))) AS oldEdit,

                # print(query)
                df = pd.DataFrame()
                for chunk in pd.read_sql(queryBig, con=conn, chunksize=1000):
                    df = df.append(chunk)
                #columns: username, noEdits, itemDiv, editRatio
                df['timeframe'] = date

                queryOntoedit="""
    SELECT username, COUNT(*) AS noOntoedit
    FROM (SELECT * FROM timetable_temp WHERE itemId IN (SELECT DISTINCT statvalue FROM tempData) OR itemId IN (SELECT DISTINCT itemId FROM tempData WHERE statproperty != 'P31')) poopi
    GROUP BY username;
                """

                # print(query)
                df_ontoedits = pd.DataFrame()
                for chunk in pd.read_sql(queryOntoedit, con=conn, chunksize=1000):
                    df_ontoedits = df_ontoedits.append(chunk)
                #columns: username, noOntoedit

                df = df.merge(df_ontoedits, how='left')

                queryPropedit="""
    SELECT username, COUNT(*) AS noPropEdits
    FROM timetable_temp WHERE itemId ~* '[P][0-9]{1,}'
    GROUP BY username;
                """

                # print(query)
                df_Propedits = pd.DataFrame()
                for chunk in pd.read_sql(queryPropedit, con=conn, chunksize=1000):
                    df_Propedits = df_Propedits.append(chunk)
                #columns: username, noPropEdits

                df = df.merge(df_Propedits, how='left')

                queryCommedit="""
    SELECT user_name AS username, COUNT(*) AS noCommEdits
    FROM revision_pages_201710 WHERE (time_stamp > '"""+ datePrev + """ 00:00:00' AND  time_stamp < '"""+ date + """ 00:00:00') AND (user_name NOT IN (SELECT bot_name FROM bot_list))
    AND item_id !~* 'Property:P*'
    GROUP BY user_name;
                """

                # print(query)
                df_Commedits = pd.DataFrame()
                for chunk in pd.read_sql(queryCommedit, con=conn, chunksize=1000):
                    df_Commedits = df_Commedits.append(chunk)
                #columns: username, noCommEdits

                df = df.merge(df_Commedits, how='left')

                queryTaxo = """
    SELECT username, COUNT(*) AS noTaxoEdits
    FROM statementDated WHERE (timestamp > '"""+ datePrev + """ 00:00:00' AND  timestamp < '"""+ date + """ 00:00:00')
    AND username NOT IN (SELECT bot_name FROM bot_list) AND (statProperty = 'P31' or statProperty = 'P279')
    AND username !~ '([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))')
    GROUP BY username
                """

                # print(query)
                df_Taxedits = pd.DataFrame()
                for chunk in pd.read_sql(queryTaxo, con=conn, chunksize=1000):
                    df_Taxedits = df_Taxedits.append(chunk)
                #columns: username, noTaxoEdits

                df = df.merge(df_Taxedits, how='left')

                queryBatch = """
SELECT user_name AS username, COUNT(*) AS noBatchedit
FROM (SELECT * FROM revision_history_tagged WHERE automated_tool = 't'
AND (time_stamp > '"""+ datePrev + """ 00:00:00' AND  time_stamp < '"""+ date + """ 00:00:00')) AS pippo
GROUP BY user_name
                """

                # print(query)
                df_Batchedits = pd.DataFrame()
                for chunk in pd.read_sql(queryBatch, con=conn, chunksize=1000):
                    df_Batchedits = df_Batchedits.append(chunk)
                #columns: username, noBatchedit

                df = df.merge(df_Batchedits, how='left')

                fileName = "WDuserstats-" + date + ".csv"
                df.to_csv(fileName, index=False)

                queryClose = """
                DROP TABLE timetable_temp;
                """

                # conn = get_db_params()
                # cur = conn.cursor()
                cur.execute(queryClose)
                cur.close()
                conn.commit()


            except Exception as e:
                print(e, "no df available")
                queryClose = """
                DROP TABLE timetable_temp;
                """

                # conn = get_db_params()
                # cur = conn.cursor()
                cur.execute(queryClose)
                cur.close()
                conn.commit()


        for j in range(10, 13):
            date = "20" + str(i) + "-" + str(j) + "-01"
            if j == 10:
                mt = '09'
                datePrev = "20" + str(i) + "-" + mt + "-01"
            else:
                date = "20" + str(i) + "-" + str(j-1) + "-01"
            print(date)
            try:
                queryStart = """
                SELECT * INTO timetable_temp FROM revisionData_201710 WHERE (timestamp > '"""+ datePrev + """ 00:00:00' AND  timestamp < '"""+ date + """ 00:00:00');
                """

                conn = get_db_params()
                cur = conn.cursor()
                cur.execute(queryStart)
                cur.close()
                conn.commit()

                queryBig = """
                    WITH revTempo AS (SELECT itemid, revid, timestamp, username FROM timetable_temp
                    WHERE (username NOT IN (SELECT bot_name FROM bot_list)
                    AND username !~ '([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))'))
SELECT username, COUNT(*) AS noEdits, COUNT(DISTINCT itemid) AS itemDiv, (COUNT(*)/COUNT(DISTINCT itemid)::float) AS editRatio
FROM revTempo
GROUP BY username;
                """
                #EXTRACT(EPOCH FROM ('2016-10-01 00:00:00'::timestamp - MIN(timestamp))) AS oldEdit,

                # print(query)
                df = pd.DataFrame()
                for chunk in pd.read_sql(queryBig, con=conn, chunksize=1000):
                    df = df.append(chunk)
                #columns: username, noEdits, itemDiv, editRatio
                df['timeframe'] = date

                queryOntoedit="""
    SELECT username, COUNT(*) AS noOntoedit
    FROM (SELECT * FROM timetable_temp WHERE itemId IN (SELECT DISTINCT statvalue FROM tempData) OR itemId IN (SELECT DISTINCT itemId FROM tempData WHERE statproperty != 'P31')) poopi
    GROUP BY username;
                """

                # print(query)
                df_ontoedits = pd.DataFrame()
                for chunk in pd.read_sql(queryOntoedit, con=conn, chunksize=1000):
                    df_ontoedits = df_ontoedits.append(chunk)
                #columns: username, noOntoedit

                df = df.merge(df_ontoedits, how='left')

                queryPropedit="""
    SELECT username, COUNT(*) AS noPropEdits
    FROM timetable_temp WHERE itemId ~* '[P][0-9]{1,}'
    GROUP BY username;
                """

                # print(query)
                df_Propedits = pd.DataFrame()
                for chunk in pd.read_sql(queryPropedit, con=conn, chunksize=1000):
                    df_Propedits = df_Propedits.append(chunk)
                #columns: username, noPropEdits

                df = df.merge(df_Propedits, how='left')

                queryCommedit="""
    SELECT user_name AS username, COUNT(*) AS noCommEdits
    FROM revision_pages_201710 WHERE (time_stamp > '"""+ datePrev + """ 00:00:00' AND  time_stamp < '"""+ date + """ 00:00:00') AND (user_name NOT IN (SELECT bot_name FROM bot_list))
    AND item_id !~* 'Property:P*'
    GROUP BY user_name;
                """

                # print(query)
                df_Commedits = pd.DataFrame()
                for chunk in pd.read_sql(queryCommedit, con=conn, chunksize=1000):
                    df_Commedits = df_Commedits.append(chunk)
                #columns: username, noCommEdits

                df = df.merge(df_Commedits, how='left')

                queryTaxo = """
    SELECT username, COUNT(*) AS noTaxoEdits
    FROM statementDated WHERE (timestamp > '"""+ datePrev + """ 00:00:00' AND  timestamp < '"""+ date + """ 00:00:00')
    AND username NOT IN (SELECT bot_name FROM bot_list) AND (statProperty = 'P31' or statProperty = 'P279')
    AND username !~ '([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))')
    GROUP BY username
                """

                # print(query)
                df_Taxedits = pd.DataFrame()
                for chunk in pd.read_sql(queryTaxo, con=conn, chunksize=1000):
                    df_Taxedits = df_Taxedits.append(chunk)
                #columns: username, noTaxoEdits

                df = df.merge(df_Taxedits, how='left')

                queryBatch = """
SELECT user_name AS username, COUNT(*) AS noBatchedit
FROM (SELECT * FROM revision_history_tagged WHERE automated_tool = 't'
AND (time_stamp > '"""+ datePrev + """ 00:00:00' AND  time_stamp < '"""+ date + """ 00:00:00')) AS pippo
GROUP BY user_name
                """

                # print(query)
                df_Batchedits = pd.DataFrame()
                for chunk in pd.read_sql(queryBatch, con=conn, chunksize=1000):
                    df_Batchedits = df_Batchedits.append(chunk)
                #columns: username, noBatchedit

                df = df.merge(df_Batchedits, how='left')

                fileName = "WDuserstats-" + date + ".csv"
                df.to_csv(fileName, index=False)

                queryClose = """
                DROP TABLE timetable_temp;
                """

                # conn = get_db_params()
                # cur = conn.cursor()
                cur.execute(queryClose)
                cur.close()
                conn.commit()

            except Exception as e:
                print(e, "no df available")
                queryClose = """
                DROP TABLE timetable_temp;
                """

                # conn = get_db_params()
                # cur = conn.cursor()
                cur.execute(queryClose)
                cur.close()
                conn.commit()



    # try:
    #     pickle_out = open("WDdata.pickle", "wb")
    #     pickle.dump(dictStats, pickle_out)
    #     pickle_out.close()
    # except:
    #     print("suca")

def main():
    # create_table()
    queryexecutor()


if __name__ == "__main__":
    main()
