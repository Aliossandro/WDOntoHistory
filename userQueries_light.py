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
def create_tables():
    ###statement table query
    bigQuery = """
    BEGIN;

CREATE TEMP TABLE batchEdits_bis AS (
SELECT user_name AS username, COUNT(*) AS noBatchedit
FROM (SELECT * FROM revision_history_tagged WHERE automated_tool = 't') AS pippo
GROUP BY user_name);

CREATE TEMP TABLE ontousers_bis AS (
SELECT username, COUNT(*) AS noOntoedit
FROM (SELECT * FROM revisionData_201710 WHERE itemId IN (SELECT DISTINCT statvalue FROM tempData) OR itemId IN (SELECT DISTINCT itemId FROM tempData WHERE statproperty != 'P31')) poopi
GROUP BY username);

CREATE TEMP TABLE propertyusers_bis AS (
SELECT username, COUNT(*) AS noPropEdits
FROM revisionData_201710 WHERE itemId ~* '[P][0-9]{1,}'
GROUP BY username);

CREATE TEMP TABLE communityUsers AS (
SELECT username, COUNT(*) AS noCommEdits
FROM revision_pages_201710 WHERE user_name NOT IN (SELECT bot_name FROM bot_list)
GROUP BY username);

CREATE TEMP TABLE taxoUsers AS (
SELECT username, COUNT(*) AS noTaxoEdits
FROM statementDated WHERE username NOT IN (SELECT bot_name FROM bot_list) AND (statProperty = 'P31' or statProperty = 'P279')
GROUP BY username);

CREATE TEMP TABLE tempProp AS (SELECT p.username, p.noedits, p.itemDiv, p.oldedit, p.editratio,
t.noPropEdits
FROM tempUserStatsClean p
LEFT JOIN propertyusers_bis t ON p.username = t.username);

CREATE TEMP TABLE tempUserData AS (SELECT p.username, p.noedits, p.itemDiv, p.oldedit, p.editratio,
p.noPropEdits, t.noOntoedit
FROM tempProp p
LEFT JOIN ontousers_bis t ON p.username = t.username);

CREATE TEMP TABLE tempUserData_bis AS (SELECT p.username, p.noedits, p.itemDiv, p.oldedit, p.editratio,
p.noPropEdits, p.noOntoedit, t.noBatchedit
FROM tempUserData p
LEFT JOIN batchEdits_bis t ON p.username = t.username);

CREATE TEMP TABLE tempUserData_ter AS (SELECT p.username, p.noedits, p.itemDiv, p.oldedit, p.editratio,
p.noPropEdits, p.noOntoedit, p.noBatchedit, t.noCommEdits
FROM tempUserData_bis p
LEFT JOIN communityUsers t ON p.username = t.username);

CREATE TABLE humanUserData AS (SELECT p.username, p.noedits, p.itemDiv, p.oldedit, p.editratio,
p.noPropEdits, p.noOntoedit, p.noBatchedit, p.noCommEdits, t.noTaxoEdits
FROM tempUserData_ter p
LEFT JOIN taxoUsers t ON p.username = t.username);

COMMIT;
    """


    conn = None

    try:
        conn = get_db_params()
        cur = conn.cursor()
        print("I execute the big query!")
        cur.execute(bigQuery)
        cur.close()
        conn.commit()
        print("query committed!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    conn = None


def main():
    create_tables()


if __name__ == "__main__":
    main()
