import os
import sys

#reload(sys)
# sys.setdefaultencoding("utf8")

import pandas as pd
import io
import bz2




def list_cleaner(rev_list):

    rev_list = rev_list.replace('\t', '')
    rev_list = rev_list.replace('\n', '')
    rev_list = re.sub(
        r"<id>|</id>|<parentid>|</parentid>|<timestamp>|</timestamp>|<username>|</username>|<ip>|</ip>|<comment>|</comment>",
        '', rev_list)
    rev_list = rev_list.lstrip(' ')

    return rev_list



def userTranslator(fileName):
    print(fileName)
    with bz2.BZ2File(fileName, 'rb') as inputfile:
        dictUsers = {}

        prev_line = None
        for line in inputfile:

            if '<id>' in line and '<username>' in prev_line:
                userId = line_cleaner(line)
                userName = list_cleaner(prev_line)
                if userName not in dictUsers.keys():
                    dictUsers[userName] = userId

            prev_line = line

    userIDf = pd.DataFrame.from_dict(dictUsers, orient='index')
    userIDf.to_csv('userIDmatches.csv', mode='a')


def main():

    fin = sys.argv[1]
    userTranslator(fin)


if __name__ == "__main__":
    main()

