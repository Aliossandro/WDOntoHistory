
import ujson
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#load the data
path = '/Users/alessandro/Documents/PhD/userstats'
file_1 = path + '/WDataStats_1.json'
wdStats = pd.read_json(file_1, orient='index')

file_2 = path + '/WDataStats_2.json'
wdStats_2 = pd.read_json(file_2, orient='index')

file_3 = path + '/WDepth.json'
wdStats_3 = pd.read_json(file_3, orient='index')

wdStats = pd.concat([wdStats, wdStats_2], axis=0)
wdStats.drop('avgDepth', axis = 1, inplace=True)


wdStats = wdStats.fillna(0)
wdStats.reset_index(inplace=True)
wdStats['timeframe'] = pd.to_datetime(wdStats['index'])

###create grid
g = sns.FacetGrid(wdStats, col=['avgDepth', 'iRichness', 'cRichness'], hue=['avgDepth', 'iRichness', 'cRichness'], col_wrap=3, )



###generate the graphs

plt.plot( 'timeframe', 'avgDepth', data=wdStats, marker='', color='olive', linewidth=2,  label="Avg. depth")
# plt.plot( 'timeframe', 'avgPop', data=wdStats, marker='', color='olive', linewidth=2, linestyle='dashed', label="Avg. population")
plt.plot( 'timeframe', 'cRichness', data=wdStats, marker='', color='olive', linewidth=2, linestyle='dashed', label="Class Richness")
plt.plot( 'timeframe', 'iRichness', data=wdStats, marker='', color='olive', linewidth=2, linestyle='-.', label="Inheritance Richness")
plt.legend()