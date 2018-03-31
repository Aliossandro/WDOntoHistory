import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ujson

#load the data
path = '/Users/alessandro/Documents/PhD/userstats'
file_1 = path + '/WDataStats_1.json'
wdStats = pd.read_json(file_1, orient='index')

file_2 = path + '/WDataStats_2.json'
wdStats_2 = pd.read_json(file_2, orient='index')

wdStats = pd.concat([wdStats, wdStats_2], axis=0)

wdStats = wdStats.fillna(0)
wdStats.reset_index(inplace=True)
wdStats['timeframe'] = pd.to_datetime(wdStats['index'])


###generate the graphs

plt.plot( 'timeframe', 'avgDepth', data=wdStats, marker='', color='olive', linewidth=2,  label="Avg. depth")
# plt.plot( 'timeframe', 'avgPop', data=wdStats, marker='', color='olive', linewidth=2, linestyle='dashed', label="Avg. population")
plt.plot( 'timeframe', 'cRichness', data=wdStats, marker='', color='olive', linewidth=2, linestyle='dashed', label="Class Richness")
plt.plot( 'timeframe', 'iRichness', data=wdStats, marker='', color='olive', linewidth=2, linestyle='-.', label="Inheritance Richness")
plt.legend()