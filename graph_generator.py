
import ujson
import numpy as np
# import seaborn as sns
import pandas as pd
import string
import matplotlib
import matplotlib.ticker as ticker

# matplotlib.use('WX')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
# import matplotlib.mlab as mlab


#load the data
path = '/Users/alessandro/Documents/PhD/userstats'
file_1 = path + '/WDataStats_all.json'
wdStats = pd.read_json(file_1, orient='index')

# file_1 = path + '/WDataStats_1.json'
# wdStats = pd.read_json(file_1, orient='index')
#
# file_2 = path + '/WDataStats_2.json'
# wdStats_2 = pd.read_json(file_2, orient='index')
#
file_3 = path + '/WDepth_new.json'
wdStats_3 = pd.read_json(file_3, orient='index')
#
# wdStats = pd.concat([wdStats, wdStats_2], axis=0)
# wdStats.drop('avgDepth', axis = 1, inplace=True)
# wdStats = pd.concat([wdStats, wdStats_3], axis=0)

wdStats = wdStats.fillna(0)
wdStats.reset_index(inplace=True)
wdStats['timeframe'] = pd.to_datetime(wdStats['index'])
wdStats.sort_values(by='timeframe', inplace=True)
wdStats['month'] = wdStats['timeframe'].apply(lambda x: x.strftime('%B %Y'))

wdStats_3 = wdStats_3.fillna(0)
wdStats_3.reset_index(inplace=True)
wdStats_3['timeframe'] = pd.to_datetime(wdStats_3['index'])
wdStats_3.sort_values(by='timeframe', inplace=True)
wdStats_3['month'] = wdStats_3['timeframe'].apply(lambda x: x.strftime('%B %Y'))

wdStats['noInstances'] = wdStats['avgPop'] * wdStats['noClasses']
wdStats['trueRichness'] = wdStats['classesWInstances']/wdStats['noClasses']

# ###create grid
# g = sns.FacetGrid(wdStats, col=['avgDepth', 'iRichness', 'cRichness'], hue=['avgDepth', 'iRichness', 'cRichness'], col_wrap=3, )
#
#
#
# ###generate the graphs
#
# plt.plot( 'timeframe', 'avgDepth', data=wdStats, marker='', color='olive', linewidth=2,  label="Avg. depth")
# # plt.plot( 'timeframe', 'avgPop', data=wdStats, marker='', color='olive', linewidth=2, linestyle='dashed', label="Avg. population")
# plt.plot( 'timeframe', 'cRichness', data=wdStats, marker='', color='olive', linewidth=2, linestyle='dashed', label="Class Richness")
# plt.plot( 'timeframe', 'iRichness', data=wdStats, marker='', color='olive', linewidth=2, linestyle='-.', label="Inheritance Richness")
# plt.legend()
#

###another one

# from itertools import izip,chain

def myticks(x, pos):

    if x == 0: return "$0$"

    # exponent = int(np.log10(100000))
    # coeff = x/10**exponent
    coeff = x/10000


    # return r"${:2.0f}\times 10^{{{:2d}}}$".format(coeff,exponent)
    return int(coeff)

def myticks_prop(x, pos):

    if x == 0: return "$0$"

    # exponent = int(np.log10(100000))
    # coeff = x/10**exponent
    coeff = x/100
    coeff = int(coeff)


    # return r"${:2.0f}\times 10^{{{:2d}}}$".format(coeff,exponent)
    return coeff

def myticks_root(x, pos):

    if x == 0: return "$0$"

    # exponent = int(np.log10(100000))
    # coeff = x/10**exponent
    coeff = x/1000
    coeff = int(coeff)


    # return r"${:2.0f}\times 10^{{{:2d}}}$".format(coeff,exponent)
    return coeff

###no classes
f2,((ax1,ax2, ax3)) = plt.subplots(1,3, sharex='col')

font = {'size': 10}

matplotlib.rc('font', **font)

ax1.plot(wdStats['timeframe'],wdStats['noClasses'],  marker='.', markevery=0.05)
ax1.plot(wdStats['timeframe'],wdStats['noRoot'], marker='x', markevery=0.05)
ax1.plot(wdStats['timeframe'],wdStats['noLeaf'], marker='^', markevery=0.05)
ax1.grid(color='gray', linestyle='--', linewidth=.5)
# forceAspect(ax1,aspect=1)
ax1.set_ylabel('No. Classes')

ax2.plot(wdStats['timeframe'],wdStats['noInstances'])
ax2.grid(color='gray', linestyle='--', linewidth=.5)

ax2.set_ylabel('No. Instances')

ax3.plot(wdStats['timeframe'],wdStats['noProps'])
ax3.grid(color='gray', linestyle='--', linewidth=.5)

ax3.set_ylabel('No. Properties')

ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=8))   #to get a tick every 15 minutes
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))     #optional formatting
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=8))   #to get a tick every 15 minutes
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))     #optional formatting
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=8))   #to get a tick every 15 minutes
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))     #optional formatting

f2.autofmt_xdate()

# ax.set_aspect('equal')
# plt.axes().set_aspect('equal')


# f.legend([ax4], 'cosi')
# plt.ylim(-width)
# plt.yticks(range(length), domains[0][0:length])

# plt.tight_layout()
plt.show()


f,((ax1,ax2, ax3), (ax4,ax5, ax6)) = plt.subplots(2,3, sharex='col')

font = {'size': 10}

matplotlib.rc('font', **font)

# matplotlib.rcdefaults()


ax1.plot(wdStats['timeframe'],wdStats['avgPop'], marker='.', markevery=0.05)
ax1.grid(color='gray', linestyle='--', linewidth=.5)
ax1.set_ylabel('Avg. population')
# ax1.yaxis.set_label_position('top')
ax2.plot(wdStats['timeframe'], wdStats['trueRichness'])
ax2.grid(color='gray', linestyle='--', linewidth=.5)
ax2.set_ylabel('Class richness')
ax3.plot(wdStats['timeframe'],wdStats['iRichness'])
ax3.grid(color='gray', linestyle='--', linewidth=.5)
ax3.set_ylabel('Inheritance richness')

ax4.plot(wdStats['timeframe'],wdStats['iRichness'])
ax4.grid(color='gray', linestyle='--', linewidth=.5)
ax4.set_ylabel('Relationship richness')


ax5.plot(wdStats_3['timeframe'],wdStats_3['avgDepth'],  marker='.', markevery=0.05)
ax5.plot(wdStats_3['timeframe'],wdStats_3['medianDepth'], marker='x', markevery=0.05)
ax5.plot(wdStats_3['timeframe'],wdStats_3['maxDepth'], marker='^', markevery=0.05)
ax5.grid(color='gray', linestyle='--', linewidth=.5)
ax5.set_ylabel('Graph depth')



ax6.plot(wdStats['timeframe'],wdStats['noProps'])
ax6.grid(color='gray', linestyle='--', linewidth=.5)
ax6.set_ylabel(r'Properties ($n*10^2$)')
ax6.yaxis.set_major_formatter(ticker.FuncFormatter(myticks_prop))
#
# ax7.plot(wdStats['timeframe'],wdStats['noClasses'])
# ax7.grid(color='gray', linestyle='--', linewidth=.5)
# ax7.set_ylabel(r'Classes ($n*10^4$)')
# ax7.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
#
# ax8.plot(wdStats['timeframe'],wdStats['noLeaf'])
# ax8.grid(color='gray', linestyle='--', linewidth=.5)
# ax8.set_ylabel(r'Leaf classes ($n*10^4$)')
# ax8.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
#
# ax9.plot(wdStats['timeframe'],wdStats['noRoot'])
# ax9.grid(color='gray', linestyle='--', linewidth=.5)
# ax9.set_ylabel(r'Root classes ($n*10^3$)')
# ax9.yaxis.set_major_formatter(ticker.FuncFormatter(myticks_root))



# plt.grid(which='both')
# plt.xlabel(r'$\%$ on the Total Number of Matches')



# # locator = mdates.AutoDateLocator()
# ax1.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax2.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax3.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax4.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax5.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax6.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax7.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax8.fmt_xdata = mdates.DateFormatter('%B %Y')
# ax9.fmt_xdata = mdates.DateFormatter('%B %Y')

ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))   #to get a tick every 15 minutes
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))     #optional formatting
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))   #to get a tick every 15 minutes
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))     #optional formatting
ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=6))   #to get a tick every 15 minutes
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))     #optional formatting

f.autofmt_xdate()


# f.legend([ax4], 'cosi')
# plt.ylim(-width)
# plt.yticks(range(length), domains[0][0:length])
plt.tight_layout()
plt.savefig("ontometrics.pdf", format="pdf")

plt.savefig('ontometrics.eps', format='eps', transparent=True)

plt.show()


