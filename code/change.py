# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({'figure.max_open_warning': 0})


# file = r'D:\本科\任务\沼气工程\0315\能源经济大赛\dta厌氧工厂_'
file = r'D:\本科\任务\沼气工程\0315\能源经济大赛\data湿式工厂_'

data = pd.read_csv(file+'change.csv')[-50:].reset_index(drop=True)

# features = [
#             '水解pH',
#             '水解VFA',
#             '水解氨氮',
#             '水解COD',
#             '发酵pH',
#             '发酵VFA',
#             '发酵碱度',
#             '发酵氨氮',
#             '发酵COD',
# ]

features = ['水解温度', '水解pH', '水解总流量',
                   '发酵液位', '发酵温度', '发酵OLR', '发酵VFA', '发酵碱度',
                   '发酵pH', '发酵总流量', '螺杆泵出料口压力', '压力'
]

english = features
z = 'abcdefghijklmnopqrstuvwxyz'


plt.figure(figsize=(8,5))

for i in range(len(features)):

    plt.subplot(4, 3, i+1)

    x = data.shape[0]
    x = np.linspace(1,x,x)
    f = features[i]

    for j in range(data[f].shape[0]):
        if data[f+'_change'][j]-data[f][j] > 0:
            c='r'
        else:
            c ='g'
        plt.plot([x[j],x[j]], [data[f][j], data[f+'_change'][j]], c=c, linewidth=1)

    plt.scatter(x, data[f], s=2, c='b', label='原始')#Raw
    plt.scatter(x, data[f+'_change'], marker='s',s=2, c='k', label='优化')#Change

    plt.xticks([])
    # plt.xlabel(r'$\mathrm{'+english[i]+'}$', weight='bold')
    plt.xlabel(english[i])
    
    plt.text(0, max(np.max(data[f]),np.max(data[f+'_change'])), s='('+z[i]+')      ', fontsize=10, ha='right', weight='bold', va='center_baseline')
    
    if i==2:
        plt.legend(frameon=False,  prop={'size': 12}, loc=2, bbox_to_anchor=(1,1.0), borderaxespad = 0.)
    
plt.tight_layout()
plt.savefig(file+'.png', dpi=1200, bbox_inches='tight')

