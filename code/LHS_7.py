
from pyDOE import lhs
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy import stats

def cdf(xi,name,b):
    res_freq = stats.relfreq(xi, numbins=b, defaultreallimits=(xi.min(),xi.max()))
    pdf_value = res_freq.frequency
    cdf_value = np.cumsum(pdf_value)
    
    xx = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    
    return xx, cdf_value

def lhs_(x,c):
    d = x.shape[1]
    design = lhs(d, c)

    for i in range(d):
        t1 = time.time()
        print(str(int((t1-t0)/60))+'分钟')
        xi = x[x.columns[i]]
        b = xi.value_counts().shape[0]
        xx, cdf_value = cdf(xi, x.columns[i], b)

        cdf_ = pd.DataFrame()
        cdf_[x.columns[i]+'_value'] = xx
        cdf_[x.columns[i]+'_cdf_value'] = cdf_value

        for j in range(design[:, i].shape[0]):
            big = cdf_.loc[cdf_[x.columns[i]+'_cdf_value'] >= design[j, i], 
                           [x.columns[i]+'_value']].iloc[0].values[0]
            design[j, i] = big

    return cdf_,design


if __name__ == '__main__':
    t0 = time.time()
    
    file = r'D:\本科\任务\沼气工程\0315\能源经济大赛\dta厌氧工厂.csv'
    data = pd.read_csv(file)
    
    
    # data = data.loc[:,['水解液位', '水解温度', '水解pH', '水解TS', '水解VS', '水解VS/TS', '水解总流量',
    #        '发酵液位', '发酵温度', '进料量', '发酵TS', '发酵VS', '发酵OLR', '发酵VFA', '发酵碱度',
    #        '发酵酸碱比', '发酵pH', '发酵总流量', '螺杆泵出料口压力', '压力', '产气量']]
    
    data = data.loc[:,['水解pH', '水解VFA', '水解TS', '水解VS', '水解氨氮', '水解COD', '发酵pH', '发酵VFA',
           '发酵碱度', '发酵TS', '发酵VS', '发酵氨氮', '发酵COD', '进料量', '产气量', '单位VS产气量']]
    
    x = data
    
    c = int(1e5) # 扩增数
    cdf_,design = lhs_(x,c)
    
    xx = np.array(x)
    
    design = pd.DataFrame(design)
    design.columns = x.columns
    
    design.to_csv(r'D:\本科\任务\沼气工程\0315\能源经济大赛\dta厌氧工厂_lhs.csv',index = False,encoding='utf_8_sig')
    
