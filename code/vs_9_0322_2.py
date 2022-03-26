# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:23:04 2022

@author: LiLinhui2019010686

'Hello world !'
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


import torch
from torch import nn
import numpy as np
import pandas as pd
np.random.seed(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class ddNet(nn.Module):

    def __init__(self):
        super(ddNet, self).__init__()

        # xw+b
        self.fc0 = nn.Linear(18, 200, bias=True)
        self.dd = nn.Linear(200, 200, bias=True)
        self.dd2 = nn.Linear(200, 200, bias=True)
        self.dd3 = nn.Linear(200, 200, bias=True)
        self.dd4 = nn.Linear(200, 200, bias=True)

        self.fc2 = nn.Linear(200, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.fc0(x)
        c = x     
        x=self.dd(x)*c
        x=self.dd2(x)*c
        x=self.dd3(x)*c
        x=self.dd4(x)*c

        x = self.fc2(x)
        return x

filename = r'D:\本科\任务\沼气工程\0315\能源经济大赛\data湿式工厂.csv'
data = pd.read_csv(filename)

features = [
            # "水解液位",
            "水解温度",
            "水解pH",
            "水解TS",
            "水解VS",
            "水解VS/TS",
            "水解总流量",
            "发酵液位",
            "发酵温度",
            "进料量",
            "发酵TS",
            "发酵VS",
            "发酵OLR",
            "发酵VFA",
            "发酵碱度",
            "发酵pH",
            "发酵总流量",
            "螺杆泵出料口压力",
            "压力"
]
target = ['产气量']

x = np.array(data[features])
y = np.array(data[target]).reshape(-1,1)

###############################################################################
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y,
                                                                test_size=0.2,
                                                                random_state=233
                                                                )
trans_x = StandardScaler()
trans_y = StandardScaler()
x_train = trans_x.fit_transform(X_train)
y_train = trans_y.fit_transform(Y_train)

net=torch.load('ddnet_1.pkl')#载入模型

def write(xxx):
    features2 = [ '水解温度_change', '水解pH_change', '水解TS', '水解VS', '水解VS/TS', '水解总流量_change',
           '发酵液位_change', '发酵温度_change', '进料量', '发酵TS', '发酵VS', '发酵OLR_change', '发酵VFA_change', '发酵碱度_change','发酵pH_change', '发酵总流量_change', '螺杆泵出料口压力_change', '压力_change']
    xx = xxx.loc[:,features2]
    
    
    x_pre = trans_x.transform(xx)
    pre_x = torch.from_numpy(x_pre).float()

    net.eval()
    pre_y = net(pre_x).detach().numpy()
    y_2 = trans_y.inverse_transform(pre_y)
    
    xxx['产气量_change'] = y_2
    xxx['单位VS产甲烷量_change'] = xxx['产气量_change']/(xxx['进料量']*xxx['水解VS']*xxx['水解TS']*1.1)*0.6
    
    xxx['产气量_提升'] = xxx['产气量_change']-xxx['产气量']
    
    f = ['日期', '水解液位', '水解温度','水解温度_change', '水解pH','水解pH_change', '水解TS', '水解VS', '水解VS/TS', '水解总流量','水解总流量_change',
           '发酵液位','发酵液位_change', '发酵温度','发酵温度_change', '进料量', '发酵TS', '发酵VS', '发酵OLR','发酵OLR_change', '发酵VFA', '发酵VFA_change','发酵碱度','发酵碱度_change',
           '发酵酸碱比','发酵pH','发酵pH_change', '发酵总流量', '发酵总流量_change', '螺杆泵出料口压力', '螺杆泵出料口压力_change','压力','压力_change', '产气量', '产气量_change', '产气量_提升', '单位VS产甲烷量_change',]
    xxx = xxx.loc[:,f]
    
    xxx.to_csv(file+'_change.csv',encoding='utf_8_sig',index=None)
    
    return xxx

def tiao(b, pd, trans_x, trans_y):
    features = ["水解温度",
            "水解pH",
            "水解TS",
            "水解VS",
            "水解VS/TS",
            "水解总流量",
            "发酵液位",
            "发酵温度",
            "进料量",
            "发酵TS",
            "发酵VS",
            "发酵OLR",
            "发酵VFA",
            "发酵碱度",
            "发酵pH",
            "发酵总流量",
            "螺杆泵出料口压力",
            "压力"]
    for w in shuxing:
        b[w] = pd[w]
    xx = b.loc[:,features]
    
    x_pre = trans_x.transform(xx)
    pre_x = torch.from_numpy(x_pre).float()

    net.eval()
    pre_y = net(pre_x).detach().numpy()
    y_2 = trans_y.inverse_transform(pre_y)
    
    b['产气量'] = y_2
    b['单位VS产甲烷量'] = b['产气量']/(b['进料量']*b['水解VS']*b['水解TS']*1.1)*0.6
    
    features = ['水解温度', '水解pH', '水解总流量',
           '发酵液位', '发酵温度', '发酵OLR', '发酵VFA', '发酵碱度',
           '发酵酸碱比', '发酵pH', '发酵总流量', '螺杆泵出料口压力', '压力']
    
    return b.loc[b.单位VS产甲烷量 == max(b.单位VS产甲烷量), features]
    


if __name__ == '__main__':
    
    file = r'D:\本科\任务\沼气工程\0315\能源经济大赛\data湿式工厂'
    
    data1 = pd.read_csv(file + '.csv')
    
    data2 = pd.read_csv(file + '_lhs.csv_.csv')
    
    features = ['水解温度', '水解pH', '水解总流量',
                       '发酵液位', '发酵温度', '发酵OLR', '发酵VFA', '发酵碱度',
                        '发酵pH', '发酵总流量', '螺杆泵出料口压力', '压力'
    ]
    
    target = '单位VS产甲烷量'
    
    qianli = 1000
    d = data2.loc[(data2.单位VS产甲烷量>qianli), :].reset_index(drop=True)
    
    shuxing = ['水解TS', '水解VS','水解VS/TS', '发酵TS', '发酵VS', '进料量']
    
    
    
    trans = StandardScaler()
    d_fit = trans.fit_transform(d[shuxing])
    
    print('聚类{}'.format(d.shape[0]))
    kmeans = KMeans(n_clusters=500, random_state=0).fit(d_fit)
    
    i = pd.DataFrame(kmeans.predict(d_fit), columns=['lab'])
    d = pd.concat([d,i],axis=1)
    
    x = data1.shape[0]
    ind = kmeans.predict(trans.transform(np.array(data1[shuxing].iloc[:x,:])))
    change = pd.DataFrame()
    
    print('换')
    
    for i in range(x):
        print('\r'+str(i),end='')
        b = d.loc[(d.lab == ind[i]), shuxing+features+['单位VS产甲烷量']].reset_index(drop=True)
        b = tiao(b, data1.iloc[i], trans_x, trans_y)
        b.index = [i]
        b.columns = [x+'_change' for x in features]
        b_change = pd.merge(data1,b,left_index=True,right_index=True)
        change = pd.concat([change,b_change])
    
    write(change)