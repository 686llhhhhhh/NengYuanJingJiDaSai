# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:10:55 2022

@author: LiLinhui2019010686

'Hello world !'
"""

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
        self.fc0 = nn.Linear(14, 200, bias=True)
        self.dd = nn.Linear(200, 200, bias=True)
        self.dd2 = nn.Linear(200, 200, bias=True)
        self.dd3 = nn.Linear(200, 200, bias=True)
        self.dd4 = nn.Linear(200, 200, bias=True)

        self.fc2 = nn.Linear(200, 1, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        c = x
        x=self.dd(x)*c
        x=self.dd2(x)*c
        x=self.dd3(x)*c
        x=self.dd4(x) * c
        x = self.fc2(x)
        return x

filename = r'D:\本科\任务\沼气工程\0315\能源经济大赛\dta厌氧工厂.csv'
data = pd.read_csv(filename)

features = [
            '水解pH',
            '水解VFA',
            '水解TS',
            '水解VS',
            '水解氨氮',
            '水解COD',
            '发酵pH',
            '发酵VFA',
            '发酵碱度',
            '发酵TS',
            '发酵VS',
            '发酵氨氮',
            '发酵COD',
            '进料量',
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

###############################################################################

filename = r'D:\本科\任务\沼气工程\0315\能源经济大赛\dta厌氧工厂_lhs.csv'
data = pd.read_csv(filename)

x = np.array(data[features])

x_pre = trans_x.transform(x)

###############################################################################

pre_x = torch.from_numpy(x_pre).float()

###############################################################################
net=torch.load('ddnet_2.pkl')#载入模型

net.eval()
pre_y = net(pre_x).detach().numpy()
pre_y = trans_y.inverse_transform(pre_y)

data[target] = pre_y

h = data
h['单位VS产气量'] = h['产气量']/(h['进料量']*1.1*h['水解TS']*h['水解VS'])
h['单位VS产甲烷量'] = h['单位VS产气量']*0.6
h.sort_values(by="单位VS产甲烷量",axis=0,ascending=False,inplace=True)


h.to_csv(filename+'_.csv',encoding='utf_8_sig',index=None)
