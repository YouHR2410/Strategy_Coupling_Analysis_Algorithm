# -*- coding: utf-8 -*-
import pandas as pd


def calculate_real_from_csv(csv_name,start_round):
    data = pd.read_csv(csv_name)
    start = start_round
    data = data[start:len(data)]
    sum_lst = []
    for i in range(len(data.iloc[0])):
        data2 = pd.DataFrame(data.iloc[:,i],columns=['A'])
        data2['A'] = data.iloc[:,i]
        for j in range(i,len(data.iloc[0])):
            data2['B']=data.iloc[:,j]
            t = data2.A.cov(data2.B)
            sum_lst.append(t)
    return  sum_lst



