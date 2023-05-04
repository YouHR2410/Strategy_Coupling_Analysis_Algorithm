# -*- coding: utf-8 -*-
import pandas as pd


#two functions have the same result 
def calculate_real_from_csv(csv_name,start_round):
    data = pd.read_csv(csv_name)
    start = start_round
    data = data[start:len(data)]
    avg = data.mean().tolist()
    sum_lst = []
    for i in range(len(data.iloc[0])):
        for j in range(i,len(data.iloc[0])):
            sum = 0
            for k in range(len(data)):
                sum+=(data.iloc[k,i]-avg[i])*(data.iloc[k,j]-avg[j])
            sum_lst.append(sum/len(data))
    return  sum_lst


def calculate_real_from_csv_2(csv_name,start_round):
    data = pd.read_csv(csv_name)
    start = start_round
    data = data[start:len(data)]
    avg = data.mean().tolist()
    sum_lst = []
    for i in range(len(data.iloc[0])):
        data2 = pd.DataFrame(data.iloc[:,i],columns=['A'])
        data2['A'] = data.iloc[:,i]
        for j in range(i,len(data.iloc[0])):
            data2['B']=data.iloc[:,j]
            t = data2.A.cov(data2.B)
            sum_lst.append(t)
    return  sum_lst



