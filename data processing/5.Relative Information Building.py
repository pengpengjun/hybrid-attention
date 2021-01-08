import pandas as pd
import numpy as np

def rel_info_building(num):
    path = r'./data/2_all_built/' + str(num) + '_all_built.csv'
    data = pd.read_csv(path)
    
    def help_func(data,pos,item):
        data['Rel' + pos.capitalize() + item] = data[pos+item] - data[item]
        del data[pos+item]
        return data
    
    position = ['fol','pre','leftPre','leftAlong','leftFol','rightPre','rightAlong','rightFol']
    items = ['x','y','xV','yV','xA','yA']
    
    for pos in position:
        for item in items:
            data = help_func(data,pos,item)

    data.to_csv('./data/3_rel_built/'+str(num)+'_rel_built.csv',index=None)
    #print(data)
    
    
for num in range(1,61):
    print(num,'\n\n')
    rel_info_building(num)