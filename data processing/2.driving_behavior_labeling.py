#!/usr/bin/env python
# coding: utf-8
#为变道行为的左右打上标签
# In[ ]:


import glob
import pandas as pd
import numpy as np
import os
import time


# In[ ]:


#core function for labeling
def label(df1):
    #label for numchanges==1
    df1['label']=2 #先给所有的行打上标签2
    d=df1[df1['numLaneChanges']==1] #取出所有变道行为为1次的行，注意索引只能使用布尔型的Series而不能使用布尔型列表，因为要对照index进行索引!
    d1=d['id'].drop_duplicates() #找到id的单值列表
    for i in d1: #对所有的id进行循环
        dfid=df1[df1['id']==i] #取出id为i的所有行
        #frame=abs(dfid['yVelocity']).argmax()
        frame=abs(dfid['yVelocity']).idxmax() #找到这些行中y方向速度最大的行的索引，注意不是帧数，而是唯一的索引，切片索引并不会变化！
        m=0
        n=0
        while abs(df1.loc[frame-m,'yVelocity'])>gate_yveloctiy and df1.loc[frame-m,'id']== df1.loc[frame,'id']:#后面这句话保证指针不会溢出到其它track
            m=m+1
            if frame-m<1:
                break
        while abs(df1.loc[frame+n,'yVelocity'])>gate_yveloctiy and df1.loc[frame+n,'id']== df1.loc[frame,'id']:
            n=n+1
        if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+n,'y']>df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=1 #向左开，向左变道
        if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+n,'y']<df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=3 #向左开，向右变道
        if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+n,'y']>df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=3 #向右开，向右变道
        if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+n,'y']<df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=1 #向右开，向左边变道
            #总之，向左变道打上标签1，向右变道打上标签3，没变道保持标签2
    #label for numchanges>1
    f=df1[df1['numLaneChanges']>1] #取出所有变道行为为多次的行
    f1=f['id'].drop_duplicates()
    s=[]
    for j in f1:
        dfid=df1[df1['id']==j]
        for i in dfid.index:
            if i==dfid.index.max():
                break
            if dfid['laneId'][i]!=dfid['laneId'][i+1]: #laneID的变化是发生在变道瞬间的，之前用最大的y方向速度来找到变道瞬间，这里用的laneID变化而已
                 s.append(i)
            for frame in s:
                m=0
                n=0
                while abs(df1.loc[frame-m,'yVelocity'])>gate_yveloctiy and df1.loc[frame-m,'id']== df1.loc[frame,'id']:
                    m=m+1
                while abs(df1.loc[frame+n,'yVelocity'])>gate_yveloctiy and df1.loc[frame+n,'id']== df1.loc[frame,'id']:
                    n=n+1
                if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+1,'y']>df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=1
                if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+1,'y']<df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=3
                if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+1,'y']>df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=3
                if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+1,'y']<df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=1


# In[ ]:


gate_yveloctiy=0.1
base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/tracksenrich1'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
i=0
for path1 in files1:
    i=i+1
    t1=time.process_time()
    full_path1 = os.path.join(base_path1, path1)
    df4=pd.read_csv(full_path1)
    label(df4)
    df4.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/trackslabel/%d_trackslabel.csv'%(i))
    
    t2=time.process_time()
    print(t2-t1)


# In[ ]:




