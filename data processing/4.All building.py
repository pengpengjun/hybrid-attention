import pandas as pd
import numpy as np

def all_building(num):
    path = r'./data/1_fol_pre_built/' + str(num) + '_fol_pre_built.csv'
    data = pd.read_csv(path)
    all_index = data.set_index(['id','frame'],drop=False)
    #左前车----------------------------------------------------------------------------------------------------------------------------------------------------
    leftPre_id_data = data.set_index(['leftPrecedingId','frame','id'],drop=False)
    self_id_data = data.set_index(['id','frame'],drop=False)
    
    leftPre_index = leftPre_id_data.index
    #print(leftPre_index)
    leftPre_index_valid = [x for x in leftPre_index if x[0] != 0]
    #print(len(leftPre_index_valid))
    data_with_leftPre = leftPre_id_data.loc[leftPre_index_valid].set_index(['id','frame'],drop=False)
    #print(data_with_leftPre.shape)
    
    leftPre_index_valid = data_with_leftPre.set_index(['leftPrecedingId','frame']).index
    leftPre_valid_data = self_id_data.loc[leftPre_index_valid,['x','y','xV','yV','xA','yA','class']]
    
    leftPre_valid_data.index = data_with_leftPre.index
    leftPre_valid_data.columns = ['leftPrex','leftPrey','leftPrexV','leftPreyV','leftPrexA','leftPreyA','leftPreClass']
    
    data_with_leftPre[['leftPrex','leftPrey','leftPrexV','leftPreyV','leftPrexA','leftPreyA','leftPreClass']] = leftPre_valid_data
    
    #data_with_leftPre.to_csv('test.csv')
    
    leftPre_index_invalid = [x for x in leftPre_index if x[0] == 0]
    data_without_leftPre = leftPre_id_data.loc[leftPre_index_invalid].set_index(['id','frame'],drop=False)
    
    data_without_leftPre_right =  data_without_leftPre[data_without_leftPre['drivingDirection']==2]
    data_without_leftPre_right.loc[:,'leftPrex'] = data_without_leftPre_right.loc[:,'x'] + 200
    data_without_leftPre_right.loc[:,'leftPrey'] = data_without_leftPre_right.loc[:,'y'] - 3.8
    data_without_leftPre_right.loc[:,'leftPrexV'] = data_without_leftPre_right.loc[:,'xV'] 
    data_without_leftPre_right.loc[:,'leftPreyV'] = data_without_leftPre_right.loc[:,'yV'] 
    data_without_leftPre_right.loc[:,'leftPrexA'] = data_without_leftPre_right.loc[:,'xA'] 
    data_without_leftPre_right.loc[:,'leftPreyA'] = data_without_leftPre_right.loc[:,'yA']
    data_without_leftPre_right.loc[:,'leftPreClass'] = data_without_leftPre_right.loc[:,'class']
    
    data_without_leftPre_left =  data_without_leftPre[data_without_leftPre['drivingDirection']==1]
    data_without_leftPre_left.loc[:,'leftPrex'] = data_without_leftPre_left.loc[:,'x'] - 200
    data_without_leftPre_left.loc[:,'leftPrey'] = data_without_leftPre_left.loc[:,'y'] + 3.8
    data_without_leftPre_left.loc[:,'leftPrexV'] = data_without_leftPre_left.loc[:,'xV'] 
    data_without_leftPre_left.loc[:,'leftPreyV'] = data_without_leftPre_left.loc[:,'yV'] 
    data_without_leftPre_left.loc[:,'leftPrexA'] = data_without_leftPre_left.loc[:,'xA'] 
    data_without_leftPre_left.loc[:,'leftPreyA'] = data_without_leftPre_left.loc[:,'yA']
    data_without_leftPre_left.loc[:,'leftPreClass'] = data_without_leftPre_left.loc[:,'class']
    
    data_processed = pd.concat([data_without_leftPre_right,data_without_leftPre_left,data_with_leftPre],axis=0).sort_index()
    data_processed = data_processed.drop(['frame','id'],axis=1)
    data_processed = data_processed.reset_index()
    
    #data_processed.to_csv('test.csv')
    #print(data_without_leftPre.shape)
    #print(data)
    
    #左边平行----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    leftAlong_id_data = data_processed.set_index(['leftAlongsideId','frame','id'],drop=False)
    self_id_data = data_processed.set_index(['id','frame'],drop=False)
    
    leftAlong_index = leftAlong_id_data.index
    #print(leftAlong_index)
    leftAlong_index_valid = [x for x in leftAlong_index if x[0] != 0]
    #print(len(leftAlong_index_valid))
    data_with_leftAlong = leftAlong_id_data.loc[leftAlong_index_valid].set_index(['id','frame'],drop=False)
    #print(data_with_leftAlong.shape)
    
    leftAlong_index_valid = data_with_leftAlong.set_index(['leftAlongsideId','frame']).index
    leftAlong_valid_data = self_id_data.loc[leftAlong_index_valid,['x','y','xV','yV','xA','yA','class']]
    
    leftAlong_valid_data.index = data_with_leftAlong.index
    leftAlong_valid_data.columns = ['leftAlongx','leftAlongy','leftAlongxV','leftAlongyV','leftAlongxA','leftAlongyA','leftAlongClass']
    
    data_with_leftAlong[['leftAlongx','leftAlongy','leftAlongxV','leftAlongyV','leftAlongxA','leftAlongyA','leftAlongClass']] = leftAlong_valid_data
    
    #data_with_leftAlong.to_csv('test.csv')
    
    leftAlong_index_invalid = [x for x in leftAlong_index if x[0] == 0]
    data_without_leftAlong = leftAlong_id_data.loc[leftAlong_index_invalid].set_index(['id','frame'],drop=False)
    
    data_without_leftAlong_right =  data_without_leftAlong[data_without_leftAlong['drivingDirection']==2]
    data_without_leftAlong_right.loc[:,'leftAlongx'] = data_without_leftAlong_right.loc[:,'x']
    data_without_leftAlong_right.loc[:,'leftAlongy'] = data_without_leftAlong_right.loc[:,'y'] - 3.8
    data_without_leftAlong_right.loc[:,'leftAlongxV'] = data_without_leftAlong_right.loc[:,'xV'] 
    data_without_leftAlong_right.loc[:,'leftAlongyV'] = data_without_leftAlong_right.loc[:,'yV'] 
    data_without_leftAlong_right.loc[:,'leftAlongxA'] = data_without_leftAlong_right.loc[:,'xA'] 
    data_without_leftAlong_right.loc[:,'leftAlongyA'] = data_without_leftAlong_right.loc[:,'yA']
    data_without_leftAlong_right.loc[:,'leftAlongClass'] = data_without_leftAlong_right.loc[:,'class']
    
    data_without_leftAlong_left =  data_without_leftAlong[data_without_leftAlong['drivingDirection']==1]
    data_without_leftAlong_left.loc[:,'leftAlongx'] = data_without_leftAlong_left.loc[:,'x']
    data_without_leftAlong_left.loc[:,'leftAlongy'] = data_without_leftAlong_left.loc[:,'y'] + 3.8
    data_without_leftAlong_left.loc[:,'leftAlongxV'] = data_without_leftAlong_left.loc[:,'xV'] 
    data_without_leftAlong_left.loc[:,'leftAlongyV'] = data_without_leftAlong_left.loc[:,'yV'] 
    data_without_leftAlong_left.loc[:,'leftAlongxA'] = data_without_leftAlong_left.loc[:,'xA'] 
    data_without_leftAlong_left.loc[:,'leftAlongyA'] = data_without_leftAlong_left.loc[:,'yA']
    data_without_leftAlong_left.loc[:,'leftAlongClass'] = data_without_leftAlong_left.loc[:,'class']
    
    data_processed = pd.concat([data_without_leftAlong_right,data_without_leftAlong_left,data_with_leftAlong],axis=0).sort_index()
    data_processed = data_processed.drop(['frame','id'],axis=1)
    data_processed = data_processed.reset_index()
    
    #data_processed.to_csv('test.csv')
    
    #左边后车-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    leftFol_id_data = data_processed.set_index(['leftFollowingId','frame','id'],drop=False)
    self_id_data = data_processed.set_index(['id','frame'],drop=False)
    
    leftFol_index = leftFol_id_data.index
    #print(leftFol_index)
    leftFol_index_valid = [x for x in leftFol_index if x[0] != 0]
    #print(len(leftFol_index_valid))
    data_with_leftFol = leftFol_id_data.loc[leftFol_index_valid].set_index(['id','frame'],drop=False)
    #print(data_with_leftFol.shape)
    
    leftFol_index_valid = data_with_leftFol.set_index(['leftFollowingId','frame']).index
    leftFol_valid_data = self_id_data.loc[leftFol_index_valid,['x','y','xV','yV','xA','yA','class']]
    
    leftFol_valid_data.index = data_with_leftFol.index
    leftFol_valid_data.columns = ['leftFolx','leftFoly','leftFolxV','leftFolyV','leftFolxA','leftFolyA','leftFolClass']
    
    data_with_leftFol[['leftFolx','leftFoly','leftFolxV','leftFolyV','leftFolxA','leftFolyA','leftFolClass']] = leftFol_valid_data
    
    #data_with_leftFol.to_csv('test.csv')
    
    leftFol_index_invalid = [x for x in leftFol_index if x[0] == 0]
    data_without_leftFol = leftFol_id_data.loc[leftFol_index_invalid].set_index(['id','frame'],drop=False)
    
    data_without_leftFol_right =  data_without_leftFol[data_without_leftFol['drivingDirection']==2]
    data_without_leftFol_right.loc[:,'leftFolx'] = data_without_leftFol_right.loc[:,'x'] - 200
    data_without_leftFol_right.loc[:,'leftFoly'] = data_without_leftFol_right.loc[:,'y'] - 3.8
    data_without_leftFol_right.loc[:,'leftFolxV'] = data_without_leftFol_right.loc[:,'xV'] 
    data_without_leftFol_right.loc[:,'leftFolyV'] = data_without_leftFol_right.loc[:,'yV'] 
    data_without_leftFol_right.loc[:,'leftFolxA'] = data_without_leftFol_right.loc[:,'xA'] 
    data_without_leftFol_right.loc[:,'leftFolyA'] = data_without_leftFol_right.loc[:,'yA']
    data_without_leftFol_right.loc[:,'leftFolClass'] = data_without_leftFol_right.loc[:,'class']
    
    data_without_leftFol_left =  data_without_leftFol[data_without_leftFol['drivingDirection']==1]
    data_without_leftFol_left.loc[:,'leftFolx'] = data_without_leftFol_left.loc[:,'x'] + 200
    data_without_leftFol_left.loc[:,'leftFoly'] = data_without_leftFol_left.loc[:,'y'] + 3.8
    data_without_leftFol_left.loc[:,'leftFolxV'] = data_without_leftFol_left.loc[:,'xV'] 
    data_without_leftFol_left.loc[:,'leftFolyV'] = data_without_leftFol_left.loc[:,'yV'] 
    data_without_leftFol_left.loc[:,'leftFolxA'] = data_without_leftFol_left.loc[:,'xA'] 
    data_without_leftFol_left.loc[:,'leftFolyA'] = data_without_leftFol_left.loc[:,'yA']
    data_without_leftFol_left.loc[:,'leftFolClass'] = data_without_leftFol_left.loc[:,'class']
    
    data_processed = pd.concat([data_without_leftFol_right,data_without_leftFol_left,data_with_leftFol],axis=0).sort_index()
    data_processed = data_processed.drop(['frame','id'],axis=1)
    data_processed = data_processed.reset_index()
    
    #data_processed.to_csv('test.csv')
    
    #右前车----------------------------------------------------------------------------------------------------------------------------------------------------
    rightPre_id_data = data_processed.set_index(['rightPrecedingId','frame','id'],drop=False)
    self_id_data = data_processed.set_index(['id','frame'],drop=False)
    
    rightPre_index = rightPre_id_data.index
    #print(rightPre_index)
    rightPre_index_valid = [x for x in rightPre_index if x[0] != 0]
    #print('rightPre_index_valid shape is:\n',len(rightPre_index_valid),'\n\n')
    #print(len(rightPre_index_valid))
    data_with_rightPre = rightPre_id_data.loc[rightPre_index_valid].set_index(['id','frame'],drop=False)
    #print(data_with_rightPre.shape)
    
    rightPre_index_valid = data_with_rightPre.set_index(['rightPrecedingId','frame']).index
    rightPre_valid_data = self_id_data.loc[rightPre_index_valid,['x','y','xV','yV','xA','yA','class']]
    
    rightPre_valid_data.index = data_with_rightPre.index
    rightPre_valid_data.columns = ['rightPrex','rightPrey','rightPrexV','rightPreyV','rightPrexA','rightPreyA','rightPreClass']
    
    data_with_rightPre[['rightPrex','rightPrey','rightPrexV','rightPreyV','rightPrexA','rightPreyA','rightPreClass']] = rightPre_valid_data
    
    #data_with_rightPre.to_csv('test.csv')
    
    rightPre_index_invalid = [x for x in rightPre_index if x[0] == 0]
    #print('rightPre_index_invalid shape is:\n',len(rightPre_index_invalid),'\n\n')
    data_without_rightPre = rightPre_id_data.loc[rightPre_index_invalid].set_index(['id','frame'],drop=False)
    #print(data_without_rightPre.shape)
    data_without_rightPre_right =  data_without_rightPre[data_without_rightPre['drivingDirection']==2]
    #print(data_without_rightPre_right.shape)
    data_without_rightPre_right.loc[:,'rightPrex'] = data_without_rightPre_right.loc[:,'x'] + 200
    data_without_rightPre_right.loc[:,'rightPrey'] = data_without_rightPre_right.loc[:,'y'] + 3.8
    data_without_rightPre_right.loc[:,'rightPrexV'] = data_without_rightPre_right.loc[:,'xV'] 
    data_without_rightPre_right.loc[:,'rightPreyV'] = data_without_rightPre_right.loc[:,'yV'] 
    data_without_rightPre_right.loc[:,'rightPrexA'] = data_without_rightPre_right.loc[:,'xA'] 
    data_without_rightPre_right.loc[:,'rightPreyA'] = data_without_rightPre_right.loc[:,'yA']
    data_without_rightPre_right.loc[:,'rightPreClass'] = data_without_rightPre_right.loc[:,'class']
    
    data_without_rightPre_left =  data_without_rightPre[data_without_rightPre['drivingDirection']==1]
    data_without_rightPre_left.loc[:,'rightPrex'] = data_without_rightPre_left.loc[:,'x'] - 200
    data_without_rightPre_left.loc[:,'rightPrey'] = data_without_rightPre_left.loc[:,'y'] - 3.8
    data_without_rightPre_left.loc[:,'rightPrexV'] = data_without_rightPre_left.loc[:,'xV'] 
    data_without_rightPre_left.loc[:,'rightPreyV'] = data_without_rightPre_left.loc[:,'yV'] 
    data_without_rightPre_left.loc[:,'rightPrexA'] = data_without_rightPre_left.loc[:,'xA'] 
    data_without_rightPre_left.loc[:,'rightPreyA'] = data_without_rightPre_left.loc[:,'yA']
    data_without_rightPre_left.loc[:,'rightPreClass'] = data_without_rightPre_left.loc[:,'class']
    
    data_processed = pd.concat([data_without_rightPre_right,data_without_rightPre_left,data_with_rightPre],axis=0).sort_index()
    data_processed = data_processed.drop(['frame','id'],axis=1)
    data_processed = data_processed.reset_index()
    
    data_processed.to_csv('rightPre.csv')
    #print(data_without_rightPre.shape)
    #print(data)
    
    #右边平行----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    rightAlong_id_data = data_processed.set_index(['rightAlongsideId','frame','id'],drop=False)
    self_id_data = data_processed.set_index(['id','frame'],drop=False)
    
    rightAlong_index = rightAlong_id_data.index
    #print(rightAlong_index)
    rightAlong_index_valid = [x for x in rightAlong_index if x[0] != 0]
    #print(len(rightAlong_index_valid))
    data_with_rightAlong = rightAlong_id_data.loc[rightAlong_index_valid].set_index(['id','frame'],drop=False)
    #print(data_with_rightAlong.shape)
    
    rightAlong_index_valid = data_with_rightAlong.set_index(['rightAlongsideId','frame']).index
    #print(rightAlong_index_valid)
    #print('无效索引为：\n',[x for x in rightAlong_index_valid if x not in all_index])
    rightAlong_valid_data = self_id_data.loc[rightAlong_index_valid,['x','y','xV','yV','xA','yA','class']]
    
    rightAlong_valid_data.index = data_with_rightAlong.index
    rightAlong_valid_data.columns = ['rightAlongx','rightAlongy','rightAlongxV','rightAlongyV','rightAlongxA','rightAlongyA','rightAlongClass']
    
    data_with_rightAlong[['rightAlongx','rightAlongy','rightAlongxV','rightAlongyV','rightAlongxA','rightAlongyA','rightAlongClass']] = rightAlong_valid_data
    
    #data_with_rightAlong.to_csv('test.csv')
    
    rightAlong_index_invalid = [x for x in rightAlong_index if x[0] == 0]
    data_without_rightAlong = rightAlong_id_data.loc[rightAlong_index_invalid].set_index(['id','frame'],drop=False)
    
    data_without_rightAlong_right =  data_without_rightAlong[data_without_rightAlong['drivingDirection']==2]
    data_without_rightAlong_right.loc[:,'rightAlongx'] = data_without_rightAlong_right.loc[:,'x']
    data_without_rightAlong_right.loc[:,'rightAlongy'] = data_without_rightAlong_right.loc[:,'y'] + 3.8
    data_without_rightAlong_right.loc[:,'rightAlongxV'] = data_without_rightAlong_right.loc[:,'xV'] 
    data_without_rightAlong_right.loc[:,'rightAlongyV'] = data_without_rightAlong_right.loc[:,'yV'] 
    data_without_rightAlong_right.loc[:,'rightAlongxA'] = data_without_rightAlong_right.loc[:,'xA'] 
    data_without_rightAlong_right.loc[:,'rightAlongyA'] = data_without_rightAlong_right.loc[:,'yA']
    data_without_rightAlong_right.loc[:,'rightAlongClass'] = data_without_rightAlong_right.loc[:,'class']
    
    data_without_rightAlong_left =  data_without_rightAlong[data_without_rightAlong['drivingDirection']==1]
    data_without_rightAlong_left.loc[:,'rightAlongx'] = data_without_rightAlong_left.loc[:,'x']
    data_without_rightAlong_left.loc[:,'rightAlongy'] = data_without_rightAlong_left.loc[:,'y'] - 3.8
    data_without_rightAlong_left.loc[:,'rightAlongxV'] = data_without_rightAlong_left.loc[:,'xV'] 
    data_without_rightAlong_left.loc[:,'rightAlongyV'] = data_without_rightAlong_left.loc[:,'yV'] 
    data_without_rightAlong_left.loc[:,'rightAlongxA'] = data_without_rightAlong_left.loc[:,'xA'] 
    data_without_rightAlong_left.loc[:,'rightAlongyA'] = data_without_rightAlong_left.loc[:,'yA']
    data_without_rightAlong_left.loc[:,'rightAlongClass'] = data_without_rightAlong_left.loc[:,'class']
    
    data_processed = pd.concat([data_without_rightAlong_right,data_without_rightAlong_left,data_with_rightAlong],axis=0)
    data_processed = data_processed.drop(['frame','id'],axis=1)
    data_processed = data_processed.reset_index()
    
    #data_processed.to_csv('test.csv')
    
    #右边后车-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    rightFol_id_data = data_processed.set_index(['rightFollowingId','frame','id'],drop=False)
    self_id_data = data_processed.set_index(['id','frame'],drop=False)
    
    rightFol_index = rightFol_id_data.index
    #print(rightFol_index)
    rightFol_index_valid = [x for x in rightFol_index if x[0] != 0]
    #print(len(rightFol_index_valid))
    data_with_rightFol = rightFol_id_data.loc[rightFol_index_valid].set_index(['id','frame'],drop=False)
    #print(data_with_rightFol.shape)
    
    rightFol_index_valid = data_with_rightFol.set_index(['rightFollowingId','frame']).index
    rightFol_valid_data = self_id_data.loc[rightFol_index_valid,['x','y','xV','yV','xA','yA','class']]
    
    rightFol_valid_data.index = data_with_rightFol.index
    rightFol_valid_data.columns = ['rightFolx','rightFoly','rightFolxV','rightFolyV','rightFolxA','rightFolyA','rightFolClass']
    
    data_with_rightFol[['rightFolx','rightFoly','rightFolxV','rightFolyV','rightFolxA','rightFolyA','rightFolClass']] = rightFol_valid_data
    
    #data_with_rightFol.to_csv('test.csv')
    
    rightFol_index_invalid = [x for x in rightFol_index if x[0] == 0]
    data_without_rightFol = rightFol_id_data.loc[rightFol_index_invalid].set_index(['id','frame'],drop=False)
    
    data_without_rightFol_right =  data_without_rightFol[data_without_rightFol['drivingDirection']==2]
    data_without_rightFol_right.loc[:,'rightFolx'] = data_without_rightFol_right.loc[:,'x'] - 200
    data_without_rightFol_right.loc[:,'rightFoly'] = data_without_rightFol_right.loc[:,'y'] + 3.8
    data_without_rightFol_right.loc[:,'rightFolxV'] = data_without_rightFol_right.loc[:,'xV'] 
    data_without_rightFol_right.loc[:,'rightFolyV'] = data_without_rightFol_right.loc[:,'yV'] 
    data_without_rightFol_right.loc[:,'rightFolxA'] = data_without_rightFol_right.loc[:,'xA'] 
    data_without_rightFol_right.loc[:,'rightFolyA'] = data_without_rightFol_right.loc[:,'yA']
    data_without_rightFol_right.loc[:,'rightFolClass'] = data_without_rightFol_right.loc[:,'class']
    
    data_without_rightFol_left =  data_without_rightFol[data_without_rightFol['drivingDirection']==1]
    data_without_rightFol_left.loc[:,'rightFolx'] = data_without_rightFol_left.loc[:,'x'] + 200
    data_without_rightFol_left.loc[:,'rightFoly'] = data_without_rightFol_left.loc[:,'y'] - 3.8
    data_without_rightFol_left.loc[:,'rightFolxV'] = data_without_rightFol_left.loc[:,'xV'] 
    data_without_rightFol_left.loc[:,'rightFolyV'] = data_without_rightFol_left.loc[:,'yV'] 
    data_without_rightFol_left.loc[:,'rightFolxA'] = data_without_rightFol_left.loc[:,'xA'] 
    data_without_rightFol_left.loc[:,'rightFolyA'] = data_without_rightFol_left.loc[:,'yA']
    data_without_rightFol_left.loc[:,'rightFolClass'] = data_without_rightFol_left.loc[:,'class']
    
    data_processed = pd.concat([data_without_rightFol_right,data_without_rightFol_left,data_with_rightFol],axis=0)
    data_processed = data_processed.drop(['frame','id'],axis=1)
    data_processed = data_processed.reset_index()
    
    data_processed = data_processed.drop(['precedingId','followingId','leftPrecedingId','leftAlongsideId','leftFollowingId','rightPrecedingId','rightFollowingId','rightAlongsideId'],axis=1)
    data_processed.to_csv('./data/2_all_built/'+str(num)+'_all_built.csv',index=None)
    
for num in range(1,61):
    print(num,'\n\n')
    all_building(num)