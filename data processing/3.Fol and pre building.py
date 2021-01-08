import pandas as pd
import numpy as np

def fol_pre_building(num):
    #'F:\Jupyter_Workspace\Spatial Attention Mechanism for Vehicle Trajectory\Spatial-attention-master\My Data\4_TrackLabeled'
    path = r'F:/Jupyter_Workspace/Spatial Attention Mechanism for Vehicle Trajectory/Spatial-attention-master/My Data/4_TrackLabeled/' + str(num) + '_trackslabel.csv'
    data = pd.read_csv(path)
    del data['Unnamed: 0']
    del data['Unnamed: 0.1']
    data = data.drop(['width','height','frontSightDistance','backSightDistance','dhw','thw','ttc','precedingXVelocity'],axis=1)
    data.rename(columns={'xVelocity':'xV','yVelocity':'yV','xAcceleration':'xA','yAcceleration':'yA'},inplace=True)
    data.loc[data['class']=='Car','class'] = 0
    data.loc[data['class']=='Truck','class'] = 1
    #print('data.shape is:\n',data.shape,'\n\n')
    #data.to_csv(r'data.csv')
    self_id_data = data.set_index(['id','frame'],drop=False)
    all_index = self_id_data.index
    
    #后车-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fol_id_data = data.set_index(['followingId','frame','id'],drop=False)
    fol_index = fol_id_data.index
    fol_index_valid = [x for x in fol_index if x[0] != 0]
    
    #print(len(fol_index_valid))
    
    data_with_following = fol_id_data.loc[fol_index_valid].set_index(['id','frame'],drop = False)
    #print('data_with_following.shape is:\n',data_with_following.shape,'\n\n')
    fol_index_valid = data_with_following.set_index(['followingId','frame']).index
    fol_valid_series = self_id_data.loc[fol_index_valid,['x','y','xV','yV','xA','yA','class']]
    #print(data_with_following.shape)
    #print(fol_valid_series.shape)
    
    fol_valid_series.index = data_with_following.index
    fol_valid_series.columns = ['folx','foly','folxV','folyV','folxA','folyA','folClass']
    data_with_following[['folx','foly','folxV','folyV','folxA','folyA','folClass']] = fol_valid_series
    
    fol_index_invalid = [x for x in fol_index if x[0] == 0]
    
    data_without_following = fol_id_data.loc[fol_index_invalid]
    #print('data_without_following.shape is:\n',data_without_following.shape,'\n\n')
    
    data_without_following_right = data_without_following.loc[data_without_following['drivingDirection']==2]
    data_without_following_right.loc[:,'folx'] = data_without_following_right.loc[:,'x']-200
    data_without_following_right.loc[:,'foly'] = data_without_following_right.loc[:,'y']
    data_without_following_right.loc[:,'folxV'] = data_without_following_right.loc[:,'xV']
    data_without_following_right.loc[:,'folyV'] = data_without_following_right.loc[:,'yV']
    data_without_following_right.loc[:,'folxA'] = data_without_following_right.loc[:,'xA']
    data_without_following_right.loc[:,'folyA'] = data_without_following_right.loc[:,'yA']
    data_without_following_right.loc[:,'folClass'] = data_without_following_right.loc[:,'class']
    
    data_without_following_left = data_without_following.loc[data_without_following['drivingDirection']==1]
    data_without_following_left.loc[:,'folx'] = data_without_following_left.loc[:,'x']+200
    data_without_following_left.loc[:,'foly'] = data_without_following_left.loc[:,'y']
    data_without_following_left.loc[:,'folxV'] = data_without_following_left.loc[:,'xV']
    data_without_following_left.loc[:,'folyV'] = data_without_following_left.loc[:,'yV']
    data_without_following_left.loc[:,'folxA'] = data_without_following_left.loc[:,'xA']
    data_without_following_left.loc[:,'folyA'] = data_without_following_left.loc[:,'yA'] 
    data_without_following_left.loc[:,'folClass'] = data_without_following_left.loc[:,'class'] 
    
    data_processed = pd.concat([data_without_following_left,data_without_following_right,data_with_following],axis=0).sort_index()
    #data_processed = data_processed.drop(['frame','id'],axis=1)
    
    #print(data_processed)
    #data_processed.to_csv(r'data_processed.csv')

    #print(data_processed)
    
    #前车--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    self_id_data = data_processed.set_index(['id','frame'],drop=False) 
    pre_id_data = data_processed.set_index(['precedingId','frame','id'],drop=False)
    
    pre_index = pre_id_data.index
    pre_index_valid = [x for x in pre_index if x[0] != 0]
    
    data_with_preceding = pre_id_data.loc[pre_index_valid].set_index(['id','frame'],drop=False)
    pre_index_valid = data_with_preceding.set_index(['precedingId','frame']).index
    pre_valid_series = self_id_data.loc[pre_index_valid,['x','y','xV','yV','xA','yA','class']]
    pre_valid_series.index = data_with_preceding.index
    
    pre_valid_series.columns = ['prex','prey','prexV','preyV','prexA','preyA','preClass']
    
    data_with_preceding[['prex','prey','prexV','preyV','prexA','preyA','preClass']] = pre_valid_series
    
    pre_index_invalid = [x for x in pre_index if x[0] == 0]
    
    data_without_preceding = pre_id_data.loc[pre_index_invalid]
    
    data_without_preceding_right = data_without_preceding.loc[data_without_preceding['drivingDirection']==2]
    data_without_preceding_right.loc[:,'prex'] = data_without_preceding_right.loc[:,'x']+200
    data_without_preceding_right.loc[:,'prey'] = data_without_preceding_right.loc[:,'y']
    data_without_preceding_right.loc[:,'prexV'] = data_without_preceding_right.loc[:,'xV']
    data_without_preceding_right.loc[:,'preyV'] = data_without_preceding_right.loc[:,'yV']
    data_without_preceding_right.loc[:,'prexA'] = data_without_preceding_right.loc[:,'xA']
    data_without_preceding_right.loc[:,'preyA'] = data_without_preceding_right.loc[:,'yA']
    data_without_preceding_right.loc[:,'preClass'] = data_without_preceding_right.loc[:,'class']
    
    #print(data_without_preceding_right,'\n\n\n\n\n\n\n\n\n\n\n')
    data_without_preceding_left = data_without_preceding.loc[data_without_preceding['drivingDirection']==1]
    data_without_preceding_left.loc[:,'prex'] = data_without_preceding_left.loc[:,'x']-200
    data_without_preceding_left.loc[:,'prey'] = data_without_preceding_left.loc[:,'y']
    data_without_preceding_left.loc[:,'prexV'] = data_without_preceding_left.loc[:,'xV']
    data_without_preceding_left.loc[:,'preyV'] = data_without_preceding_left.loc[:,'yV']
    data_without_preceding_left.loc[:,'prexA'] = data_without_preceding_left.loc[:,'xA']
    data_without_preceding_left.loc[:,'preyA'] = data_without_preceding_left.loc[:,'yA']
    data_without_preceding_left.loc[:,'preClass'] = data_without_preceding_left.loc[:,'class']
    #print(data_without_preceding_left)
    
    data_processed = pd.concat([data_without_preceding_left,data_without_preceding_right,data_with_preceding],axis=0)
    data_processed = data_processed.set_index(['id','frame'],drop=False).sort_index()
    data_processed.to_csv('./data/1_fol_pre_built/'+str(num)+'_fol_pre_built.csv',index=None)
    
for num in range(1,61):
    print(num,'\n\n')
    fol_pre_building(num)