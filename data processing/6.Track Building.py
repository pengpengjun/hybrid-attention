import pandas as pd
import numpy as np

def track_building(num):
    global lane_change_number
    global no_lane_change_number
    global temp_j
    
    path = r'./data/3_rel_built/' + str(num) + '_rel_built.csv'
    data = pd.read_csv(path)
    data = data.drop(['laneId','numFrames','label'],axis=1)
    data = data.set_index(['id','frame']).sort_index()
    data = data.reset_index()
    order = ['id','frame','numLaneChanges','drivingDirection',
            'x','y','xV','yV','xA','yA','class',
            'RelFolx','RelFoly','RelFolxV','RelFolyV','RelFolxA','RelFolyA','folClass',
            'RelPrex','RelPrey','RelPrexV','RelPreyV','RelPrexA','RelPreyA','preClass',
            'RelLeftprex','RelLeftprey','RelLeftprexV','RelLeftpreyV','RelLeftprexA','RelLeftpreyA','leftPreClass',
            'RelLeftalongx','RelLeftalongy','RelLeftalongxV','RelLeftalongyV','RelLeftalongxA','RelLeftalongyA','leftAlongClass',
            'RelLeftfolx','RelLeftfoly','RelLeftfolxV','RelLeftfolyV','RelLeftfolxA','RelLeftfolyA','leftFolClass',
            'RelRightprex','RelRightprey','RelRightprexV','RelRightpreyV','RelRightprexA','RelRightpreyA','rightPreClass',
            'RelRightalongx','RelRightalongy','RelRightalongxV','RelRightalongyV','RelRightalongxA','RelRightalongyA','rightAlongClass',
            'RelRightfolx','RelRightfoly','RelRightfolxV','RelRightfolyV','RelRightfolxA','RelRightfolyA','rightFolClass'
           ]
    
    data = data[order]
    #data.to_csv('test.csv')
    
    #print(data)
    
    def direction_rebuild(data):
        
        max_x = data.loc[:,'x'].max()
        max_y = data.loc[:,'y'].max()
        
        data.loc[:,'x'] = max_x - data.loc[:,'x']
        data.loc[:,'y'] = max_y - data.loc[:,'y']
        
        data.loc[:,'RelFolx'] = -data.loc[:,'RelFolx']
        data.loc[:,'RelFoly'] = -data.loc[:,'RelFoly']
        data.loc[:,'RelFolxV'] = -data.loc[:,'RelFolxV']
        data.loc[:,'RelFolyV'] = -data.loc[:,'RelFolyV']
        data.loc[:,'RelFolxA'] = -data.loc[:,'RelFolxA']
        data.loc[:,'RelFolyA'] = -data.loc[:,'RelFolyA']
        
        data.loc[:,'RelPrex'] = -data.loc[:,'RelPrex']
        data.loc[:,'RelPrey'] = -data.loc[:,'RelPrey']
        data.loc[:,'RelPrexV'] = -data.loc[:,'RelPrexV']
        data.loc[:,'RelPreyV'] = -data.loc[:,'RelPreyV']
        data.loc[:,'RelPrexA'] = -data.loc[:,'RelPrexA']
        data.loc[:,'RelPreyA'] = -data.loc[:,'RelPreyA']
        
        data.loc[:,'RelLeftprex'] = -data.loc[:,'RelLeftprex']
        data.loc[:,'RelLeftprey'] = -data.loc[:,'RelLeftprey']
        data.loc[:,'RelLeftprexV'] = -data.loc[:,'RelLeftprexV']
        data.loc[:,'RelLeftpreyV'] = -data.loc[:,'RelLeftpreyV']
        data.loc[:,'RelLeftprexA'] = -data.loc[:,'RelLeftprexA']
        data.loc[:,'RelLeftpreyA'] = -data.loc[:,'RelLeftpreyA']
        
        data.loc[:,'RelLeftalongx'] = -data.loc[:,'RelLeftalongx']
        data.loc[:,'RelLeftalongy'] = -data.loc[:,'RelLeftalongy']
        data.loc[:,'RelLeftalongxV'] = -data.loc[:,'RelLeftalongxV']
        data.loc[:,'RelLeftalongyV'] = -data.loc[:,'RelLeftalongyV']
        data.loc[:,'RelLeftalongxA'] = -data.loc[:,'RelLeftalongxA']
        data.loc[:,'RelLeftalongyA'] = -data.loc[:,'RelLeftalongyA']
        
        data.loc[:,'RelLeftfolx'] = -data.loc[:,'RelLeftfolx']
        data.loc[:,'RelLeftfoly'] = -data.loc[:,'RelLeftfoly']
        data.loc[:,'RelLeftfolxV'] = -data.loc[:,'RelLeftfolxV']
        data.loc[:,'RelLeftfolyV'] = -data.loc[:,'RelLeftfolyV']
        data.loc[:,'RelLeftfolxA'] = -data.loc[:,'RelLeftfolxA']
        data.loc[:,'RelLeftfolyA'] = -data.loc[:,'RelLeftfolyA']
        
        data.loc[:,'RelRightprex'] = -data.loc[:,'RelRightprex']
        data.loc[:,'RelRightprey'] = -data.loc[:,'RelRightprey']
        data.loc[:,'RelRightprexV'] = -data.loc[:,'RelRightprexV']
        data.loc[:,'RelRightpreyV'] = -data.loc[:,'RelRightpreyV']
        data.loc[:,'RelRightprexA'] = -data.loc[:,'RelRightprexA']
        data.loc[:,'RelRightpreyA'] = -data.loc[:,'RelRightpreyA']
        
        data.loc[:,'RelRightalongx'] = -data.loc[:,'RelRightalongx']
        data.loc[:,'RelRightalongy'] = -data.loc[:,'RelRightalongy']
        data.loc[:,'RelRightalongxV'] = -data.loc[:,'RelRightalongxV']
        data.loc[:,'RelRightalongyV'] = -data.loc[:,'RelRightalongyV']
        data.loc[:,'RelRightalongxA'] = -data.loc[:,'RelRightalongxA']
        data.loc[:,'RelRightalongyA'] = -data.loc[:,'RelRightalongyA']
        
        data.loc[:,'RelRightfolx'] = -data.loc[:,'RelRightfolx']
        data.loc[:,'RelRightfoly'] = -data.loc[:,'RelRightfoly']
        data.loc[:,'RelRightfolxV'] = -data.loc[:,'RelRightfolxV']
        data.loc[:,'RelRightfolyV'] = -data.loc[:,'RelRightfolyV']
        data.loc[:,'RelRightfolxA'] = -data.loc[:,'RelRightfolxA']
        data.loc[:,'RelRightfolyA'] = -data.loc[:,'RelRightfolyA']
        
        return data
    
    def No_Lane_Change(data,direction):
        
        global j
        global track_input
        global track_output
        global lane_change_number
        global no_lane_change_number
        global temp_j
        
        data = data[data['drivingDirection']==direction]
        
        if direction == 1:
            data = direction_rebuild(data)
        
        data = data[data['frame']%5 == 0]
        
        data.index = range(data.shape[0])
        data.index.name = 'counter'
        
        vehicle_id = data.loc[:,'id'].drop_duplicates().values
        
        data = data.reset_index().set_index(['id'],drop=False)
        #print(vehicle_id)
        for vehicle in vehicle_id:
            #print(vehicle)
            #if vehicle%50 == 0:
                #print("The process is now %.2f%%"%(100*vehicle/vehicle_id.max()))
            #print("j original:",j,'\n\n')
            #start_time = time.process_time()
            vehicle_data = data.loc[vehicle]
            shape = vehicle_data.shape

            if shape[0] >= 40 and len(shape) == 2:
                vehicle_data = vehicle_data.set_index('counter',drop=False)
                slice_begin = 0
                
                while slice_begin + 40 <= shape[0]:
                    track_input[j,:,:] = vehicle_data.iloc[slice_begin:slice_begin+15,5:].values
                    track_output[j,:,:] = vehicle_data.iloc[slice_begin+14:slice_begin+40,5:].values
                    j += 1
                    slice_begin += 5
        #print('------------------------------------------------------------------------------------')
        
    def Lane_Change(data,direction):
        
        global j
        global track_input
        global track_output
        global lane_change_number
        global no_lane_change_number
        global temp_j


        data = data[data['drivingDirection']==direction]
        
        if direction == 1:
            data = direction_rebuild(data)
            
        for mode in range(5):
            data_mode = data[data['frame']%5 == mode]

            data_mode.index = range(data_mode.shape[0])
            data_mode.index.name = 'counter'

            vehicle_id = data_mode.loc[:,'id'].drop_duplicates().values

            data_mode = data_mode.reset_index().set_index(['id'],drop=False)
            #print(vehicle_id)
            for vehicle in vehicle_id:
                #if vehicle%5 == 0:
                    #print("The process is now %.2f%%"%(100*vehicle/vehicle_id.max()))
                #print("j original:",j,'\n\n')
                #start_time = time.process_time()
                vehicle_data_mode = data_mode.loc[vehicle]
                shape = vehicle_data_mode.shape
                
                if shape[0] >= 40 and len(shape) == 2:
                    vehicle_data_mode = vehicle_data_mode.set_index('counter',drop=False)
                    slice_begin = 0

                    while slice_begin + 40 <= shape[0]:
                        track_input[j,:,:] = vehicle_data_mode.iloc[slice_begin:slice_begin+15,5:].values
                        track_output[j,:,:] = vehicle_data_mode.iloc[slice_begin+14:slice_begin+40,5:].values
                        j += 1
                        slice_begin += 5
            #print('------------------------------------------------------------------------------------')
        
    data_nolanechange = data[data['numLaneChanges'] == 0]
    data_lanechange = data[data['numLaneChanges'] > 0]
    
    #No_Lane_Change(data_nolanechange,1)
    Lane_Change(data_lanechange,1)
    #temp_j = j
    #No_Lane_Change(data_nolanechange,2)
    #no_lane_change_number += j - temp_j
    #temp_j = j
    #Lane_Change(data_lanechange,2)
    #lane_change_number += j - temp_j
    #print("no_lane_change_number:",no_lane_change_number)
    #print("lane_change_number:",lane_change_number)
    #temp_j = j
    No_Lane_Change(data_nolanechange,2)
    
  
global j
global track_input
global track_output
global lane_change_number
global no_lane_change_number
global temp_j


shape1=[1300000,15,63]
shape2=[1300000,26,63]

j = 0
temp_j = 0
lane_change_number = 0
no_lane_change_number = 0

track_input = np.zeros(shape=shape1)
track_output = np.zeros(shape=shape2)

for num in range(1,61):
    print(num,'\n\n')
    track_building(num)

print("j is:",j)
np.save(r'./data/4_tracks/track_input' + '.npy',track_input[0:j,:,:])
np.save(r'./data/4_tracks/track_output' + '.npy',track_output[0:j,:,:])
