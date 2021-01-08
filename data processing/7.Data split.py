import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

#print(pd.__version__,'\n\n')


def data_split(random_sta=42):
    path1 = r'F:/Jupyter_Workspace/Spatial Attention Mechanism for Vehicle Trajectory/Spatial-attention-master/HighD Reprocessing/data/4_tracks/track_input.npy'
    path2 = r'F:/Jupyter_Workspace/Spatial Attention Mechanism for Vehicle Trajectory/Spatial-attention-master/HighD Reprocessing/data/4_tracks/track_output.npy'

    track_input = np.load(path1)
    track_output = np.load(path2)
    print("the shape of track_input is:",track_input.shape)
    test_size = int(track_input.shape[0] * 0.1)
    track_output = track_output[:,:,:2]
    print('the size of track_input is:\n',track_input.shape,'\n')
    print('the size of track_output is:\n',track_output.shape,'\n')
    
    
    print('test_size is:\n',test_size,'\n\n')
    x_test=np.zeros(shape=[test_size,15,63])
    y_test=np.zeros(shape=[test_size,26,2])

    sample_pos = random.sample(range(track_input.shape[0]), test_size)
    
    print('track_input.shape[0] is:\n',track_input.shape[0],'\n\n')

    print('sample_pos.shape is:\n',len(sample_pos),'\n\n')
    
    counter = 0
    for sample in sample_pos:
        x_test[counter,:,:] = track_input[sample,:,:]
        y_test[counter,:,:] = track_output[sample,:,:]
        counter = counter+1
    print(counter)
    x_train,x_validation,y_train,y_validation = train_test_split(track_input,track_output, test_size=0.1, random_state = random_sta)
    
#     np.save(r'./data/5_data_split/x_test.npy',x_test)
#     np.save(r'./data/5_data_split/x_train.npy',x_train)
#     np.save(r'./data/5_data_split/x_validation.npy',x_validation)
#     np.save(r'./data/5_data_split/y_test_allinfo.npy',y_test)
#     np.save(r'./data/5_data_split/y_train_allinfo.npy',y_train)
#     np.save(r'./data/5_data_split/y_validation_allinfo.npy',y_validation)  
    np.save(r'./data/5_data_split/x_test_no_lanechange.npy',x_test)
    np.save(r'./data/5_data_split/x_train_no_lanechange.npy',x_train)
    np.save(r'./data/5_data_split/x_validation_no_lanechange.npy',x_validation)
    np.save(r'./data/5_data_split/y_test_no_lanechange.npy',y_test)
    np.save(r'./data/5_data_split/y_train_no_lanechange.npy',y_train)
    np.save(r'./data/5_data_split/y_validation_no_lanechange.npy',y_validation)  
    
data_split()