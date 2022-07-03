import os
import glob
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as inter




class GETDATA():
    def __init__(self, dir):
        print ('loading data from:', dir)
        
        self.data_paths = glob.glob(os.path.join(dir, 's*', '*','*','*.txt'))
        self.data_paths.sort()
        
    
    def get_data(self, test_set_folder=3):
        
        
        cross_set = {}
        cross_set[0] = ['s01s02', 's03s04', 's05s02', 's06s04']
        cross_set[1] = ['ds02s03', 's02s07', 's03s05', 's05s03']
        cross_set[2] = ['s01s03', 's01s07', 's07s01', 's07s03']
        cross_set[3] = ['s02s01', 's02s06', 's03s02', 's03s06']
        cross_set[4] = ['s04s02', 's04s03', 's04s06', 's06s02', 's06s03']
        
        def read_txt(path):
            r_csv = pd.read_csv(path,header=None).T
            r_csv = r_csv[1:]
            return r_csv.values
        

        train_set = []
        test_set = []
        for i in range(len(cross_set)):
            if i == test_set_folder:
                test_set += cross_set[i]
            else:
                train_set += cross_set[i]

        train = {} 
        test = {}

        for i in range(1,9):
            train[i] = []
            test[i] = []

        for data_path in self.data_paths:
            skeletal_data = read_txt(data_path)
            if data_path.split('/')[-4] in train_set:   
                train[int(data_path.split('/')[-3])].append(skeletal_data) 
            else:
                test[int(data_path.split('/')[-3])].append(skeletal_data) 

        return train, test
        



        
        
#Transfer to orginial coordinates for plotting
def coord2org(normalized_coord): 
    orignal_coord = np.empty_like(normalized_coord)
    for i in range(15):
        orignal_coord[i,0] = 640 - (normalized_coord[i,0] * 640)
        orignal_coord[i,1] = 480 - (normalized_coord[i,1] * 240)
    return orignal_coord






#Plotting the pose
def draw_2d_pose(skel_points): 
    
    # connections of the joints
    f_ind = np.array([
        [2,1,0],
        [3,6,2,3],
        [3,4,5],
        [6,7,8],
        [2,12,13,14],
        [2,9,10,11],
    ])

    fig = plt.figure()
    
    axes = plt.gca()
    axes.set_xlim([0,640])
    axes.set_ylim([0,480])

    ax = fig.add_subplot(111)
    
    
    for gtorig,color in zip(skel_points,['r','b']):
        
        gtorig = coord2org(gtorig)
        
        for i in range(f_ind.shape[0]):
        
            ax.plot(gtorig[f_ind[i], 0], gtorig[f_ind[i], 1], c=color)
            ax.scatter(gtorig[f_ind[i], 0], gtorig[f_ind[i], 1],s=10,c=color)
        
    plt.show()

    

    

#Switch two persons' position
#used for mirror augmentation
def mirror(person1,person2):
    person1_new = np.copy(person1)
    person2_new = np.copy(person2)
    person1_new[:,:,0] = abs(person1_new[:,:,0]-1)
    person2_new[:,:,0] = abs(person2_new[:,:,0]-1)
    return person1_new, person2_new





#Rescale to be 16 frames
def zoom(p):
    l = p.shape[0]
    p_new = np.empty([16,15,3]) 
    for m in range(15):
        for n in range(3):
            p_new[:,m,n] = inter.zoom(p[:,m,n],16/l)[:16]
    return p_new



def get_person_pose(original_pose):

    pose = np.copy(original_pose)
    pose = pose.reshape([-1,15,3])

    frames = pose.shape[0]           # the number of all frames

    if frames>16:                   # sample the range from crop size of [16,frames]
        ratio = np.random.uniform(1,frames/16)
        l = int(16*ratio)
        start = random.sample(range(frames-l),1)[0]
        end = start+l
        pose = pose[start:end,:,:]
        pose = zoom(pose)
        
    elif frames<16:
        pose = zoom(pose)
    
    return pose



def temporal_difference(skel_posture):

    temp_diff = skel_posture[1:,:,:] - skel_posture[:-1,:,:]
    temp_diff = np.concatenate((temp_diff, np.expand_dims(temp_diff[-1,:,:], axis=0)))
    return temp_diff


