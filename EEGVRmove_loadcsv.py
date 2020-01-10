#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:47:17 2019

@author: bolger
"""

import pandas as pd
import os
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import re
#import pysal


## Function definitions here

def velocity_calc(dataIn,time):
    #calculate the euclidean distance between X Y Z data
    from scipy.spatial import distance
    
    xyzdist = distance.pdist(dataInxyz,metric='euclidean')
    xyzdists = distance.squareform(xyzdist, force='no')
    D = np.diagonal(xyzdists,offset=1)    #the first upper diagonal
    vel = D/np.diff(time)
    return (vel, xyzdists)

def acceleration_calc(velIn,time):
    
    veldiff = np.diff(velIn)  #first-order velocity difference
    tdiff = np.diff(time)
    accel = veldiff/tdiff[1:]
    return accel

def straightness_calc(distarray):
    lowerdist = np.diagonal(distarray,offset=-1)
    upperdist = np.diagonal(distarray, offset=1)
    uplowdist = np.diagonal(distarray, offset=2)
    S = (lowerdist[1:]+upperdist[1:])/uplowdist
    return S

def samplent_time(dataIn, sampent_in, T, frac):         #Function to calculate sample entropy over time
    
    import nolds
    
    if len(sampent_in) ==0:
        n = np.size(dataIn,axis=0)
        n= round(n/frac)
        sampent_out = nolds.sampen(dataIn[0:n,],emb_dim=2)
    
    n2 = np.size(dataIn, axis=0)
    n2b = round(n2/frac)
    n_all = range(n2b,n2)
    for i in n_all:
            S = nolds.sampen(dataIn[0:i,],emb_dim=2)
            sampent_out = np.append(sampent_out,S)
    
    return (sampent_out, n2b)

#---------------------------------------------------
# Define paths
#---------------------------------------------------    
    
basepath1 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials05/"
entries1 = glob.glob(os.path.join(basepath1,'*.txt'))
basepath2 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials06/"
entries2 = glob.glob(os.path.join(basepath2,'*.txt'))
basepath3 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials07/"
entries3 = glob.glob(os.path.join(basepath3,'*.txt'))
basepath4 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials08/"
entries4 = glob.glob(os.path.join(basepath4,'*.txt'))
basepath5 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials09/"
entries5 = glob.glob(os.path.join(basepath5,'*.txt'))
basepath6 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials10/"
entries6 = glob.glob(os.path.join(basepath6,'*.txt'))
basepath7 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials11/"
entries7 = glob.glob(os.path.join(basepath7,'*.txt'))
basepath8 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials12/"
entries8 = glob.glob(os.path.join(basepath8,'*.txt'))
basepath9 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials15/"
entries9 = glob.glob(os.path.join(basepath9,'*.txt'))
basepath10 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials18/"
entries10 = glob.glob(os.path.join(basepath10,'*.txt'))
basepath11 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials20/"
entries11 = glob.glob(os.path.join(basepath11,'*.txt'))
basepath12 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials21/"
entries12 = glob.glob(os.path.join(basepath12,'*.txt'))
basepath13 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials22/"
entries13 = glob.glob(os.path.join(basepath13,'*.txt'))
basepath14 = "/Volumes/deepassport/Projects/Project_EEGVR_move/Model-movement-complexity/Trials23/"
entries14 = glob.glob(os.path.join(basepath14,'*.txt'))

#Concatenate the entries list
entries = entries1+entries2+entries3+entries4+entries5+entries6+entries7+entries8+entries9+entries10+entries11+entries12+entries13+entries14
verboi = 'empiler'

#Can read through the entries list
fig = plt.figure()
figvel = plt.figure()
figaccel = plt.figure()
figstrait = plt.figure()
figsampent = plt.figure()

##Prepare the files to write the movement features to
filepath = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/"

fIDvelt = open(os.path.join(filepath,'veltime-'+verboi+'.txt'),'w+') 
fIDvel  = open(os.path.join(filepath,'velocitydata-'+verboi+'.txt'), 'w+')
fIDaccel  = open(os.path.join(filepath,'acceldata-'+verboi+'.txt'), 'w+') 
fIDstrait  = open(os.path.join(filepath,'straitdata-'+verboi+'.txt'), 'w+') 
fIDtrial = open(os.path.join(filepath,'trialinfo-'+verboi+'.txt'), 'w+')

i = 1
sampent_vel = np.zeros((30,),dtype=float)
sampent_accel = np.zeros((30,),dtype=float)
sampent_strait = np.zeros((30,),dtype=float)
veldata_all = []
veltime_all = []
sampent_time = []
trialcurr = []
sujcurr = []
sujtitre = []
trialindx = []


rows = 5
cols = 6

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'bold',
        'size': 8,
        }
bandcntr = 0
for cntr in entries[0:np.size(entries,0)]: 
    
    trial_curr = pd.read_csv(cntr,sep = '\t', header = None, nrows = 5)
    curr_verb = trial_curr.loc[0,1]
    trialtype = trial_curr.loc[3,1]
        
    if verboi == curr_verb and trialtype == 'GO' and cntr[-16:] !='END_OF_BLOCK.txt':
        
        r = cntr.find('Trial_Order')
        s = cntr.find('Trials')
        tcurr = cntr[r:r+13]
        scurr = cntr[s+6:s+8]
        if tcurr[-1] == '_':
             tcurr = cntr[r:r+12]
        trialcurr.append(tcurr)
        sujcurr.append(scurr)
        print(cntr)
        titrecurr = scurr+'-'+tcurr
        sujtitre.append(titrecurr)  # write to file
        tcurr_int = int(tcurr[11:])
        trialindx.append(tcurr_int) # write to file
        fIDtrial.write("%s\t %d\r\n" % (titrecurr,tcurr_int))
        
         
        dataIn = pd.read_csv(cntr,sep = '\t', header = 0, names = ['SessionTime', 'TryTime','Phase', 'hand_posx', 'hand_posy', 'hand_posz', 'hand_rotx', 'hand_roty', 'hand_rotz'],  usecols=['SessionTime', 'TryTime', 'Phase', 'hand_posx', 'hand_posy', 'hand_posz', 'hand_rotx', 'hand_roty', 'hand_rotz'], skiprows = 5, index_col = ['Phase'])
        
        
        action_curr = dataIn.loc['Action']
        data_posx = action_curr['hand_posx']
        data_posy = action_curr['hand_posy']
        data_posz = action_curr['hand_posz']
        handposx = data_posx.values
        handposy = data_posy.values
        handposz = data_posz.values
        
        trytime = action_curr['TryTime'] 
        T = trytime.values    # Returns the time vector as an ndarray
        sessiont = action_curr['SessionTime']
        sessT = sessiont.values
        
        diffT =  np.diff(T)
        diffST = np.diff(sessT)
        
        from scipy import interpolate
    
        # Creating a spline representation of the curve
        tck_handposx = interpolate.splrep(T,handposx)
        tck_handposy = interpolate.splrep(T,handposy)
        tck_handposz = interpolate.splrep(T,handposz)
        
        Tnew = np.arange(T[0],T[-1],.05)
        np.savetxt(fIDvelt,[Tnew],'%f',delimiter=',',newline='\n')
        
        ynew_handposx = interpolate.splev(Tnew,tck_handposx)
        ynew_handposy = interpolate.splev(Tnew,tck_handposy)
        ynew_handposz = interpolate.splev(Tnew,tck_handposz)
        dataxyz = np.array((ynew_handposx,ynew_handposy, ynew_handposz))
        dataInxyz = dataxyz.transpose()
        
    
       
        # Apply a gradient function to be able to create quiver plot
        handposx_grad = np.gradient(ynew_handposx)
        handposy_grad = np.gradient(ynew_handposy)
        handposz_grad = np.gradient(ynew_handposz)
        
        spacers = '-'
        str_title = trial_curr.loc[0,1]+spacers+ trial_curr.loc[3,1]+spacers+trial_curr.loc[4,1]
        
        ax = fig.add_subplot(rows,cols,i, projection='3d')
        ax.quiver(ynew_handposx,ynew_handposy,ynew_handposz,handposx_grad,handposy_grad,handposz_grad)    
        ax.scatter(ynew_handposx[0],ynew_handposy[0],ynew_handposz[0],s=50, c='g')
        ax.scatter(ynew_handposx[-1],ynew_handposy[-1],ynew_handposz[-1],s=50, c='r')
        ax.set_title(str_title ,fontsize=8)
        ax.set_xlabel('posX',fontsize=8)
        ax.set_ylabel('posY',fontsize=8)
        ax.set_zlabel('posZ',fontsize=8)
        ax.tick_params(labelsize=6)
        fig.suptitle('Trajectory: XYZ coordinates')
        
        
        velout, distall = velocity_calc(dataInxyz,Tnew)   #Call of function to calculate velocity
        accelout = acceleration_calc(velout,Tnew)
        straitout = straightness_calc(distall)
        
        # Create a list of arrays for each verb time series
        veldata_all.append(velout)
        veltime_all.append(Tnew[1:])
        
        
        velmean = [np.mean(velout)] * np.size(velout,0)
        accelmean = [np.mean(accelout)] * np.size(accelout,0)
        straitmean = [np.mean(straitout)] * np.size(straitout,0)
        velstd = np.std(velout)
        accelstd = np.std(accelout)
        stratstd = np.std(straitout)
        
        ax1 = figvel.add_subplot(rows,cols,i)
        ax1.plot(Tnew[1:],velout,Tnew[1:],velmean,'r--')
        ax1.text(Tnew[-1], velmean[0], r'mean',fontdict=font)
        ax1.set_title(str_title ,fontsize=8)
        ax1.tick_params(labelsize=6)
        figvel.suptitle('Velocity Profile')
        
        ax2 = figaccel.add_subplot(rows,cols,i)
        ax2.plot(Tnew[2:],accelout,Tnew[2:],accelmean,'r--')
        ax2.text(Tnew[-1], accelmean[0], r'mean',fontdict=font)
        ax2.set_title(str_title ,fontsize=8)
        ax2.tick_params(labelsize=6)
        figaccel.suptitle('Acceleration Profile')
        
        ax3 = figstrait.add_subplot(rows,cols,i)
        ax3.plot(Tnew[2:],straitout,Tnew[2:],straitmean,'r--')
        ax3.text(Tnew[-1], straitmean[0], r'mean',fontdict=font)
        ax3.set_title(str_title ,fontsize=8)
        ax3.tick_params(labelsize=6)
        figstrait.suptitle('Straightness Profile')
        
        import nolds
        #import sampen
        
        svel = nolds.sampen(velout,emb_dim=2)   #calculate sample entroy
        sampent_vel[i-1,] = svel*100
        sampent_accel[i-1,] = nolds.sampen(accelout,emb_dim=2)
        sampent_strait[i-1,] = nolds.sampen(straitout,emb_dim=2)
        
        sampentvel_mean = np.mean(sampent_vel)
        sampentvel_std = np.std(sampent_vel)
        
        SIn = []
        SOut, Nb = samplent_time(velout, SIn, Tnew, 2)
        sampent_time.append(SOut)
        
        ax4 = figsampent.add_subplot(rows,cols,i)
        ax4.plot(Tnew[Nb:,], SOut)
        ax4.set_title(str_title ,fontsize=8)
        ax4.tick_params(labelsize=6)
        ax4.set_ylim([0,2])
        figsampent.suptitle('Sample Entropy Profile')
        
        i = i+1
    
    else:
    
        print("Current verb:", trial_curr.loc[0,1]) 
        

sampent_data = [sampent_vel, sampent_accel, sampent_strait]
figsamp = plt.figure()
ax4 = figsamp.gca()
bp = ax4.boxplot(sampent_data,1, labels=['velocity', 'acceleration','straightness'],patch_artist = True)
ax4.set_xlabel('Motion Features')
ax4.set_ylabel('Sample Entropy')
ax4.set_title('Verb: '+verboi)

for patch in bp['boxes']:
  patch.set_color('skyblue')
  patch.set_edgecolor('black')
  
fIDvel.close()
fIDtrial .close()

from tslearn.utils import save_timeseries_txt

filepath_full = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/velocitydata-"+ verboi+".txt"
filepath_time = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/veltime-"+ verboi+".txt"
filepath_sampent = "/Users/bolger/Documents/work/Projects/Project-VRMove/AnalysisFiles/sampent-"+verboi+".txt"

save_timeseries_txt(filepath_full,veldata_all)
save_timeseries_txt(filepath_time,veltime_all)
save_timeseries_txt(filepath_sampent,sampent_time)



