import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import xarray as xr
import numpy as np

import TensorDynamics.sphere_harm_new as sh
from spherical_conv import SphereModel
from training_functions import get_train_step, get_val_step, training_loop
from plot_results import plot_predictions, plot_hs_stats

######################################################################################
########################## Choose parameters #########################################
######################################################################################
# Set which data to use
nlats=64 # latitude
nlevels="15" # vertical levels in HS simulation
interval="12" # save interval in hours

# index for which level to plot (after concatenating variables)
plot_ind=13

# characteristics of the SFNO model
embed_dim=96
nblocks=4
dfact=1.6 # downscaling factor


# set number of total samples, then split into validation and test
n_samples=8000
n_vali=int(n_samples/10)
n_train=n_samples-n_vali

# whether to load, train, save model
load_model=True
trainModel=False
saveModel=False
mname='models/SFNO_model_lat'+str(nlats)+'_ed'+str(embed_dim)+'.keras'


# training hyperparameters
skip_rounds=0
loops_list=[1,2,3,4,5,6,8,10] # rollout lengths to iterate over
lrate_facts = [1,1,0.5,0.5,0.25,0.25,0.125,0.125] # learning rate factors
epochs_list = [2 for i in range(len(loops_list))] # epochs for each rollout length
spec_ratio_list=[0,0.02,0.02,0.02,0.02,0.02,0.02,0.02] # scaling factor for the spectral loss
lrate=0.002 # learning rate

######################################################################################
########################## Load and normalize data ###################################
######################################################################################

# load data, concat along the "level" dimension
data = xr.open_dataset("hs_data/output_hs_"+str(nlats)+"_"+nlevels+"_"+interval+".nc")
sigmas=data["level"].values

values= np.concatenate([np.log(data["surface_pressure"][0:n_samples].values[:,None,:,:]),data["temperature"][0:n_samples].values,data["u_component_of_wind"][0:n_samples].values,data["v_component_of_wind"][0:n_samples].values],axis=1)

# close xarray object to save memory
data.close()
del data

# check for nan values
number_nans=np.sum(np.isnan(values[:,0,0,0]))
if number_nans>0:
    print("Number of nans: ",number_nans)
    quit()

# normalize data along each level, using first 500 data points only
means=np.mean(values[0:500],axis=(0,2,3),keepdims=True)
values=values-means

stds=np.std(values[0:500],axis=(0,2,3),keepdims=True)
values= values/stds

values=tf.constant(values) 

# number of concatenated "levels"
nlevels=np.shape(values)[1]

######################################################################################
########################## DEFINE LOSS FUNCTIONS  ####################################
######################################################################################

# calculate truncation limit, then create spherical harmonic object
trunc=int(nlats/dfact)
grid_obj = sh.sh_obj(nlats=nlats,trunc=trunc)

def L1_loss(y_true,y_pred):
    """
    Calculate MAE, weighted by cosine of latitude
    """
    
    error=tf.math.abs(y_true-y_pred)
    
    coslats=tf.math.cos(grid_obj.lats)
    norm_fact=tf.math.reduce_sum(coslats)*nlats*2*nlevels
    
    loss= tf.math.reduce_sum(error*coslats[:,None])/norm_fact
    return loss

def spec_loss(y_true,y_pred):
    """
    Calculate absolute energy error at each total wavenumber
    """
    
    true_harms=grid_obj.calc_sh_coeffs(y_true)
    pred_harms=grid_obj.calc_sh_coeffs(y_pred[0])

    pred_energy = (tf.math.abs(pred_harms)**2)/(8*nlats**2)
    true_energy = (tf.math.abs(true_harms)**2)/(8*nlats**2)

    pred_energy=2*tf.math.reduce_sum(pred_energy[:,:,1:],axis=-1)+pred_energy[:,:,0]
    true_energy=2*tf.math.reduce_sum(true_energy[:,:,1:],axis=-1)+true_energy[:,:,0]
        
    loss= tf.math.reduce_sum(tf.math.abs(true_energy-pred_energy))/nlevels
    return loss
    
######################################################################################
########################## LOAD, TRAIN, SAVE MODEL  ##################################
######################################################################################

# create the keras model object, with given hyperparameters
model = SphereModel(nlats=np.shape(values)[2],nlevels=np.shape(values)[1],embed_dim=embed_dim,nblocks=nblocks,dfact=dfact)
model(values[0:1])

# if desired, load weights from previous file
if load_model:
    model.load_weights(mname)
model.summary()

zipped_lists=zip(np.arange(len(epochs_list)),lrate_facts,epochs_list,loops_list,spec_ratio_list)

# train the model, looping through each rollout length
if trainModel:
    for round,lrate_fact,epochs,loops,spec_ratio in zipped_lists:
        # whether to skip certain rollout lengths
        if round<skip_rounds:
            continue
            
        print(lrate_fact,epochs,loops,spec_ratio)

        # Define functions for training step and validation step
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lrate*lrate_fact)
        train_step=get_train_step(model,optimizer,L1_loss,spec_loss,spec_ratio,loops)
        val_step=get_val_step(model,L1_loss,spec_loss,loops)
       
        # perform training and validation
        training_loop(values,train_step,val_step,epochs,loops,n_train,n_vali)
        
        if saveModel:
            model.save(mname)


######################################################################################
########################## Plot Results ##############################################
######################################################################################

# time index for plotting (assuming at least 100 validation samples)
tind=-100

# functions for calling the model that are XLA compiled
@tf.function(jit_compile=True)
def call_model(x):
    return model(x)

def infer(x,loops=1):
    for i in range(loops):
        x=call_model(x)
    return x


print("plotting example rollout")
plot_predictions(grid_obj,values,plot_ind,tind,infer,interval,stds,means,sigmas)

print("plotting hs stats")
plot_hs_stats(grid_obj,values,tind,infer,stds,means,sigmas)


