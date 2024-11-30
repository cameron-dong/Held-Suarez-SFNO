"""
Performs Held-Suarez Simulations, saving the output data to train SFNO
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import TensorDynamics.model_def as md
import TensorDynamics.constants as constants

import matplotlib.pyplot as plt
from scipy.signal import periodogram
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import time

#############################################################################################
# CREATE THE DYNAMICAL CORE MODEL OBJECT

# Define some parameters of the model
nlevels=15 # number of vertical levels
trunc=42 # spectral truncation limit
nlats=64 # number of gaussian latitudes

# create the atmospheric model
int_type="leap"
model=md.model(nlats,trunc,nlevels,int_type=int_type,physics_list=["held_suarez"])

# runtime of 100 days
days=1000
runtime=48*days # in units of hours

# create arrays for lats, lons, sigmas for ease of use
lats=model.f_obj.lats
wlats=lats[None,:,None] # wide lats array for broadcasting
lons=model.f_obj.lons
wlons=lons[None,None,:] # wide lons array for broadcasting
sigmas=model.sigmas
wsigmas=model.sigmas[:,None,None] # wide sigmas array for broadcasting

# latitudes in degrees, for plotting
latsDeg=lats*180/np.pi
#############################################################################################
# CREATE THE INITIAL FIELD

# some constants
A_EARTH=constants.A_EARTH

# initial wind and temperature fields
u_init=tf.zeros(shape=(nlevels,nlats,nlats*2),dtype=np.single)
v_init=tf.zeros_like(u_init)
T_init=tf.random.uniform((nlevels,nlats,nlats*2),maxval=0.001,dtype=np.single)+300 # add small perturbations to break equilibrium

# surface geopotential
geopot_surface=tf.zeros(shape=(1,nlats,nlats*2),dtype=np.single)

# surface pressure
PS=(tf.ones(shape=(1,len(lats),len(lons)),dtype=np.single)*100000)

# create dictionaries for the steady state initial conditions and the perturbed initial conditions
# set specific humidity to zero
init_state={"surface_pressure":PS,"temperature":T_init,"specific_humidity":T_init*0,
	"u_component_of_wind":u_init,"v_component_of_wind":v_init,"geopotential_at_surface":geopot_surface}

#############################################################################################
# RUN THE SIMULATIONS

stime=time.time()
print("starting sims")

# do a spinup simulation
final_state=model.stepper(runtime*4,init_state)

print(time.time()-stime)

# output parameters
output_int=12 # hours
history=[]
out_vars=["u_component_of_wind","v_component_of_wind","temperature"]

# perform the HS simulation, saving output
for i in range(10):
    print(i,time.time()-stime,np.mean(final_state["u_component_of_wind"]))
    final_state,tmp_history=model.stepper(runtime*2,final_state,output_interval=output_int)
    for th in tmp_history:
        desired = {key: th[key][1::3] for key in out_vars}
        desired["surface_pressure"]=th["surface_pressure"]
        history.append(desired)
    del tmp_history

print(time.time()-stime)


#############################################################################################
# OUTPUT RESULTS TO FILE FOR USE WITH SFNO

import xarray as xr

# organize the lists into numpy arrays
u_component_of_wind = np.stack([h["u_component_of_wind"] for h in history])
v_component_of_wind = np.stack([h["v_component_of_wind"] for h in history])
temp = np.stack([h["temperature"] for h in history])
surface_pressure = np.squeeze(np.stack([h["surface_pressure"] for h in history]))

# create the xarray dataset
ds = xr.Dataset(
    data_vars=dict(
        temperature=(["time", "level", "lat","lon"], temp),
        u_component_of_wind=(["time", "level", "lat","lon"], u_component_of_wind),
        v_component_of_wind=(["time", "level", "lat","lon"], v_component_of_wind),
        surface_pressure=(["time", "lat","lon"], surface_pressure),
    ),
    coords=dict(
        lon=model.f_obj.lons,
        lat=model.f_obj.lats,
        level=model.sigmas[1::3],
        time=[i*output_int for i in range(np.shape(temp)[0])],
    ),
    attrs=dict(description="Held-Suarez data."),
)

# save the dataset to file
ds.to_netcdf("hs_data/output_hs_"+str(nlats)+"_"+str(nlevels)+"_"+str(output_int)+".nc",engine="scipy")

