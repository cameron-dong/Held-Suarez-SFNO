import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_predictions(grid_obj,values,plot_ind,tind,infer,interval,stds,means,sigmas):
    """
    Plots model predictions at successively longer rollout lengths
    
    Args:
        grid_obj - spherical harmonic grid object
        values - training/validation data
        plot_ind - index for which "level" to plot
        tind - index for which timestep to plot from
        infer - function that steps forward the keras model
        interval - model step size, in hours
        stds - normalization standard deviation
        means - normalization mean
    """
    latsDeg=grid_obj.lats*180/np.pi
    lonsDeg=grid_obj.lons*180/np.pi
    fig=plt.figure(figsize=[6.5,8])

    mag=40
    c_levels= np.linspace(-mag,mag,num=50)
    
    sigma_val = sigmas[(plot_ind-1)%5]
    # loop over rollout times
    for i in range(4):
        steps=6 # number of model steps between each plot row

        # plot the true values
        plt.subplot(4,2,i*2+1)
        im=plt.contourf(lonsDeg,latsDeg,(values*stds+means)[tind+i*steps,plot_ind],levels=c_levels,extend="both")
        plt.title("Truth ("+str(int(i*steps*int(interval)/24))+ " days)",size=9)

        # formatting
        plt.xticks(ticks=[0,60,120,180,240,300,360],size=6)
        plt.yticks(ticks=[-90,-60,-30,0,30,60,90],size=6)
        plt.ylabel("latitude (°N)",size=7)
        plt.xlabel("longitude (°E)",size=7)

        # plot the model prediction
        plt.subplot(4,2,i*2+2)
        im=plt.contourf(lonsDeg,latsDeg,(infer(values[tind:tind+1,0:],i*steps)*stds+means)[0,plot_ind],levels=c_levels,extend="both")
        plt.title("Prediction ("+str(int(i*steps*int(interval)/24))+ " days)",size=9)

        # formatting
        plt.xticks(ticks=[0,60,120,180,240,300,360],size=6)
        plt.yticks(ticks=[-90,-60,-30,0,30,60,90],size=6)
        plt.ylabel("latitude (°N)",size=7)
        plt.xlabel("longitude (°E)",size=7)


    # add a shared colorbar
    cbar_top=0.12
    fig.subplots_adjust(bottom=cbar_top)
    cbar_ax = fig.add_axes([0.10, 0.11, 0.84, 0.025])
    
    cbar=fig.colorbar(im,orientation='horizontal',label="m/s", cax=cbar_ax,ticks=np.linspace(-mag,mag,num=11))
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label(label="m/s",size=10)

    

    # tight_layout then save figure to file
    plt.suptitle("Meridional Wind (sigma="+str(sigma_val)+")")
    plt.tight_layout(rect=[0,cbar_top,1,1])
    plt.savefig("figures/example_rollout.png",dpi=200)

#############################################################################################################
#############################################################################################################


def plot_hs_stats(grid_obj,values,tind,infer,stds,means,sigmas):
    """
    Performs 1000 day rollout of model, then plots statistics as in Held-Suarez
    
    Args:
        grid_obj - spherical harmonic grid object
        values - training/validation data
        tind - index for which timestep to plot from
        infer - function that steps forward the keras model
        stds - normalization standard deviation
        means - normalization mean
        sigmas - sigma levels for the model
    """
    latsDeg=grid_obj.lats*180/np.pi
    lonsDeg=grid_obj.lons*180/np.pi

    # roll out the model for 1000 days (2000 steps)
    hs_vals=[values[tind:tind+1]]
    for i in range(1000):
        hs_vals.append(infer(hs_vals[i],loops=2))
    hs_vals=np.squeeze(np.array(hs_vals))
    
    # take only the second half of the data
    hs_vals=hs_vals[500:]*stds+means
    
    # Calculate mean state (assuming 5 levels of model data for each variable, plus surface pressure)
    temperature_mean = np.mean(hs_vals[:,1:6],axis=(0))
    u_mean = np.mean(hs_vals[:,6:11],axis=(0))
    
    # calculate temperature eddy variance using anomalies from the mean state
    temperature_eddy = np.mean((hs_vals[:,1:6]-temperature_mean)**2,axis=0)
    
    # for eddy variance of U in spectral space, use periodogram
    U_eddy_variance=np.mean((periodogram((hs_vals[:,6:11]-u_mean),scaling="spectrum")[1]),axis=(0,1))
    
    plt.figure()

    # plot model mean zonal wind, function of latitude/sigma
    plt.subplot(2,2,1)
    plt.contour(latsDeg,sigmas,np.mean(u_mean,axis=-1),levels=np.arange(-32,36,4))
    plt.title("zonal wind mean state")
    plt.xlabel("latitude")
    plt.ylabel("sigma")
    plt.xticks(np.arange(-90,91,30))
    plt.ylim([1, 0])
    plt.colorbar(label="m/s")
    
    # plot model mean temperature, function of latitude/sigma
    plt.subplot(2,2,2)
    plt.title("temperature mean state")
    plt.xlabel("latitude")
    plt.ylabel("sigma")
    plt.contour(latsDeg,sigmas,np.mean(temperature_mean,axis=-1),levels=np.arange(190,306,5))
    plt.colorbar(label="K")
    plt.xticks(np.arange(-90,91,30))
    plt.ylim([1, 0])
    
    # plot zonal wind eddy variance, function of latitude/wavenumber
    plt.subplot(2,2,3)
    plt.contour([m for m in range(np.shape(U_eddy_variance)[1])],latsDeg,U_eddy_variance,levels=np.arange(0,20,2))
    plt.xlim([0, 15])
    plt.title("zonal wind eddy variance")
    plt.colorbar(label=r"m$^2$ s$^{-2}$")
    plt.xlabel("zonal wavenumber")
    plt.ylabel("latitude")
    plt.yticks([-90,-45,0,45,90])
    plt.gca().set_aspect("auto")
    
    
    # plot temperature eddy variance, function of latitude/sigma
    plt.subplot(2,2,4)
    plt.title("temperature eddy variance")
    plt.xlabel("latitude")
    plt.ylabel("sigma")
    plt.contour(latsDeg,sigmas,np.mean(temperature_eddy,axis=-1),levels=np.arange(0,50,5))
    plt.colorbar(label=r"K$^2$")
    plt.xticks(np.arange(-90,91,30))
    plt.ylim([1, 0])

    # tight_layout then save figure to file
    plt.tight_layout()
    plt.savefig("figures/hs_stats_SFNO.png")