#!/usr/bin/env python
# coding: utf-8

# In[49]:




# IMPORT
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from netCDF4 import Dataset
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import xarray as xr
import glob
from datetime import datetime, timedelta
from numba import jit


@jit()
def rms(true,prediction):
    return np.sqrt(np.nanmean((prediction-true)**2))

@jit()
def mean_error(true,prediction):
    return np.nanmean(prediction - true)

@jit()
def rms_tendency(variable,hours):
    variable_tendency = np.zeros((hours))
    variable = np.exp(variable) * 1000.0
    for i in range(hours):
        variable_tendency[i] = np.sqrt(np.mean((variable[i+1] - variable[i])**2.0))
    return variable_tendency

def latituded_weighted_rmse(true,prediction,lats):
    diff = prediction-true
    weights = np.cos(np.deg2rad(lats))
    weights2d = np.zeros(np.shape(diff))
    diff_squared = diff**2.0
    #weights = np.ones((10,96))
    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)
    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
    weighted_average = np.ma.average(masked,weights=weights2d)
    return np.sqrt(weighted_average)

def latituded_weighted_bias(true,prediction,lats):
    diff = prediction-true
#     print(diff[0])
    weights = np.cos(np.deg2rad(lats))
#     print(weights[0])
    weights2d = np.zeros(np.shape(diff))
    diff_squared = diff
    #weights = np.ones((10,96))
    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)
#     print(weights2d)
    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
#     print(masked[0])
    weighted_average = np.ma.average(masked,weights=weights2d)
#     print(weighted_average)
    return weighted_average



# In[51]:


# take climo surf pressure at each gridpoint of letkfspeedy 1.9 28 years and then do the same with ERA5 
#speedy_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_1_9_19810101_20100101/mean_output/out.nc'
speedy_file = '/skydata2/dylanelliott/troy_data_free_run/speedy_era_start_climo_sst_decade_sim12_31_1999_00.nc'
hybrid_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_9_19810101_20100101/mean_output/out.nc'

start_year = 2000
end_year = 2009

startdate = datetime(start_year,1,1,0)
enddate = datetime(end_year,12,31,18)
time_slice = slice(startdate,enddate)

#level = 0.95 #0.2#0.95#0.51
#level_era = 7 #2#7 #4

variable_speedy = 'ps'
var_da = variable_speedy

if variable_speedy == 'q':
    variable_era = 'Specific_Humidity'
if variable_speedy == 't':
    variable_era = 'Temperature'
if variable_speedy == 'v':
    variable_era = 'V-wind'
if variable_speedy == 'u':
    variable_era = 'U-wind'
if variable_speedy == 'ps':
    variable_era = 'logp'

var_era = variable_era
# create empty list to store indiviudal datasets
era5sets = []
print('made it to the for loop...')

# LOAD DATA HERE 
print('LOADING DATA...')
# FOR ERA5
timestep_6hrly = 6
# loop over the range of years and open each ds
for year in range(start_year, end_year + 1):
    nature_file = f'/skydata2/troyarcomano/ERA_5/{year}/era_5_y{year}_regridded_mpi_fixed_var.nc'
    # only load var_era selected and only load level_era selected from above
    if variable_speedy == 'ps': # don't select level if variable is 'ps'
        ds_nature = xr.open_dataset(nature_file)[var_era]
    else:
        ds_nature = xr.open_dataset(nature_file)[var_era].sel(Sigma_Level=level_era)
    # Read in every 6th timestep
    ds_nature = ds_nature.isel(Timestep=slice(None, None, timestep_6hrly))
    era5sets.append(ds_nature)

print('concatinating...')

ds_nature = xr.concat(era5sets, dim = 'Timestep')
ds_nature = ds_nature.sortby('Timestep')
print('Done concat and sortby Timestep...')

# if surface pressure variable selected, then don't select a level. Theres only one for surface pressure..
if var_era == 'logp':
    #convert to hPa for era5 ds
    print('here1')
    #ds_era5 = np.exp(ds_nature.values) * 1000.0
    ds_era5 = ds_nature.values 
    #convert to hPa for letkf analysis                         
    print('here2')
    ds_analysis_mean_speedy = xr.open_dataset(speedy_file)['logp']
    #ds_analysis_mean_speedy = np.exp(ds_analysis_mean_speedy) * 1000.0
    print('here3')
    ds_analysis_mean_hybrid_1_9_1_9 = xr.open_dataset(hybrid_1_9_1_9_file)[var_da].sel(time=time_slice) / 100.0

print('Done.')


# In[52]:


print('era5 shape = ',np.shape(ds_era5))
print('speedy shape = ',np.shape(ds_analysis_mean_speedy))
print('hybrid shape = ',np.shape(ds_analysis_mean_hybrid_1_9_1_9))

#find smallest index value to set that as the "length"
speedy_index = ds_analysis_mean_speedy.shape[0]
nature_index = ds_era5.shape[0]
smallest_index = min(speedy_index,nature_index)

if smallest_index == speedy_index:
    length = speedy_index #- 1
elif smallest_index == nature_index:
    length = nature_index
print('the smallest length is',length)


# In[ ]:





# In[53]:


print(ds_analysis_mean_speedy[0,5,20])
print(ds_era5[0,5,20])


# In[54]:


xgrid = 96
ygrid = 48
# take average by gridpoint
climate_era = np.zeros((ygrid,xgrid))
climate_speedy = np.zeros((ygrid,xgrid))
climate_hybrid_1_9_1_9 = np.zeros((ygrid,xgrid))

# i want average through time of each gridpoint, the climate of each gridpoint

climate_speedy[:,:] = np.average((ds_analysis_mean_speedy),axis=0)
climate_era[:,:] = np.average((ds_era5),axis=0)
climate_hybrid_1_9_1_9[:,:] = np.average((ds_analysis_mean_hybrid_1_9_1_9),axis=0)
    
print('done')

print('speedy 5,20 ',climate_speedy[5,20])
print('era    5,20 ',climate_era[5,20])


# In[55]:


print(np.shape(climate_speedy))
print(np.shape(climate_era))
print(np.shape(climate_hybrid_1_9_1_9))


# In[56]:


# cyclic data load to make plotable
lat = ds_analysis_mean_hybrid_1_9_1_9.lat.values
lon = ds_analysis_mean_hybrid_1_9_1_9.lon.values  
    
cyclic_data_climate_speedy, cyclic_lons = add_cyclic_point(climate_speedy,coord=lon)
cyclic_data_climate_hybrid_1_9_1_9, cyclic_lons = add_cyclic_point(climate_hybrid_1_9_1_9,coord=lon)
cyclic_data_era, cyclic_lons = add_cyclic_point(climate_era,coord=lon)

# define the grid here
lons2d,lats2d = np.meshgrid(cyclic_lons,lat)


# In[72]:


# plotvariables 
imshow_v_min = 0
imshow_v_max = 0.06
ipickcolormap = 'inferno'#'jet' #'viridis'
fs_cbar = 14
fs = 12
units='(hPa)'

fig, axs = plt.subplots(2, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# Plot HYBRID 1
img1 = axs[0].imshow(cyclic_data_climate_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = imshow_v_min, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
axs[0].coastlines()
#     axs[0].gridlines()
cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
cbar1.ax.tick_params(labelsize = fs_cbar)
cbar1.set_label(units, fontsize =fs)
axs[0].set_title('SPEEDY Free Run 2000-2010 Surface Pressure Climate')


#img2 = axs[1].imshow(cyclic_data_climate_hybrid_1_9_1_9,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = imshow_v_min, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#axs[1].coastlines()
#cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
#cbar2.ax.tick_params(labelsize = fs_cbar)
#cbar2.set_label(units, fontsize = fs)
#axs[1].set_title('Hybrid 1.9,1.9 Mean Analysis Surface Pressure Climate', fontsize = fs)

img3 = axs[1].imshow(cyclic_data_era,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = imshow_v_min, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
axs[1].coastlines()
cbar3 = plt.colorbar(img3, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
cbar3.ax.tick_params(labelsize= fs_cbar)
cbar3.set_label(units, fontsize = fs)
axs[1].set_title('ERA5 Reanalysis 2000 - 2010 Surface Pressure Climate', fontsize = fs)

#     std_dev_now = np.sqrt(variance_now)

#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')

# plt.tight_layout()
plt.show()


# In[81]:


## DIFFERENCE PLOT ##

# plotvariables 
imshow_v_min = -60
imshow_v_max = 60
ipickcolormap = 'seismic'#'inferno'#'jet' #'viridis'
fs_cbar = 14
fs = 12
units='(hPa)'

fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# Plot HYBRID 1
img1 = axs[0].imshow(cyclic_data_climate_speedy - cyclic_data_era, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = imshow_v_min, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
axs[0].coastlines()
#     axs[0].gridlines()
cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
cbar1.ax.tick_params(labelsize = fs_cbar)
cbar1.set_label(units, fontsize =fs)
axs[0].set_title('Δ (SPEEDY Free Run - ERA5 Reanalysis)\nSurface Pressure Climate')


img2 = axs[1].imshow(cyclic_data_climate_hybrid_1_9_1_9 - cyclic_data_era,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = imshow_v_min, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
axs[1].coastlines()
cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
cbar2.ax.tick_params(labelsize = fs_cbar)
cbar2.set_label(units, fontsize = fs)
axs[1].set_title('Hybrid (1.9,1.9) - ERA5\nSurface Pressure Climate', fontsize = fs)

img3 = axs[2].imshow(cyclic_data_climate_hybrid_1_9_1_9 - cyclic_data_climate_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = imshow_v_min, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
axs[2].coastlines()
cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
cbar3.ax.tick_params(labelsize= fs_cbar)
cbar3.set_label(units, fontsize = fs)
axs[2].set_title('Hybrid - SPEEDY\nSurface Pressure Climate', fontsize = fs)

#     std_dev_now = np.sqrt(variance_now)

#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')

# plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


## MAIN SCRIPT
## **** CHANGE FILES AND DATES HERE

def rmse_time_series_plot_and_maps(level_in_speedy,variable_speedy):
    # Define: Initial FILES, dates, Variable, and Level desired

    analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_1_9_uniform_20110101_20110501/mean.nc'
    
    
#     analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_cov_1_3_40MEM_individual_ens_member_20110101_20110601/mean_output/out.nc'
#     analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_analysis_covar_1_9_20110101_20110301/mean.nc'
    
    # analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_cov_1_5_40MEM_individual_ens_member_20110101_20110601/mean_output_gues/out.nc'

    ## ERA5 crash tests
    # analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/era5_28_yr_trained_weights_MEM1_eq_MEM2/mean_output/out.nc'
    #analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/era5_crash_ens_members_test/mean_output/out.nc'

    #analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/TROY_ERA5_RESULTS_TEST2/mean_output/out.nc'
#     analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/iterative1_speedystatesfixed_offbyonebugfixed_hybrid_1_5_1_3/mean.nc'
    ### THIS IS THE OLD Hybrid MEAN 1.3,1.3 ## BEFORE BUGG FIXES
    #analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/40member_hybrid_20110101_2011052906/anal/mean/mean.nc'
    '''
    
    
    ANAL
    
    
    '''
    ### ERA5  
    analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/ERA5_1_9/mean_output/out.nc'
    
    ## hybrid 1.9,1.9
    #### NOT HYBRID,, SPEEDY 1.9 correction pressure off 
    hybrid_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_1_9_surface_pressure_bias_turned_off_20110101_20110301/mean_output_gues/out.nc'

    
    ## SPEEDY CORRECTION PRESSURE ON
    hybrid_1_9_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_1_9_20110101_20120101/mean_output_gues/out.nc'
    
    
    spread_file_ERA5 = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/ERA5_1_3_6hr_timestep_1_24_24_20110101_20120101/sprd_output/out.nc'
    
    
    spread_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_cov_1_3_40MEM_individual_ens_member_20110101_20110601/sprd_output/out.nc'
#     spread_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/40member_speedy_covar1_3_20110101_20120901/sprd.nc' 
    
    # SPREAD FILES
    ds_spread_ERA5 = xr.open_dataset(spread_file_ERA5)
    ds_spread_speedy = xr.open_dataset(spread_file_speedy)
    

    start_year = 2011
    end_year = 2011

    startdate = datetime(2011,1,1,0)
    enddate = datetime(2011,3,1,0)
    time_slice = slice(startdate,enddate)

    #level = 0.95 #0.2#0.95#0.51
    #level_era = 7 #2#7 #4
    
    level = level_in_speedy
    if level_in_speedy == .95:
        level_in_era = 7
    if level_in_speedy == .2:
        level_in_era = 2
    level_era = level_in_era

    var_da = variable_speedy
    if variable_speedy == 'q':
        variable_era = 'Specific_Humidity'
    if variable_speedy == 't':
        variable_era = 'Temperature'
    if variable_speedy == 'v':
        variable_era = 'V-wind'
    if variable_speedy == 'u':
        variable_era = 'U-wind'
    if variable_speedy == 'ps':
        variable_era = 'logp'
        
    
#     var_era = variable_era
    print(variable_era)
    var_era = variable_era

    #var_era = 'V-wind'#'Specific_Humidity'#'Temperature' #'V-wind'
    #var_da =  'v'#'q'#'t'#'v'
    print('you selected for variable =',var_era)
    print('at level =',level)
    


    # create empty list to store indiviudal datasets
    era5sets = []
    print('made it to the for loop...')

    # LOAD DATA HERE 
    print('LOADING DATA...')
    # FOR ERA5
    timestep_6hrly = 6
    # loop over the range of years and open each ds
    for year in range(start_year, end_year + 1):
        nature_file = f'/skydata2/troyarcomano/ERA_5/{year}/era_5_y{year}_regridded_mpi_fixed_var.nc'
        # only load var_era selected and only load level_era selected from above
        if variable_speedy == 'ps': # don't select level if variable is 'ps'
            ds_nature = xr.open_dataset(nature_file)[var_era]
        else:
            ds_nature = xr.open_dataset(nature_file)[var_era].sel(Sigma_Level=level_era)
        # Read in every 6th timestep
        ds_nature = ds_nature.isel(Timestep=slice(None, None, timestep_6hrly))
        era5sets.append(ds_nature)

    print('Now its concatinating them all together...')

    ds_nature = xr.concat(era5sets, dim = 'Timestep')
    ds_nature = ds_nature.sortby('Timestep')
    print('Done concat and sortby Timestep...')
#     ds_era5 = ds_nature.values

    # if surface pressure variable selected, then don't select a level. Theres only one for surface pressure..
    if var_era == 'logp':
        #convert to hPa for era5 ds
        ds_era5 = np.exp(ds_nature.values) * 1000.0
        #convert to hPa for letkf analysis                         
        ds_analysis_mean = xr.open_dataset(analysis_file)[var_da].sel(time=time_slice) / 100.0
        ds_analysis_mean_speedy = xr.open_dataset(analysis_file_speedy)[var_da].sel(time=time_slice).values / 100.0
        ds_hybrid_1_9_1_9 = xr.open_dataset(hybrid_1_9_1_9_file)[var_da].sel(time=time_slice).values / 100.0
    else:     
        ds_era5 = ds_nature.values
        ds_analysis_mean = xr.open_dataset(analysis_file)[var_da].sel(lev=level,time=time_slice)
        ds_analysis_mean_speedy = xr.open_dataset(analysis_file_speedy)[var_da].sel(lev=level,time=time_slice)
        ds_hybrid_1_9_1_9 = xr.open_dataset(hybrid_1_9_1_9_file)[var_da].sel(lev=level,time=time_slice).values

    temp_500_analysis = ds_analysis_mean
    # temp_500_analysis = ds_analysis_mean[var_da].sel(lev=level).values
    temp_500_analysis_speedy = ds_analysis_mean_speedy
    # temp_500_analysis_speedy = ds_analysis_mean_speedy[var_da].sel(lev=level,time=time_slice).values
    
    
#     temp_500_spread_era5 = ds_spread_ERA5[var_da].sel(lev=level).values
#     temp_500_spread_speedy = ds_spread_speedy[var_da].sel(lev=level).values
    


    print('era5 shape = ',np.shape(ds_era5))
    print('speedy shape = ',np.shape(temp_500_analysis_speedy))
    print('hybrid shape = ',np.shape(temp_500_analysis))

    #find smallest index value to set that as the "length"
    speedy_index = temp_500_analysis_speedy.shape[0]
    nature_index = ds_era5.shape[0]
    hybrid_index = temp_500_analysis.shape[0]
    smallest_index = min(speedy_index,nature_index,hybrid_index)

    if smallest_index == speedy_index:
        length = speedy_index #- 1
    elif smallest_index == nature_index:
        length = nature_index
    else:
        length = hybrid_index
    print('the smallest length is',length)
    
    #ps_nature = ds_nature['logp'].values
    #ps_nature = 1000.0 * np.exp(ps_nature)
    #ps_analysis = ds_analysis_mean['ps'].values/100.0

    xgrid = 96
    ygrid = 48
    #length =365*4*2 #1952-7 # 240 for 3 months  #1450 ##338 #160#64#177#1400#455

    analysis_rmse = np.zeros((length))
    analysis_rmse_speedy = np.zeros((length))
    global_average_ensemble_spread_era5 = np.zeros((length))
    global_average_ensemble_spread_speedy = np.zeros((length))
    
    hybrid_1_9_1_9_anal_rmse = np.zeros((length))
    #ps_rmse = np.zeros((length))

    analysis_error = np.zeros((length,ygrid,xgrid))
    analysis_error_speedy = np.zeros((length,ygrid,xgrid))
    
    hybrid_1_9_1_9_anal_error = np.zeros((length,ygrid,xgrid))
    
    analysis_bias = np.zeros((length))
    analysis_bias_speedy = np.zeros((length))

    hybrid_1_9_1_9_bias = np.zeros((length))
    
    
    print(np.shape(analysis_error))
    print(np.shape(analysis_error_speedy))
    
#     print('Test unit check:')
#     print(ds_analysis_mean[0])

    print('Now its calculating analysis RMSE...')
    lats = ds_nature.Lat
    
    
    for i in range(length):
        # TIME AVERAGED ERROR
        analysis_rmse[i] = latituded_weighted_rmse(ds_era5[i,:,:],temp_500_analysis[i,:,:],lats)
        analysis_rmse_speedy[i] = latituded_weighted_rmse(ds_era5[i,:,:],temp_500_analysis_speedy[i,:,:],lats)
        #ps_rmse[i] = rms(ps_nature[i*6,:,:],ps_analysis[i,:,:])
        hybrid_1_9_1_9_anal_rmse[i] = latituded_weighted_rmse(ds_era5[i,:,:], ds_hybrid_1_9_1_9[i,:,:],lats)
        
        # ERROR BY GRIDPOINT
        analysis_error[i,:,:] = temp_500_analysis[i,:,:] - ds_era5[i,:,:]
        analysis_error_speedy[i,:,:] = temp_500_analysis_speedy[i,:,:] - ds_era5[i,:,:]
        hybrid_1_9_1_9_anal_error[i,:,:] = ds_hybrid_1_9_1_9[i,:,:] - ds_era5[i,:,:]
        
        # BIAS FOR MAPS 
        analysis_bias[i] = latituded_weighted_bias(ds_era5[i,:,:],temp_500_analysis[i,:,:],lats)
        analysis_bias_speedy[i] = latituded_weighted_bias(ds_era5[i,:,:],temp_500_analysis_speedy[i,:,:],lats)
        hybrid_1_9_1_9_bias[i] = latituded_weighted_bias(ds_era5[i,:,:],ds_hybrid_1_9_1_9[i,:,:],lats)
        
#         global_average_ensemble_spread_era5[i] = np.average(temp_500_spread_era5[i,:,:])
#         global_average_ensemble_spread_speedy[i] = np.average(temp_500_spread_speedy[i,:,:])

    # print('mean analysis_rmse = ',analysis_rmse)

    print('DONE CALCULATING ERROR AT EVERY GRIDPOINT AT EVERY TIMESTEP.')
    
    ############################
    # LOAD TROYS MEAN ANAL

    # Define the base path for the files
    # base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/40member_hybrid_20110101_2011052906/anal'
    # base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/era5_28_yr_trained_weights/'

    ### ERA 5 crash test
#     base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/troy_test_speedy_trained_12monthrun/'
#     troy_anal_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/troy_test_speedy_trained_12monthrun/mean_output/out.nc'
    ### ERA5 1 year run worked path
#     base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/ERA5_weights_sourcecodeupdated_1_20_24/'
    
    #### HYBRID NEW 1_3_1_3
#     hybrid_base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_3_1_3_original_weights_20110101_20120101/'

    # Define the variable name, level, and time slice
    start_year = 2011
    end_year = 2011

    # startdate = datetime(2011,1,1,0)
    # enddate = datetime(2011,6,1,0)
    time_slice = slice(startdate,enddate)

    # level = 0.95 #0.2#0.95#0.51
    # level_era = 7 #2#7 #4

    # var_era = 'Temperature'#'Specific_Humidity'#'Temperature' #'V-wind'
    # var_da =  't'#'q'#'t'#'v'
    print('you selected for variable =',var_era)
    print('at level =',level)
    
    ## ADDING MEAN of Hybrid
#     path_mean_anal_hybrid_retest = hybrid_base_path + 'mean_output/out.nc'
    
#     path_mean_anal_hybrid_retest = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_3_mem1_fixed_20110101_20120501/mean_output/out.nc'
    path_mean_anal_hybrid_retest = hybrid_1_9_1_9_1_9_file


    if variable_speedy == 'ps':
        ds_mean_anal_hybrid_1_9_1_9_1_9 = xr.open_dataset(path_mean_anal_hybrid_retest)[var_da].sel(time=time_slice) / 100.0
    else:    
        ds_mean_anal_hybrid_1_9_1_9_1_9 = xr.open_dataset(path_mean_anal_hybrid_retest)[var_da].sel(lev=level, time=time_slice)

    # SPREAD OF HYBRID
#     spread_file_hybrid = hybrid_base_path + 'sprd_output/out.nc'
#     ds_spread_hybrid = xr.open_dataset(spread_file_hybrid)
#     temp_500_spread_hybrid = ds_spread_hybrid[var_da].sel(lev=level).values
#     global_average_ensemble_spread_hybrid = np.zeros((length))
    
#     # Create an empty list to store the datasets
#     ds_list = []

#     # ens member list
#     ens_member_list = range(1,40+1)
# #     print("opening all files...")
#     # Loop through the member numbers and read in the corresponding files
#     for member_number in ens_member_list:
#         file_path = f'{base_path}/{member_number:03d}_output/out.nc'
#         # file_path = f'{base_path}/{member_number:03d}_output/{member_number:03d}.nc'
#         ds = xr.open_dataset(file_path)[var_da].sel(lev=level, time=time_slice)
#         ds_list.append(ds)
#     print('shape test =', np.shape(ds_list))
    
    
    
#     ds_anal_troy = xr.open_dataset(troy_anal_path)[var_da].sel(lev=level, time=time_slice)
#     print('shape test =', np.shape(ds_anal_troy))
    
    # print('ds_list[0] =',ds_list[0])

    # Assign each element in ds_list to be called ds_member_{i}
    # for i, ds in enumerate(ds_list, start=1):
    #     globals()[f'ds_member_{i}'] = ds

    # print(ds_member_1) 

    lats = ds_nature.Lat

    # analysis_error is for maps, see its a 3d array, 
    # MAKING analysis_rmse now

    print('MAKING zeros arrays..')
#     quantity_of_ens_members = 40
#     analysis_rmse_object = np.zeros((quantity_of_ens_members, length))
    
#     anal_rmse_troy = np.zeros((length))
#     anal_error_troy = np.zeros((length,ygrid,xgrid))
    anal_mean_error_hybrid_1_9_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    anal_mean_rmse_hybrid_1_9_1_9_1_9 = np.zeros((length))
    
    anal_bias_hybrid_1_9_1_9_1_9 = np.zeros((length))
    # check shape, yes they are equal and vibing
    # print('new = ',np.shape(analysis_rmse_object[0]))
    # print('old = ', np.shape(analysis_rmse_1))

#     print('looping through each ERA5 ens_member at every timestep..')
#     counter = 0
#     for member_number in range(0,40):
#         # loop through each timestep
#         counter = counter + 1
#         print('MEM counter = ',counter)
#         for l in range(length):
#     #     for l in range(length):
#             analysis_rmse_object[member_number,l] = latituded_weighted_rmse(ds_era5[l,:,:],ds_list[member_number][l,:,:],lats)
    
        
    ##### and calc for mem 1 with length == 3:
    # analysis_rmse_mem1 = latituded_weighted_rmse(ds_era5[3,:,:],ds_list[1][3,:,:],lats)
    # print(analysis_rmse_mem1)
    #####

    # print('analysis_rmse_object[0] =',analysis_rmse_object[0])
    # print('analysis_rmse_object[39] =',analysis_rmse_object[39])    
    print('Calc hybrid analysis_error and bias')
    
    for i in range(length):
        
        anal_mean_error_hybrid_1_9_1_9_1_9[i,:,:] = ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:] - ds_era5[i,:,:]
        anal_mean_rmse_hybrid_1_9_1_9_1_9[i] = latituded_weighted_rmse(ds_era5[i,:,:], ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:],lats)
        anal_bias_hybrid_1_9_1_9_1_9[i] = latituded_weighted_bias(ds_era5[i,:,:],ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:],lats)
        
    print('Done with speedy trained hybrid analysis.')
    
    # take average of anal_bias_hybrid_1_9_1_9_1_9
    global_mean_anal_bias_hybrid_1_9_1_9_1_9 = np.average(anal_bias_hybrid_1_9_1_9_1_9[24::])
    
    print('global mean bias of speedy trained hybrid analysis = ', global_mean_anal_bias_hybrid_1_9_1_9_1_9)
    
    # take avg of speedy_anal_bias 
    global_mean_anal_bias_speedy = np.average(analysis_bias_speedy[24::])
    print('global mean bias of speedy analysis = ', global_mean_anal_bias_speedy)
    
#     if var_era == 'Specific_Humidity': #convert to g/kg
#         global_mean_anal_bias_hybrid_1_9_1_9_1_9 = global_mean_anal_bias_hybrid_1_9_1_9_1_9*1000
#         global_mean_anal_bias_speedy = global_mean_anal_bias_speedy*1000
        
    
    
    #########################################################
    # MAKE MAP FROM ABOVE 

    ''' 24(below) instead of 28 to cut transient event (ML spin up) out in first few weeks '''
    ####### WHICH AVGERAGE ERROR DO YOU WANT?? I want the analysis error of the mean ERA5
    if var_era == 'Temperature':
        units='(K)'
    if var_era == 'Specific_Humidity':
        units='(g/kg)'
    if var_era == 'V-wind':
        units='(m/s)'
    if var_era == 'U-wind':
        units='(m/s)'
    if var_era == 'logp':
        units='(hPa)' # converted above
    print(units)
    if level == .95:
        title_level = 'Low Level '
    if level == .2:
        title_level = '200 hPa '
    print(title_level)
    if var_era == 'Specific_Humidity':
        title_var_era = 'Specific Humidity'
    if var_era == 'V-wind':
        title_var_era = "Meridional Wind"
    if var_era == 'Temperature':
        title_var_era = 'Temperature'
    if var_era == 'U-wind':
        title_var_era = 'Zonal Wind'
    if var_era == 'logp':
        title_var_era = 'Surface Pressure'
    print(title_var_era)
    
#     obs_network_file = '/skydata2/troyarcomano/letkf-hybrid-speedy/obs/networks/uniform.txt'
#     network_ij = np.loadtxt(obs_network_file,skiprows=2,dtype=int)
#     network_ij = network_ij - 1
    
    
    #### TAKING OUT ABSOLUTE VALUE 
#     averaged_error = np.average(abs(analysis_error[24::,:,:]),axis=0)
    
    averaged_error = np.average((analysis_error[24::,:,:]),axis=0)
    averaged_error_speedy = np.average((analysis_error_speedy[24::,:,:]),axis=0)
    
  
    # SeeeEEeee no abs value taken so its the bias
    
    lat = ds_analysis_mean.lat.values
    lon = ds_analysis_mean.lon.values  
#     lons2d, lats2d = np.meshgrid(lon,lat)

    print('Now plotting and meshing...')
    
    # data for plot Hybrid and Speedy Map
    if var_era == 'Specific_Humidity':
        cyclic_data, cyclic_lons = add_cyclic_point(averaged_error*1000, coord=lon)
        cyclic_data_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy*1000, coord=lon)
    else: 
        cyclic_data, cyclic_lons = add_cyclic_point(averaged_error, coord=lon)
        cyclic_data_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
    
    # data for plot difference 
    diff = averaged_error - averaged_error_speedy
    if var_era == 'Specific_Humidity':
        cyclic_data_diff, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
    else:
        cyclic_data_diff, cyclic_lons = add_cyclic_point(diff, coord=lon)
        
#     print('cyclic_data ', cyclic_data, np.shape(cyclic_data))
#     print('cyclic_lons ', cyclic_lons, np.shape(cyclic_lons))
    
    lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     lons2d, lats2d = np.meshgrid(lon,lat)
    
    ## SET VMIN AND VMAX using old code
    if level == .95 and var_era == 'Temperature':
        adapted_range = np.arange(-5.05,5.05,.05)
        adapted_difference_range = np.arange(-5,5,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)

    if level == .2 and var_era == 'Temperature':
#         adapted_range = np.arange(0,.1,.001)
#         adapted_difference_range = np.arange(-.05,.05,.001)
        adapted_range = np.arange(-5.05,5.05,.05)
        adapted_difference_range = np.arange(-5,5,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
        
    if level == .95 and var_era == 'Specific_Humidity':
        adapted_range = np.arange(-3,3,.05)
        adapted_difference_range = np.arange(-2,2,.001)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
        
    if level == .2 and var_era == 'Specific_Humidity':
        adapted_range = np.arange(-.1,.1,.001)
        adapted_difference_range = np.arange(-.05,.05,.001)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level == .2 and var_era == 'V-wind':
        adapted_range = np.arange(-10.05,10.05,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level == .95 and var_era == 'V-wind':
        adapted_range = np.arange(-5,5,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level == .95 and var_era == 'U-wind':
        adapted_range = np.arange(-10.05,10.05,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level == .2 and var_era == 'U-wind':
        adapted_range = np.arange(-10.05,10.05,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    
    ## ERA5 trained Hybrid Map
    
#     fig, axs = plt.subplots(3, 1, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})
#     # Plot HYBRID
#     img1 = axs[0].imshow(cyclic_data, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = range_min, vmax= range_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[0].coastlines()
# #     axs[0].gridlines()
#     cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar1.set_label('Bias '+ units)
#     axs[0].set_title('LETKF Analysis Bias\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'ERA5 trained Hybrid')

#     # Plot SPEEDY
#     img2 = axs[1].imshow(cyclic_data_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = range_min, vmax=range_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[1].coastlines()
# #     axs[1].gridlines()
#     cbar2 = plt.colorbar(img2, ax=axs[1], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar2.set_label('Bias '+ units)  # Change label for Data 2
#     axs[1].set_title('SPEEDY')

#     # PLOT DIFFERENCE
#     img3 = axs[2].imshow(cyclic_data_diff, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = diff_min, vmax= diff_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[2].coastlines()
# #     axs[2].gridlines()
#     cbar3 = plt.colorbar(img3, ax=axs[2], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar3.set_label(units)
#     axs[2].set_title('Difference (Hybrid - SPEEDY)')
#     for ax_index in range(len(axs)):
#         for i in range(np.shape(network_ij)[0]):
#             axs[ax_index].scatter(lon[network_ij[i][0]], lat[network_ij[i][1]], s=2, color='red', marker='*')
    
#     plt.tight_layout()
#     plt.show()
#     ###############
#     # data for plot Hybrid and Speedy Map
    
    
    averaged_error_speedyhybrid = np.average((anal_mean_error_hybrid_1_9_1_9_1_9[24::,:,:]),axis=0) # this is average at each grid point. all I need to do is subtract 1 number from all these
    
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedyhybrid*1000, coord=lon)
#         cyclic_data_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy*1000, coord=lon)
#     else: 
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedyhybrid, coord=lon)
#         cyclic_data_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
    
#     # data for plot difference 
#     diff = averaged_error_speedyhybrid - averaged_error_speedy
#     if var_era == 'Specific_Humidity':
#         cyclic_data_diff, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
#     else:
#         cyclic_data_diff, cyclic_lons = add_cyclic_point(diff, coord=lon)
        
# #     print('cyclic_data ', cyclic_data, np.shape(cyclic_data))
# #     print('cyclic_lons ', cyclic_lons, np.shape(cyclic_lons))
    
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)
    
    
    
#      ## SPEEDY trained Hybrid Map
    
#     fig, axs = plt.subplots(3, 1, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})
#     # Plot HYBRID
#     img1 = axs[0].imshow(cyclic_data, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = range_min, vmax= range_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[0].coastlines()
# #     axs[0].gridlines()
#     cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar1.set_label('Bias '+ units)
#     axs[0].set_title('LETKF Analysis Bias\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'SPEEDY trained Hybrid')

#     # Plot SPEEDY
#     img2 = axs[1].imshow(cyclic_data_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = range_min, vmax=range_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[1].coastlines()
# #     axs[1].gridlines()
#     cbar2 = plt.colorbar(img2, ax=axs[1], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar2.set_label('Bias '+ units)  # Change label for Data 2
#     axs[1].set_title('SPEEDY')

#     # PLOT DIFFERENCE
#     img3 = axs[2].imshow(cyclic_data_diff, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = diff_min, vmax= diff_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[2].coastlines()
# #     axs[2].gridlines()
#     cbar3 = plt.colorbar(img3, ax=axs[2], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar3.set_label(units)
#     axs[2].set_title('Difference (Hybrid - SPEEDY)')
#     for ax_index in range(len(axs)):
#         for i in range(np.shape(network_ij)[0]):
#             axs[ax_index].scatter(lon[network_ij[i][0]], lat[network_ij[i][1]], s=2, color='red', marker='*')
    
#     plt.tight_layout()
#     plt.show()
    
#     # VARIABILITY PLOT BELOW
#     # take global mean and subtract at every gridpoint
#     # here it is: latitude weighted -> global_mean_anal_bias_hybrid_1_9_1_9_1_9 
    
    
#     #global_mean_anal_bias_hybrid_1_9_1_9_1_9 = np.average(anal_bias_hybrid_1_9_1_9_1_9[24::])
# #     print('global mean bias of speedy trained hybrid analysis = ', global_mean_anal_bias_hybrid_1_9_1_9_1_9)
# #     # take avg of speedy_anal_bias 
# #     global_mean_anal_bias_speedy = np.average(analysis_bias_speedy[24::])
# #     print('global mean bias of speedy analysis = ', global_mean_anal_bias_speedy)
    
#     if var_era == 'Specific_Humidity':
#         cyclic_data_bias_global_mean_departure, cyclic_lons = add_cyclic_point((global_mean_anal_bias_hybrid_1_9_1_9_1_9*1000) - (averaged_error_speedyhybrid*1000), coord=lon)
#         cyclic_data_speedy_bias_global_mean_depature, cyclic_lons = add_cyclic_point((global_mean_anal_bias_speedy*1000)- (averaged_error_speedy*1000), coord=lon)
#     else: 
#         cyclic_data_bias_global_mean_departure, cyclic_lons = add_cyclic_point((global_mean_anal_bias_hybrid_1_9_1_9_1_9) - (averaged_error_speedyhybrid), coord=lon)
#         cyclic_data_speedy_bias_global_mean_depature, cyclic_lons = add_cyclic_point((global_mean_anal_bias_speedy)- (averaged_error_speedy), coord=lon)

#     # data for plot difference 
#     diff_variability = (global_mean_anal_bias_hybrid_1_9_1_9_1_9 - averaged_error_speedyhybrid) - (global_mean_anal_bias_speedy - averaged_error_speedy)
#     if var_era == 'Specific_Humidity':
#         cyclic_data_diff_variability, cyclic_lons = add_cyclic_point(diff_variability*1000, coord=lon)
#     else:
#         cyclic_data_diff_variability, cyclic_lons = add_cyclic_point(diff_variability, coord=lon)
        
#     ## SPEEDY trained Hybrid
#     fig, axs = plt.subplots(3, 1, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})
#     # Plot HYBRID
#     img1 = axs[0].imshow(cyclic_data_bias_global_mean_departure, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = range_min, vmax= range_max, cmap='inferno', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[0].coastlines()
# #     axs[0].gridlines()
#     cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar1.set_label('Departure from Global Mean '+ units)
#     axs[0].set_title('LETKF Global Analysis Mean Bias - Analysis Gridpoint Mean Bias\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'SPEEDY trained Hybrid')

#     # Plot SPEEDY
#     img2 = axs[1].imshow(cyclic_data_speedy_bias_global_mean_depature, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = range_min, vmax=range_max, cmap='inferno', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[1].coastlines()
# #     axs[1].gridlines()
#     cbar2 = plt.colorbar(img2, ax=axs[1], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar2.set_label('Departure from Global Mean '+ units)  # Change label for Data 2
#     axs[1].set_title('SPEEDY')

#     # PLOT DIFFERENCE
#     img3 = axs[2].imshow(cyclic_data_diff_variability, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = diff_min, vmax= diff_max, cmap='seismic', transform=ccrs.PlateCarree(), interpolation='none')
#     axs[2].coastlines()
# #     axs[2].gridlines()
#     cbar3 = plt.colorbar(img3, ax=axs[2], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar3.set_label(units)
#     axs[2].set_title('Difference (Hybrid - SPEEDY)')

    
#     plt.tight_layout()
#     plt.show()
    
    
    
    # Now plot speedy but first calculate
    
    #bias
#     analysis_error_speedy[i,:,:] = temp_500_analysis_speedy[i,:,:] - ds_era5[i,:,:]

    
#     print('bias squared and in g/kg ..\n', squared_bias_g_kg)
    
#     print(np.shape(squared_bias_g_kg))
    #average all the squared differences through time
    if variable_speedy == 'q':
        MSE_speedy = np.average((((analysis_error_speedy[24::,:,:])*1000)**2.0),axis=0)
    else: 
        MSE_speedy = np.average((((analysis_error_speedy[24::,:,:]))**2.0),axis=0)
    
#     print("MSE speedy \n", MSE_speedy)
    # make points for plotting using cyclic point
    cyclic_data_MSE_speedy, cyclic_lons = add_cyclic_point(MSE_speedy, coord=lon)
    
#     print('cyclic_data_MSE \n', cyclic_data_MSE_speedy)
    
    # take an average of the biases
    averaged_error_speedy = np.average((analysis_error_speedy[24::,:,:]),axis=0)
    
#     if variable_speedy == 'q':
#         bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 
#     # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
#     else:
#         bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0 # g/kg
        
        
        
    cyclic_data_bias_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
    # square each bias value
    if variable_speedy == 'q':
        Bias_squared_speedy = (cyclic_data_bias_speedy*1000)**2.0
    else:
        Bias_squared_speedy = cyclic_data_bias_speedy**2.0
    #calculate variance
    Variance_speedy = cyclic_data_MSE_speedy - Bias_squared_speedy
    #calculate std devia
    standard_deviation_speedy = np.sqrt(Variance_speedy)
    
    # Plot of Mean Square Error, Bias**2, and Variance of Speedy trained Hybrid
#     print('variable_speedy = ', variable_speedy)
#     print('type of variable_speedy == ', type(variable_speedy))
#     print('type of level_in_speedy == ', type(level_in_speedy))
    
    if variable_speedy == 'q': # type ===== string
        if level_in_speedy == .95: # type === float 
            imshow_v_max = 8 
        elif level_in_speedy == .2:
            imshow_v_max = .005
    elif variable_speedy == 'v':
        imshow_v_max = 30
    elif variable_speedy == 'u':
        imshow_v_max = 30
    elif variable_speedy == 't':
        imshow_v_max = 16
    elif variable_speedy == 'ps':
        imshow_v_max = 30
        
#     print('imshow_v_max = ', imshow_v_max)
    imshowsquared_units = units + '²'
#     print('imshowsquared_units ', imshowsquared_units)
    #hPa"R\u00b2 score EO: {:0.2f}".format(r2_train_EO)
    
    fs = 14
    fs_cbar =12
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot letkf-speedy
    ipickcolormap = 'viridis'
    img1 = axs[0].imshow(cyclic_data_MSE_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize=fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('LETKF SPEEDY 1.9\n'+ title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error',fontsize = fs)
    else:
        axs[0].set_title('LETKF SPEEDY 1.9\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error',fontsize = fs)

    # Plot SPEEDY

    
    img2 = axs[1].imshow(Bias_squared_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax=imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[1].coastlines()
#     axs[1].gridlines()
    cbar2 = plt.colorbar(img2, ax=axs[1], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar2.ax.tick_params(labelsize=fs_cbar)
    cbar2.set_label(imshowsquared_units,fontsize = fs)  # Change label for Data 2
    axs[1].set_title('Bias Squared',fontsize = fs)

    # PLOT variance = MS - Bias**2.0

    img3 = axs[2].imshow(Variance_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[2].coastlines()
#     axs[2].gridlines()
    cbar3 = plt.colorbar(img3, ax=axs[2], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize = fs)
#     for ax_index in range(len(axs)):
#         for i in range(np.shape(network_ij)[0]):
#             axs[ax_index].scatter(lon[network_ij[i][0]], lat[network_ij[i][1]], s=2, color='red', marker='*')
    
    
    
#     standard_deviation_speedy = np.sqrt(Variance_speedy)
    
#     img4 = axs[3].imshow(standard_deviation_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
#     axs[3].coastlines()
# #     axs[2].gridlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3], orientation='vertical', fraction=0.03, pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_SPEEDY_LETKF_3_12_24.png"
    print(filename)
    # Save the figure with the generated filename
#     plt.savefig('ONR_REPORT_MAPS_3_12_24/' + filename, dpi = 1200)
    plt.show()
    
    
    
    
    
    ############# hybrid iter 2
    bias_hybrid_1_9_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    
    for i in range(length):
        bias_hybrid_1_9_1_9_1_9[i,:,:] = ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:] - ds_era5[i,:,:]
    if variable_speedy == 'q':
        bias_hybrid_1_9_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 
    # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
    else:
        bias_hybrid_1_9_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9_1_9[24::,:,:],axis=0)))**2.0 # g/kg
#     print('bias_hybrid_1_9_1_9_squared ', bias_hybrid_1_9_1_9_squared)
    
    cyclic_data_bias_sqaured_iter1_now, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_1_9_squared, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_iter1 = np.average(((bias_hybrid_1_9_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else:
        MSE_NOW_iter1 = np.average(((bias_hybrid_1_9_1_9_1_9[24::,:,:])**2.0),axis=0)
    
    cyclic_data_mse_iter1_now, cyclic_lons = add_cyclic_point(MSE_NOW_iter1,coord=lon)
    
    #####
    variance_now_iter1 = cyclic_data_mse_iter1_now - cyclic_data_bias_sqaured_iter1_now
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of Speedy trained Hybrid
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID 1
    img1 = axs[0].imshow(cyclic_data_mse_iter1_now, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize =fs)
    if variable_speedy == 'ps':
        axs[0].set_title('LETKF SPEEDY PRESSURE ADJUSTMENT ON\n'+ title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize=fs)
    else:
        axs[0].set_title('LETKF SPEEDY PRESSURE ADJUSTMENT ON\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize=fs)
    
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_iter1_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_iter1,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize = fs)
    
#     std_dev_now = np.sqrt(variance_now)
    
#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_SPEEDY_trained_Hybrid_3_12_24.png"
    print(filename)
    # Save the figure with the generated filename
#     plt.savefig('ONR_REPORT_MAPS_3_12_24/' + filename,dpi=1200) #1200
    plt.show()
    
       ############# ERA5 hybrid
    bias_hybrid_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    
    for i in range(length):
        bias_hybrid_1_9_1_9[i,:,:] = ds_analysis_mean[i,:,:] - ds_era5[i,:,:]
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_squared ', bias_hybrid_1_9_1_9_squared)
    
    cyclic_data_bias_sqaured_now, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_squared, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW = np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW = np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now, cyclic_lons = add_cyclic_point(MSE_NOW,coord=lon)
    
    #####
    variance_now = cyclic_data_mse_now - cyclic_data_bias_sqaured_now
    std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of ERA5 Hybrid
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('LETKF ERA5 trained Hybrid 1.9\n'+ title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('LETKF ERA5 trained Hybrid 1.9\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
#     std_dev_now = np.sqrt(variance_now)
    
#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_ERA5_trained_Hybrid_3_12_24.png"
    print(filename)
    # Save the figure with the generated filename
#     plt.savefig('ONR_REPORT_MAPS_3_12_24/' + filename,dpi=1200)
    plt.show()
    
    
    #### MAKE IMSHOW MAP FOR 1.9,1.9
    bias_hybrid_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    
    for i in range(length):
        bias_hybrid_1_9_1_9[i,:,:] = ds_hybrid_1_9_1_9[i,:,:] - ds_era5[i,:,:]
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_squared ', bias_hybrid_1_9_1_9_squared)
    
    cyclic_data_bias_sqaured_now, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_squared, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW = np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW = np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now, cyclic_lons = add_cyclic_point(MSE_NOW,coord=lon)
    
    #####
    variance_now = cyclic_data_mse_now - cyclic_data_bias_sqaured_now
    std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of Speedy trained Hybrid
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    
    if variable_speedy == 'ps':
        axs[0].set_title('LETKF SPEEDY PRESSURE ADJUSTMENT OFF\n'+ title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('LETKF SPEEDY PRESSURE ADJUSTMENT OFF\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
    
    # *** IM WORKING HERE 6/11/24***
    # fixing 6/10/24 to be 2nd iter - 1st iter
    
    ipickcolormap = 'seismic'
    bias_difference_now = np.zeros((ygrid,xgrid))
    
    bias_difference_now =  bias_hybrid_1_9_1_9_1_9 - bias_hybrid_1_9_1_9  # hybrid 1.9,1.9,1.9 - hybrid 1.9,1.9
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_squared_difference = -bias_hybrid_1_9_1_9_squared + bias_hybrid_1_9_1_9_1_9_squared #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_squared_difference = -bias_hybrid_1_9_1_9_squared +bias_hybrid_1_9_1_9_1_9_squared         #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_squared_difference ', bias_hybrid_1_9_1_9_squared_difference)
    
    cyclic_data_bias_sqaured_now_difference, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_squared_difference, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_difference = -MSE_NOW + MSE_NOW_iter1 # np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
        MSE_NOW_difference = np.sqrt(MSE_NOW_difference)
    else: 
        MSE_NOW_difference = -MSE_NOW + MSE_NOW_iter1 # np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        MSE_NOW_difference = np.sqrt(MSE_NOW_difference)
    cyclic_data_mse_now_difference, cyclic_lons = add_cyclic_point(MSE_NOW_difference,coord=lon)
    
    #####
    # Below is wrong
#     variance_now_difference = -cyclic_data_mse_now_difference + cyclic_data_bias_sqaured_now_difference
    # CORRECTION HERE
    variance2nditeration = variance_now_iter1
    variance1stiteration = variance_now
    
    variance_now_difference = variance2nditeration - variance1stiteration
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of DIFFERENCE OF 2nd iteration Speedy trained Hybrid - 1st iteration
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now_difference, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label('(hPa)', fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('Difference from\nSPEEDY Presadj ON - SPEEDY Presadj OFF\n'+ title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('Difference from\nSPEEDY Presadj ON - SPEEDY Presadj OFF\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Mar 1, 2011\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_now_difference,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_difference,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    

    
#     std_dev_now = np.sqrt(variance_now)
    
#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    
#     print(filename)
    # Save the figure with the generated filename
#     plt.savefig('ONR_REPORT_MAPS_3_12_24/' + filename,dpi=1200)
    plt.show()
    
    
     
   ############ # MAKE TIME SERIES NOW  ################
    
    x = np.arange(0,length)
    base = datetime(2011,1,1,0)

    plt.figure(figsize=(22,8))
    date_list = [base + timedelta(days=x/4) for x in range(length)]
    ### PLOT ENS MEMBERS 
    
    # make colors for 40 member ens
#     from matplotlib.colors import LinearSegmentedColormap
#     start_color = np.array([1.0, 0.8, 0.8])  # Light Red (RGB values)
#     end_color = np.array([0.5, 0.0, 0.0])   # Dark Red (RGB values)
#     # Create a colormap with 40 colors by linearly interpolating between start and end colors
#     cmap = LinearSegmentedColormap.from_list("custom_colormap", np.linspace(start_color, end_color, 40))
#     # Get a list of 40 different colors from the colormap
#     num_colors = 40
#     colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

#     for each_member in range(0,40):
#         plt.plot(date_list,analysis_rmse_object[each_member],linewidth=.45,color=colors[each_member],label=each_member+1)
    # ALSO Average of MEAN LINE
    
    width = 1
    
    if var_era == 'Specific_Humidity':
        # convert (kg/kg) to (g/kg)
        
        analysis_rmse = analysis_rmse*1000
        analysis_rmse_speedy = analysis_rmse_speedy*1000
        anal_mean_rmse_hybrid_1_9_1_9_1_9 = anal_mean_rmse_hybrid_1_9_1_9_1_9*1000
        hybrid_1_9_1_9_anal_rmse = hybrid_1_9_1_9_anal_rmse*1000
        

    
#     plt.plot(date_list,anal_mean_rmse_hybrid_1_9_1_9_1_9,color='red',lw=width,label="SPEEDY trained Hybrid 1.9,1.3")
#     plt.axhline(y=np.average(anal_mean_rmse_hybrid_1_9_1_9_1_9[20::]),color='red',lw=width, linestyle='--',label="Average SPEEDY trained Hybrid 1.9,1.3")
    
    #### TROY
#     plt.plot(date_list,anal_rmse_troy,linewidth=.45,color='red',label='TROY NEW HYBRID 1.5,1.3 Mean')
#     plt.axhline(y=np.average(anal_rmse_troy[20::]), color='red',lw=.6, linestyle='--',label="Average TROY NEW HYBRID 1.5,1.3 Mean")
    
    plt.plot(date_list,analysis_rmse_speedy,label='SPEEDY 1.9',linewidth=width,color='blue')    
    plt.axhline(y=np.average(analysis_rmse_speedy[20::]), color='blue',lw=width, linestyle='--',label="Average SPEEDY 1.9")
    
    plt.plot(date_list,analysis_rmse,label='ERA5 trained Hybrid 1.9',linewidth=width,color='black')
    plt.axhline(y=np.average(analysis_rmse[20::]), color='black',lw=width, linestyle='--',label="Average ERA5 trained Hybrid 1.9")
    
    
    plt.plot(date_list,hybrid_1_9_1_9_anal_rmse,label='SPEEDY trained Hybrid 1.9, 1.9',linewidth=width,color='green')
    plt.axhline(y=np.average(hybrid_1_9_1_9_anal_rmse[20::]),color= 'green', lw=width,linestyle='--',label='SPEEDY trained Hybrid 1.9,1.9')
    
    plt.plot(date_list,anal_mean_rmse_hybrid_1_9_1_9_1_9, label='SPEEDY trained Hybrid 1.9, 1.9, 1.9', linewidth=width, color= 'red')
    plt.axhline(y=np.average(anal_mean_rmse_hybrid_1_9_1_9_1_9[20::]), color ='red', lw=width, linestyle='--', label='SPEEDY trained Hybrid 1.9, 1.9, 1.9')
    
    print(np.average(hybrid_1_9_1_9_anal_rmse[20::]))
#     plt.plot(date_list,analysis_rmse_object[0],linewidth=.45,color='green',label='MEM1 in GREEN')
#     plt.axhline(y=np.average(analysis_rmse[20::]), color='r',lw=width, linestyle='--',label="Average RMSE 1st Iteration Hybrid 1.5,1.3")
#     plt.axhline(y=np.average(analysis_rmse_speedy[20::]), color='b',lw=width, linestyle='--',label="Average RMSE SPEEDY 1.3")

    plt.title('LETKF Analysis RMS Error\n'+ title_level + title_var_era,fontsize = fs)
    if variable_speedy == 'ps':
        plt.title('LETKF Analysis RMS Error\n'+ title_var_era,fontsize = fs)
        
    plt.xlabel('Date',fontsize = fs)
    plt.ylabel(units,fontsize = fs)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=15)
    plt.xlim([startdate,datetime(2011,3,1,0)])
    
    if variable_speedy == 'q':
        if level == .95:
            plt.ylim(.5,2.25)
        elif level == .2:
            plt.ylim(0,.07)
    elif variable_speedy == 'v':
        if level == .95:
            plt.ylim(1.5,3.75)
        elif level == .2:
            plt.ylim(2,6)
    elif variable_speedy == 'u':
        if level == .95:
            plt.ylim(1.5,4)
        elif level == .2:
            plt.ylim(2,6)
    elif variable_speedy == 't':
        if level == .95:
            plt.ylim(.75, 3)
        elif level == .2:
            plt.ylim(.5,3)
    elif variable_speedy == 'ps':
        plt.ylim(17.6,18.2)
        
    # plt.ylim(1.25,2.75)
    
    #####
#     if level == .95 and var_era == 'Temperature':
#         plt.ylim(1,3)
        
#     if level == .2 and var_era == 'Temperature':
#         plt.ylim(1,3)

#     if level == .95 and var_era == 'Specific_Humidity':
#         plt.ylim(.0005,.0021)

#     if level == .2 and var_era == 'Specific_Humidity':
#         plt.ylim(0,6e-5)

#     if level == .2 and var_era == 'V-wind':
#         plt.ylim(2.4,5.5)
        
#     if level == .95 and var_era == 'V-wind':
#         plt.ylim(1.75,3.75)
        
#     if level == .95 and var_era == 'U-wind':
#         plt.ylim(1.4,3.75)
        
#     if level == .2 and var_era == 'U-wind':
#         plt.ylim(2.4,6)
        
    #####
#     handles, labels = plt.get_legend_handles_labels()
#     lgd = plt.legend(handles, labels, ncol=1,loc='center right', bbox_to_anchor=(1.7, 0.5),fontsize = 15)
#     plt.legend(ncol=1,loc='center right', bbox_to_anchor(1.5,0.5),fontsize=15, bbox_inches='tight')
    if variable_speedy == 'q':
        if level == .2:
            plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
        else:
            plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
    elif variable_speedy == 'v':
        if level == .95:
            plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
        elif level == .2:
            plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
    else:
        plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
#     plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=.2)
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    
    # Save the figure with the generated filename
    plt.tight_layout()
    
    filename2 = f"Time_Series_of_3_models_for_level_{level_in_speedy}_variable_{variable_speedy}.png"
#     plt.savefig(filename2,dpi=1200)

#     plt.savefig('ONR_REPORT_TIME_SERIES_3_12_24/'+filename2,dpi=1200) # bbox_inches='tight')
    plt.show()


# In[11]:


# TEST CELL an individual variable/level combination
rmse_time_series_plot_and_maps(.95,'ps')


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


## works!
## need to change file and dates in function above before running this cell

## LOOP THROUGH FUNCTION
# FUNCTION INPUTS
level_list_speedy = [.95,.2]
# variable_list_speedy = ['u','t','v','q','ps'] 
variable_list_speedy = ['t','v','q','ps']

for level in level_list_speedy:
    for variable in variable_list_speedy:
        if level == .2 and variable =='ps':
            break
        rmse_time_series_plot_and_maps(level,variable)
        


# In[6]:



#     diff = averaged_error - averaged_error_speedy
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
#     else:
#         cyclic_data, cyclic_lons = add_cyclic_point(diff, coord=lon)
    
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     ''' Multiply averaged_error by 1000 for spec_humid only'''

#     if var_era == 'Temperature':
#         units='(K)'
#     if var_era == 'Specific_Humidity':
#         units='(g/kg)'
#     if var_era == 'V-wind':
#         units='(m/s)'
#     if var_era == 'U-wind':
#         units='(m/s)'
#     print(units)
#     if level == .95:
#         title_level = 'Low Level '
#     if level == .2:
#         title_level = '200 hPa '

#     if var_era == 'Specific_Humidity':
#         title_var_era = 'Specific Humidity'
#     if var_era == 'V-wind':
#         title_var_era = "Meridional Wind"
#     if var_era == 'Temperature':
#         title_var_era = 'Temperature'
#     if var_era == 'U-wind':
#         title_var_era = 'Zonal Wind'
#     print(title_var_era)

#     if level == .95 and var_era == 'Temperature':
#         adapted_range = np.arange(0,5.05,.05)
#         adapted_difference_range = np.arange(-5,5,.05)

#     if level == .2 and var_era == 'Temperature':
# #         adapted_range = np.arange(0,.1,.001)
# #         adapted_difference_range = np.arange(-.05,.05,.001)
#         adapted_range = np.arange(0,5.05,.05)
#         adapted_difference_range = np.arange(-5,5,.05)

#     if level == .95 and var_era == 'Specific_Humidity':
#         adapted_range = np.arange(0,3,.05)
#         adapted_difference_range = np.arange(-2,2,.001)

#     if level == .2 and var_era == 'Specific_Humidity':
#         adapted_range = np.arange(0,.1,.001)
#         adapted_difference_range = np.arange(-.05,.05,.001)

#     if level == .2 and var_era == 'V-wind':
#         adapted_range = np.arange(0,10.05,.05)
#         adapted_difference_range = np.arange(-2,2.1,.05)
#     if level == .95 and var_era == 'V-wind':
#         adapted_range = np.arange(0,10.05,.05)
#         adapted_difference_range = np.arange(-2,2.1,.05)
#     if level == .95 and var_era == 'U-wind':
#         adapted_range = np.arange(0,10.05,.05)
#         adapted_difference_range = np.arange(-2,2.1,.05)
#     if level == .2 and var_era == 'U-wind':
#         adapted_range = np.arange(0,10.05,.05)
#         adapted_difference_range = np.arange(-2,2.1,.05)
    
#     fs = 16
    
#     ### ADAPT TO 2 small on top 1 big plot below
#     # PLOT 1

#     fig = plt.figure(figsize=(10,8))
    
#     fig.suptitle('LETKF Analysis Error\n' + title_level +  title_var_era + '\nJan 1, 2011 to Jan 1, 2012',fontsize=fs)

#     gs = fig.add_gridspec(2,2, width_ratios=[1, 1], height_ratios=[1.3, 3])
#     ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
#     ''' ax1 ===>  Makes map of hybrid letkf analysis error  '''
# #     ax1 = plt.subplot(311,projection=ccrs.PlateCarree())
#     ax1.coastlines()
    
#     # Mulitply 1000 for spec humid
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error*1000, coord=lon)
#     else: 
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error, coord=lon)
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)
        
#     cf = ax1.contourf(lons2d, lats2d,cyclic_data,levels=adapted_range,extend='both')
# #     plt.colorbar(cf,label=units,fraction=0.046, pad=0.04)

# #     plt.title('LETKF Analysis Error\n' + title_level +  title_var_era)
#     ax1.set_title('ERA5 trained Hybrid',fontsize=fs) #1.3

#     '''ax2 ===>  makes plot of speedy letkf analysis error '''
# #     ax2 = plt.subplot(312,projection=ccrs.PlateCarree())
#     ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
#     ax2.coastlines()
    
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedy*1000, coord=lon)
#     else:
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     cf = ax2.contourf(lons2d, lats2d,cyclic_data,levels=adapted_range,extend='both')
# #     plt.colorbar(cf,label=units,fraction=0.02, pad=0.04)
#     colorbar = plt.colorbar(cf,fraction=0.02, pad=0.04)
#     colorbar
#     colorbar.ax.tick_params(labelsize = 14)
#     colorbar.set_label(units,fontsize=14)
#     ax2.set_title('SPEEDY',fontsize=fs)

#     diff = averaged_error - averaged_error_speedy
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
#     else:
#         cyclic_data, cyclic_lons = add_cyclic_point(diff, coord=lon)
    
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     '''ax3 ==> Makes map of difference of hybrid and speedy '''
#     ax3 = fig.add_subplot(gs[1, :],projection=ccrs.PlateCarree())
# #     ax3 = plt.subplot(313,projection=ccrs.PlateCarree())
#     ax3.coastlines()
#     ax3.set_title('Difference (Hybrid - SPEEDY)',fontsize=fs)
    
#     # high_level_difference_range = np.arange(-.05,.05,.001)
#     # low_level_difference_range = np.arange(-5,5,.05)

#     cf = ax3.contourf(lons2d, lats2d,cyclic_data,levels=adapted_difference_range,extend='both',cmap='seismic')
#     colorbar2 = plt.colorbar(cf,fraction=0.02, pad=0.04)
#     colorbar2
#     colorbar2.ax.tick_params(labelsize= 14)
#     colorbar2.set_label(units,fontsize = 14)
#     plt.tight_layout(pad=1.0)
# #     plt.colorbar(cf,label=units,fraction=0.046, pad=0.04)
    
#     # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
#     filenameera5 = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_ERA5.png"
#     print(filenameera5)
#     # Save the figure with the generated filename
#     plt.savefig('Maps_Speedy_trained_Hybrid_ERA5_trained_Hybrid/' + filenameera5,dpi=1200)
#     plt.show()
    
# #     #####################################
#     print('looping through hybrid 1.3,1.3 ..')
#     for i in range(length):
# #         anal_error_troy[i,:,:] = ds_anal_troy[i,:,:] - ds_era5[i,:,:]
# #         anal_rmse_troy[i] = latituded_weighted_rmse(ds_era5[i,:,:],ds_anal_troy[i,:,:],lats)
#         anal_mean_error_hybrid_1_3_1_3[i,:,:] = ds_mean_anal_hybrid_retest_1_3_1_3[i,:,:] - ds_era5[i,:,:]
#         anal_mean_rmse_hybrid_1_3_1_3[i] = latituded_weighted_rmse(ds_era5[i,:,:],ds_mean_anal_hybrid_retest_1_3_1_3[i,:,:],lats)
# #         global_average_ensemble_spread_hybrid[i] = np.average(temp_500_spread_hybrid[i,:,:])
        
# ### 2nd MAP

#     averaged_error = np.average(abs(anal_mean_error_hybrid_1_3_1_3[24::,:,:]),axis=0)
#     averaged_error_speedy = np.average(abs(analysis_error_speedy[24::,:,:]),axis=0)


#     fig = plt.figure(figsize=(10,8))

#     fig.suptitle('LETKF Analysis Error\n' + title_level +  title_var_era + '\nJan 1, 2011 to Jan 1, 2012',fontsize =fs)

#     gs = fig.add_gridspec(2,2, width_ratios=[1, 1], height_ratios=[1.3, 3])
#     ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
#     ''' ax1 ===>  Makes map of hybrid letkf analysis error  '''
# #     ax1 = plt.subplot(311,projection=ccrs.PlateCarree())
#     ax1.coastlines()
    
#     # Mulitply 1000 for spec humid
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error*1000, coord=lon)
#     else: 
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error, coord=lon)
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)
        
#     cf = ax1.contourf(lons2d, lats2d,cyclic_data,levels=adapted_range,extend='both')
# #     plt.colorbar(cf,label=units,fraction=0.046, pad=0.04)

# #     plt.title('LETKF Analysis Error\n' + title_level +  title_var_era)
#     ax1.set_title('Speedy trained Hybrid',fontsize =fs) #1.3

#     '''ax2 ===>  makes plot of speedy letkf analysis error '''
# #     ax2 = plt.subplot(312,projection=ccrs.PlateCarree())
#     ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
#     ax2.coastlines()
    
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedy*1000, coord=lon)
#     else:
#         cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     cf = ax2.contourf(lons2d, lats2d,cyclic_data,levels=adapted_range,extend='both')
#     colorbar = plt.colorbar(cf,fraction=0.02, pad=0.04)
#     colorbar
#     colorbar.ax.tick_params(labelsize = 14)
#     colorbar.set_label(units,fontsize=14)
# #     plt.colorbar(cf,label=units,fraction=0.046, pad=0.04)
#     ax2.set_title('SPEEDY',fontsize =fs)

#     diff = averaged_error - averaged_error_speedy
#     if var_era == 'Specific_Humidity':
#         cyclic_data, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
#     else:
#         cyclic_data, cyclic_lons = add_cyclic_point(diff, coord=lon)
    
#     lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     '''ax3 ==> Makes map of difference of hybrid and speedy '''
#     ax3 = fig.add_subplot(gs[1, :],projection=ccrs.PlateCarree())
# #     ax3 = plt.subplot(313,projection=ccrs.PlateCarree())
#     ax3.coastlines()
#     ax3.set_title('Difference (Hybrid - SPEEDY)',fontsize =fs)
    
#     # high_level_difference_range = np.arange(-.05,.05,.001)
#     # low_level_difference_range = np.arange(-5,5,.05)

#     cf = ax3.contourf(lons2d, lats2d,cyclic_data,levels=adapted_difference_range,extend='both',cmap='seismic')
#     colorbar2 = plt.colorbar(cf,fraction=0.02, pad=0.04)
#     colorbar2 
#     colorbar2.ax.tick_params(labelsize = 14)
#     colorbar2.set_label(units,fontsize=14)
    
#     plt.tight_layout(pad=1.0)
# #     plt.colorbar(cf,label=units,fraction=0.046, pad=0.04)
# #     plt.show()
    
#     # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
#     filename_hybrid = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_hybrid_speedy.png"
#     print(filename_hybrid)
#     # Save the figure with the generated filename
# #     plt.savefig('Maps_Speedy_trained_Hybrid_ERA5_trained_Hybrid/' + filename_hybrid,dpi=1200)
#     plt.show()



'''
TIME SERIES PLOTTING CODE REMOVED HERE
'''
    
#     # code for plot every ensemble member

#     # MAKE TIME SERIES for each ensemble member
    
#     x = np.arange(0,length)
#     base = datetime(2011,1,1,0)

#     plt.figure(figsize=(12,6))
#     date_list = [base + timedelta(days=x/4) for x in range(length)]
#     ### PLOT ENS MEMBERS 
    
#     # make colors for 40 member ens
# #     from matplotlib.colors import LinearSegmentedColormap
# #     start_color = np.array([1.0, 0.8, 0.8])  # Light Red (RGB values)
# #     end_color = np.array([0.5, 0.0, 0.0])   # Dark Red (RGB values)
# #     # Create a colormap with 40 colors by linearly interpolating between start and end colors
# #     cmap = LinearSegmentedColormap.from_list("custom_colormap", np.linspace(start_color, end_color, 40))
# #     # Get a list of 40 different colors from the colormap
# #     num_colors = 40
# #     colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

# #     for each_member in range(0,40):
# #         plt.plot(date_list,analysis_rmse_object[each_member],linewidth=.45,color=colors[each_member],label=each_member+1)
#     # ALSO Average of MEAN LINE
    
#     width = 1
#     plt.plot(date_list,anal_mean_rmse_hybrid_1_3_1_3,color='red',lw=width,label="Speedy trained Hybrid")
#     plt.axhline(y=np.average(anal_mean_rmse_hybrid_1_3_1_3[20::]),color='red',lw=width, linestyle='--',label="Average Speedy trained Hybrid")
    
#     #### TROY
# #     plt.plot(date_list,anal_rmse_troy,linewidth=.45,color='red',label='TROY NEW HYBRID 1.5,1.3 Mean')
# #     plt.axhline(y=np.average(anal_rmse_troy[20::]), color='red',lw=.6, linestyle='--',label="Average TROY NEW HYBRID 1.5,1.3 Mean")
    
#     plt.plot(date_list,analysis_rmse_speedy,label='Speedy',linewidth=width,color='blue')    
#     plt.axhline(y=np.average(analysis_rmse_speedy[20::]), color='blue',lw=width, linestyle='--',label="Average Speedy")
    
#     plt.plot(date_list,analysis_rmse,label='ERA5 trained Hybrid',linewidth=width,color='black')
#     plt.axhline(y=np.average(analysis_rmse[20::]), color='black',lw=width, linestyle='--',label="Average ERA5 trained Hybrid")

# #     plt.plot(date_list,analysis_rmse_object[0],linewidth=.45,color='green',label='MEM1 in GREEN')
# #     plt.axhline(y=np.average(analysis_rmse[20::]), color='r',lw=width, linestyle='--',label="Average RMSE 1st Iteration Hybrid 1.5,1.3")
# #     plt.axhline(y=np.average(analysis_rmse_speedy[20::]), color='b',lw=width, linestyle='--',label="Average RMSE SPEEDY 1.3")

#     plt.title('LETKF Analysis Error\n'+ title_level + title_var_era,fontsize = fs)
#     plt.xlabel('Date',fontsize = fs)
#     plt.ylabel('RMSE ' + units,fontsize = fs)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.xlim([datetime(2011,1,1,0),datetime(2012,1,1,0)])
#     # plt.ylim(1.25,2.75)
    
#     #####
#     if level == .95 and var_era == 'Temperature':
#         plt.ylim(1,3)
        
#     if level == .2 and var_era == 'Temperature':
#         plt.ylim(1,3)

#     if level == .95 and var_era == 'Specific_Humidity':
#         plt.ylim(.0005,.0021)

#     if level == .2 and var_era == 'Specific_Humidity':
#         plt.ylim(0,6e-5)

#     if level == .2 and var_era == 'V-wind':
#         plt.ylim(2.4,5.5)
        
#     if level == .95 and var_era == 'V-wind':
#         plt.ylim(1.75,3.75)
        
#     if level == .95 and var_era == 'U-wind':
#         plt.ylim(1.4,3.75)
        
#     if level == .2 and var_era == 'U-wind':
#         plt.ylim(2.4,6)
#     #####
# #     handles, labels = plt.get_legend_handles_labels()
# #     lgd = plt.legend(handles, labels, ncol=1,loc='center right', bbox_to_anchor=(1.7, 0.5),fontsize = 15)
# #     plt.legend(ncol=1,loc='center right', bbox_to_anchor(1.5,0.5),fontsize=15, bbox_inches='tight')
#     plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.30, 0.5))
# #     plt.legend()
#     plt.grid(color='grey', linestyle='--', linewidth=.3)
    
#     # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    
#     # Save the figure with the generated filename
# #     plt.tight_layout()
    
#     filename2 = f"Time_Series_of_level_{level_in_speedy}_variable_{variable_speedy}.png"
# #     plt.savefig('Time_series_Speedy_trained_Hybrid_ERA5_trained_Hybrid/'+filename2,dpi=1200)

# #     plt.savefig('Time_series_Speedy_trained_Hybrid_ERA5_trained_Hybrid/'+filename2, bbox_inches='tight')
#     plt.show()
    
    
#     #SPREAD PLOT
#     # code for plotting ens spread

#     # MAKE TIME SERIES



#     x = np.arange(0,length)

#     base = datetime(2011,1,1,0)
#     plt.figure(figsize=(12,6))
#     date_list = [base + timedelta(days=x/4) for x in range(length)]
#     plt.plot(date_list,global_average_ensemble_spread_hybrid,color='r',label='Speedy trained Hybrid 1.3,1.3 retrain') #cov-infl1.3
#     plt.plot(date_list,global_average_ensemble_spread_speedy,color='b',label='SPEEDY-LETKF 1.3')
#     plt.plot(date_list,global_average_ensemble_spread_era5, color='black',label ='ERA5 trained Hybrid 1.3' )
    
#     # plt.axhline(y=np.average(analysis_rmse[20::]), color='r', linestyle='--',label="Average RMSE Hybrid 1.5,1.3")
#     # plt.axhline(y=np.average(analysis_rmse_speedy[20::]), color='b', linestyle='--',label="Average RMSE SPEEDY 1.3")

#     #plt.plot(date_list,global_average_ensemble_spread,label='Ensemble Spread')
#     #plt.title('LETKF Analysis Error\n Low Level Specific Humidity')
#     plt.title('LETKF Background Spread\n'+ title_level + title_var_era)
#     #plt.title('Ensemble Spread\nModel Level 4 Temperature')
#     plt.legend()
#     plt.xlabel('Date')

#     plt.ylabel('Ensemble Spread')
# #     plt.xlim([datetime(2011,1, 1,0), datetime(2011, 6, 5,0)])
# #     plt.ylim(0.16,.2)
#     plt.show()


    

