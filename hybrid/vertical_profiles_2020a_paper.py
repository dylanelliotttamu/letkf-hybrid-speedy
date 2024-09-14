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

def get_files(path,pattern): 
    files = glob.glob(path+pattern)
    return files

def get_speedy_files(truthfiles):
    #speedy_data_dir = '/scratch/user/troyarcomano/FortranReservoir/speedyintegration/parallel_prediction_data/speedy_data/'
    #speedy_data_dir = '/tiered/user/troyarcomano/parallel_prediction_data/speedy_data/'
    speedy_data_dir = '/scratch/user/troyarcomano/temp_storage/'
    speedyfiles = list()
    for i in range(len(truthfiles)):
        truthfile = truthfiles[i]
        endfile = truthfile[-16:-3] 
        #pattern = "speedy_era_start"+"*"+endfile+"*.nc"
        pattern = "speedy_"+"*"+endfile+"*.nc" 
        try:
           speedyfiles.append(get_files(speedy_data_dir,pattern)[0])
        except:
           print('skipping this',pattern)
    
    return speedyfiles 

def get_parallel_files(truthfiles,pattern):
    parallel_data_dir = '/scratch/user/troyarcomano/Predictions/Hybrid/'#parallel_prediction_data/'
    #parallel_data_dir = '/tiered/user/troyarcomano/parallel_prediction_data/'
    #parallel_data_dir = '/scratch/user/troyarcomano/FortranReservoir/hybridspeedy/hybrid_out_data/'
    parallel_files = list()
    for i in range(len(truthfiles)):
        truthfile = truthfiles[i]
        endfile = truthfile[-16:-3]
        full_pattern = pattern+"_"+endfile+"*.nc"
        try:
            parallel_files.append(get_files(parallel_data_dir,full_pattern)[0])
        except:
           print(endfile)
           print('skipping this',parallel_data_dir+full_pattern)
           pass
    return parallel_files

def get_llr_files(truthfiles,pattern):
    llr_data_dir = '/scratch/user/troyarcomano/Predictions/Hybrid/'
    llr_files = list()
    for i in range(len(truthfiles)):
        truthfile = truthfiles[i]
        endfile = truthfile[-16:-3]
        full_pattern = pattern+"_"+endfile+"*.nc"
        try:
            llr_files.append(get_files(llr_data_dir,full_pattern)[0])
        except:
           print(endfile)
           print('skipping this',full_pattern)
           pass
    return llr_files

def get_date_from_file(file_name,keyword):
    print(keyword)
    before, key, after = file_name.partition(keyword)
    date = after[0:-3]
    date = datetime.strptime(date,'%m_%d_%Y_%H')
    return date

def get_hour_index_from_start_of_year(current_time):
    startdate = datetime(current_time.year,1,1,0)

    delta_time = current_time - startdate

    hour_index = int(delta_time.seconds/(60*60)) + delta_time.days*24

    return hour_index

@jit(nopython=True,fastmath=True)
def lin_interp(var,ps,target_pressures,ygrid,xgrid):
    speedy_sigma = np.array([0.025, 0.095, 0.20, 0.34, 0.51, 0.685, 0.835, 0.95])

    #var = np.asarray(var)
 
    #ygrid = np.shape(var)[1]
    #xgrid = np.shape(var)[2]

    var_pressure = np.zeros((len(speedy_sigma),ygrid,xgrid))
   
    for i in range(len(speedy_sigma)):
        var_pressure[i,:,:] = speedy_sigma[i] * ps
 

    regridded_data = np.zeros((len(target_pressures),ygrid,xgrid))

    for i in range(ygrid):
        for j in range(xgrid):
            regridded_data[:,i,j] = np.interp(target_pressures,var_pressure[:,i,j],var[:,i,j])
    return regridded_data
    
def climatology_forecast(climatology_data,variable,backup_var,region,date):
    level = slice(0,8)
    mid_lat_NH_slice = slice(30,70)
    mid_lat_SH_slice = slice(-70,-30)
    trop_lat_slice = slice(-20,20)
    global_slice = slice(-90,90)

    lon_slice = slice(0,360)

    if region == 'NH':
       region_slice = mid_lat_NH_slice
    elif region == 'SH':
       region_slice = mid_lat_SH_slice
    elif region == 'Tropics':
       region_slice = trop_lat_slice
    elif region == 'Global':
       region_slice = global_slice
    else:
       print('Wrong use of rmse only regions are NH,SH,Tropic,Global')

    hour_index = get_hour_index_from_start_of_year(date)

    hour_slice = slice(hour_index-12,hour_index+12)

    try:
       climo_forecast = climatology_data[variable].sel(Timestep=hour_slice,Sigma_Level=level,Lon=lon_slice,Lat=region_slice).sum(dim='Timestep')
    except:
       climo_forecast = climatology_data[backup_var].sel(Timestep=hour_slice,Sigma_Level=level,Lon=lon_slice,Lat=region_slice).sum(dim='Timestep')

    climo_forecast = climo_forecast/24.0

    return climo_forecast

@jit()
def latituded_weighted_error_bias_and_var(true,prediction,lats):
    diff = prediction-true

    weights = np.cos(np.deg2rad(lats))

    weights2d = np.zeros(np.shape(diff))

    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)

    weights3d = np.tile(weights2d,(8,1,1))

    masked = np.ma.MaskedArray(diff, mask=np.isnan(diff))

    weighted_average_mean = np.ma.average(masked,weights=weights3d,axis=(1,2))

    error_minus_bias = diff - weighted_average_mean[:,None,None]

    squared_error_minus_bias = error_minus_bias**2.0

    masked = np.ma.MaskedArray(squared_error_minus_bias, mask=np.isnan(squared_error_minus_bias))

    weighted_average_var = np.ma.average(masked,weights=weights3d,axis=(1,2))

    #rms_error = latituded_weighted_rmse(true,prediction)
    #print(rms_error**2.0,weighted_average_mean**2.0+weighted_average_var)

    return weighted_average_mean, weighted_average_var

@jit()
def latituded_weighted_rmse(true,prediction,lats):
    diff = prediction-true

    weights = np.cos(np.deg2rad(lats))

    weights2d = np.zeros(np.shape(diff))
 
    diff_squared = diff**2.0
    #weights = np.ones((10,96))

    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)
   
    weights3d = np.tile(weights2d,(8,1,1))

    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
    weighted_average = np.ma.average(masked,weights=weights3d,axis=(1,2))
    #weighted_average = np.ma.average(masked,weights=weights2d)

    return np.sqrt(weighted_average)

def latituded_weighted_rmse_3d(true,prediction,lats):
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

def mean_and_bias_error_grid(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_time,height_level):
    level = height_level
    time_slice = plot_time - 1
    time_slice_truth = plot_time
    time_slice_speedy = plot_time*6
    region_slice = slice(-90,90)
    lon_slice = slice(0,360)

    num_of_forecasts = np.shape(truth_files)[0]

    i = 0
    for truth_file,speedy_file,hybrid_file,parallel_file in zip(truth_files,speedy_files,hybrid_files,parallel_files):
        ds_era = xr.open_dataset(truth_file)
        ds_speedy = xr.open_dataset(speedy_file)
        ds_hybrid = xr.open_dataset(hybrid_file)
        ds_parallel = xr.open_dataset(parallel_file)

        try:
           data_era = ds_era[var].sel(Timestep=time_slice_truth, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_speedy = ds_speedy[var].sel(Timestep=time_slice_speedy, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_parallel = ds_parallel[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        except:
           data_era = ds_era[var].sel(Timestep=time_slice_truth, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_speedy = ds_speedy[backup_var].sel(Timestep=time_slice_speedy, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_parallel = ds_parallel[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)

        print('shape of data_era',np.shape(data_era))

        if(i == 0):
           error_speedy = np.zeros((len(truth_files),np.shape(data_era)[0],np.shape(data_era)[1]))
           error_parallel = np.zeros((len(truth_files),np.shape(data_era)[0],np.shape(data_era)[1]))
           error_hybrid = np.zeros((len(truth_files),np.shape(data_era)[0],np.shape(data_era)[1]))

        error_speedy[i,:,:] = data_speedy - data_era
        error_parallel[i,:,:] = data_parallel - data_era
        error_hybrid[i,:,:] = data_hybrid - data_era

        i += 1

    mean_error_speedy = np.average(error_speedy,axis=0)
    mean_error_parallel = np.average(error_parallel,axis=0)
    mean_error_hybrid = np.average(error_hybrid,axis=0)

    print('mean_error_parallel[20,20]',mean_error_parallel[20,20])

    error_minus_bias_speedy = np.zeros((len(truth_files),np.shape(mean_error_speedy)[0],np.shape(mean_error_speedy)[1]))
    error_minus_bias_parallel = np.zeros((len(truth_files),np.shape(mean_error_speedy)[0],np.shape(mean_error_speedy)[1]))
    error_minus_bias_hybrid = np.zeros((len(truth_files),np.shape(mean_error_speedy)[0],np.shape(mean_error_speedy)[1])) 

    for i in range(len(truth_files)):
        error_minus_bias_speedy[i,:,:] = error_speedy[i,:] - mean_error_speedy
        error_minus_bias_parallel[i,:,:] = error_parallel[i,:] - mean_error_parallel
        error_minus_bias_hybrid[i,:,:] = error_hybrid[i,:] - mean_error_hybrid

        error_minus_bias_speedy[i,:,:] = error_minus_bias_speedy[i,:,:]**2.0
        error_minus_bias_parallel[i,:,:] = error_minus_bias_parallel[i,:,:]**2.0
        error_minus_bias_hybrid[i,:,:] = error_minus_bias_hybrid[i,:,:]**2.0

    error_sd_speedy = np.average(error_minus_bias_speedy,axis=0)
    error_sd_parallel = np.average(error_minus_bias_parallel,axis=0)
    error_sd_hybrid = np.average(error_minus_bias_hybrid,axis=0)

    return mean_error_speedy, mean_error_parallel, mean_error_hybrid, np.sqrt(error_sd_speedy), np.sqrt(error_sd_parallel), np.sqrt(error_sd_hybrid)

def rmse_grid(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_time,height_level):
    level = height_level
    time_slice = plot_time - 1
    time_slice_truth = plot_time
    time_slice_speedy = plot_time*6
    region_slice = slice(-90,90)
    lon_slice = slice(0,360)

    num_of_forecasts = np.shape(truth_files)[0]
    
    i = 0
    for truth_file,speedy_file,hybrid_file,parallel_file in zip(truth_files,speedy_files,hybrid_files,parallel_files):
        ds_era = xr.open_dataset(truth_file)
        ds_speedy = xr.open_dataset(speedy_file)
        ds_hybrid = xr.open_dataset(hybrid_file)
        ds_parallel = xr.open_dataset(parallel_file)
           
        try:
           data_era = ds_era[var].sel(Timestep=time_slice_truth, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_speedy = ds_speedy[var].sel(Timestep=time_slice_speedy, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_parallel = ds_parallel[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        except:
           data_era = ds_era[var].sel(Timestep=time_slice_truth, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_speedy = ds_speedy[backup_var].sel(Timestep=time_slice_speedy, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
           data_parallel = ds_parallel[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice) 

        print('shape of data_era',np.shape(data_era))

        if(i == 0):
           squared_error_speedy = np.zeros((np.shape(data_era)[0],np.shape(data_era)[1])) 
           squared_error_parallel = np.zeros((np.shape(data_era)[0],np.shape(data_era)[1]))
           squared_error_hybrid = np.zeros((np.shape(data_era)[0],np.shape(data_era)[1]))
        
        squared_error_speedy = squared_error_speedy + (data_speedy - data_era)**2.0
        squared_error_parallel = squared_error_parallel + (data_parallel - data_era)**2.0
        squared_error_hybrid = squared_error_hybrid + (data_hybrid - data_era)**2.0

        i += 1  

    mean_squared_error_speedy = squared_error_speedy/num_of_forecasts
    mean_squared_error_parallel = squared_error_parallel/num_of_forecasts
    mean_squared_error_hybrid = squared_error_hybrid/num_of_forecasts

    return np.sqrt(mean_squared_error_speedy),np.sqrt(mean_squared_error_parallel),np.sqrt(mean_squared_error_hybrid)

@jit()
def rms(true,prediction):
    return np.sqrt(np.nanmean((prediction-true)**2))

@jit()
def rms_tendency(variable,hours):
    variable_tendency = np.zeros((hours))
    variable = np.exp(variable) * 1000.0

    for i in range(hours):
        variable_tendency[i] = np.sqrt(np.mean((variable[i+1] - variable[i])**2.0)) 

    return variable_tendency

def variable_tendency(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_parallel,ds_llr,hours):
    
    try: 
       data_speedy = ds_speedy[var].values
    except:
       data_speedy = ds_speedy[backup_var].values
     
    try:
       data_hybrid = ds_hybrid[var].values
    except:
       data_hybrid = ds_hybrid[backup_var].values
  
    try:
       data_parallel = ds_parallel[var].values
    except:
       data_parallel = ds_parallel[backup_var].values

    try:
       data_era = ds_era[var].values
    except:
       data_era = ds_era[backup_var].values

    try:
       data_llr = ds_llr[var].values
    except:
       data_llr = ds_llr[backup_var].values

    speedy_tendency = rms_tendency(data_speedy[5:-1:6],hours)
    hybrid_tendency = rms_tendency(data_hybrid,hours)
    parallel_tendency = rms_tendency(data_parallel[1:-1:2],hours)
    era_tendency = rms_tendency(data_era,hours)
    llr_tendency = rms_tendency(data_llr,hours)
    return speedy_tendency, hybrid_tendency, parallel_tendency, era_tendency, llr_tendency

@jit()
def average_variable_tendency(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,llr_files):

    hours = 43
    speedy_hours = 43
    llr_hours = 43
    
    all_parallel_tendency = np.zeros((len(speedy_files),hours))
    all_speedy_tendency = np.zeros((len(speedy_files),speedy_hours))
    all_hybrid_tendency = np.zeros((len(speedy_files),hours))
    all_era_tendency = np.zeros((len(truth_files),hours))
    all_llr_tendency = np.zeros((len(truth_files),llr_hours))
    
    i = 0 
    test = np.zeros((speedy_hours))
    for truth_file,speedy_file,hybrid_file,parallel_file,llr_file in zip(truth_files,speedy_files,hybrid_files,parallel_files,llr_files):
        ds_era = xr.open_dataset(truth_file)
        ds_speedy = xr.open_dataset(speedy_file)
        ds_hybrid = xr.open_dataset(hybrid_file)
        ds_parallel = xr.open_dataset(parallel_file) 
        ds_llr =  xr.open_dataset(llr_file)
        
        all_speedy_tendency[i,:],all_hybrid_tendency[i,:],all_parallel_tendency[i,:],all_era_tendency[i,:],all_llr_tendency[i,:] = variable_tendency(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_parallel,ds_llr,43)

        test += all_speedy_tendency[i,:] 
        print(i,all_speedy_tendency[i])
        i += 1
 
    singlular_average_era = np.mean(all_era_tendency)
    singlular_average_era_array = np.zeros((hours))
    singlular_average_era_array[:] = singlular_average_era
    return np.average(all_speedy_tendency,axis=0),np.average(all_hybrid_tendency,axis=0),np.mean(all_parallel_tendency,axis=0),singlular_average_era_array, np.mean(all_llr_tendency,axis=0)

def rmse_average(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region):
    numlevels = 8
    endtime = plot_times[-1]+1

    all_era_presistence_error = np.zeros((len(truth_files),numlevels,endtime))
    all_speedy_error = np.zeros((len(speedy_files),numlevels,endtime))
    all_hybrid_error = np.zeros((len(hybrid_files),numlevels,endtime))
    all_parallel_error = np.zeros((len(parallel_files),numlevels,endtime))

    i = 0
    for truth_file,speedy_file,hybrid_file,parallel_file in zip(truth_files,speedy_files,hybrid_files,parallel_files):
        print(truth_file)
        print(speedy_file)
        print(hybrid_file)
        print(parallel_file)
        ds_era = xr.open_dataset(truth_file)
        ds_speedy = xr.open_dataset(speedy_file)
        ds_hybrid = xr.open_dataset(hybrid_file) 
        ds_parallel = xr.open_dataset(parallel_file)
        try:
           all_era_presistence_error[i,:,:],all_speedy_error[i,:,:],all_hybrid_error[i,:,:],all_parallel_error[i,:,:] = rmse(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_parallel,plot_times,region)
        except Exception as e:
           print(e)
           print('problem with file',hybrid_file)
           pass
        i += 1 
   
    #all_res_error = ma.masked_where(all_res_error >= np.amax(all_era_presistence_error)*1.5, all_res_error)

    average_era_presistence_error = np.average(all_era_presistence_error,axis=0)
    average_speedy_error = np.average(all_speedy_error,axis=0)
    average_hybrid_error = np.average(all_hybrid_error,axis=0)
    average_parallel_error = np.average(all_parallel_error,axis=0)

    return average_era_presistence_error,average_speedy_error,average_hybrid_error,average_parallel_error

#@jit()
def rmse_average_better(variables,backup_vars,truth_files,forecast_files,plot_times,region,stride=1,climatology=False):
    numlevels = 8
    endtime = plot_times[-1]+1

    all_era_presistence_error = np.zeros((len(variables),len(truth_files),numlevels,endtime))
    all_forecast_error = np.zeros((len(variables),len(forecast_files),numlevels,endtime))
    if climatology:
       all_climo_error = np.zeros((len(variables),len(forecast_files),numlevels,endtime))

    i = 0 
    for truth_file,forecast_file in zip(truth_files,forecast_files):
        j = 0 
        for var,backup_var in zip(variables,backup_vars):
            print(truth_file)
            print(forecast_file)
            ds_era = xr.open_dataset(truth_file)
            ds_forecast = xr.open_dataset(forecast_file)
            print(truth_file[-35:-16]) 
            if climatology:
               #ds_climo = xr.open_dataset('/tiered/user/troyarcomano/parallel_prediction_data/regridded_era_climatology1990_2010.nc')
               ds_climo = xr.open_dataset('/scratch/user/troyarcomano/temp_storage/regridded_era_climatology1990_2010.nc') 
               current_forecast_date = get_date_from_file(truth_file,truth_file[-35:-16])
               climo_forecast = climatology_forecast(ds_climo,var,backup_var,region,current_forecast_date)
            try:
               if climatology:
                  all_era_presistence_error[j,i,:,:], all_forecast_error[j,i,:,:], all_climo_error[j,i,:,:] = rmse_better(var,backup_var,ds_era,ds_forecast,plot_times,region,stride,climo=climo_forecast)
               else:
                  all_era_presistence_error[j,i,:,:], all_forecast_error[j,i,:,:] = rmse_better(var,backup_var,ds_era,ds_forecast,plot_times,region,stride)
            except: #Exception as e:
               #print(e)
               print('problem with file',forecast_file)
               pass
            j += 1
        i += 1 

    #all_res_error = ma.masked_where(all_res_error >= np.amax(all_era_presistence_error)*1.5, all_res_error)

    average_era_presistence_error = np.average(all_era_presistence_error,axis=1)
    average_forecast_error = np.average(all_forecast_error,axis=1)

    print('np.shape(average_forecast_error)',np.shape(average_forecast_error))

    if climatology:
       average_climatology_error = np.average(all_climo_error,axis=1)

    if climatology:  
       return average_era_presistence_error,average_forecast_error,average_climatology_error 
    else:
       return average_era_presistence_error,average_forecast_error

def logp_rmse(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_parallel,times,region):
    
    time_slice = slice(1,times[-1]+2)
    lon_slice = slice(0,360)
    region_slice = slice(-90,90)

    data_era = ds_era[var].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
    data_speedy = ds_speedy[var].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
    data_hybrid = ds_hybrid[var].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
    data_parallel = ds_parallel[var].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)

    era_presistence_error = np.zeros((times[-1]+1))
    speedy_error = np.zeros((times[-1]+1))
    hybrid_error = np.zeros((times[-1]+1))
    parallel_error = np.zeros((times[-1]+1))

     
    #data_era = np.exp(data_era) * 1000.0
    print(data_era.values) 
    print('speedy',data_speedy.values)

    data_speedy = np.exp(data_speedy) * 1000.0
    data_hybrid = np.exp(data_hybrid) * 1000.0
    data_parallel = np.exp(data_parallel) * 1000.0
     
    for i in times:
        era_presistence_error[i] = rms(data_era[i,:,:],data_era[0,:,:])
        speedy_error[i] = rms(data_era[i-1,:,:],data_speedy[i-1,:,:])
        hybrid_error[i] = rms(data_era[i-1,:,:],data_hybrid[i-1,:,:])
        parallel_error[i] = rms(data_era[i-1,:,:],data_parallel[i-1,:,:])

    return era_presistence_error,speedy_error,hybrid_error,parallel_error 

def master_logp_rmse(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region):
    endtime = plot_times[-1]+1

    all_era_presistence_error = np.zeros((len(truth_files),endtime))
    all_speedy_error = np.zeros((len(speedy_files),endtime))
    all_hybrid_error = np.zeros((len(hybrid_files),endtime))
    all_parallel_error = np.zeros((len(parallel_files),endtime))

    i = 0
    for truth_file,speedy_file,hybrid_file,parallel_file in zip(truth_files,speedy_files,hybrid_files,parallel_files):
        print(truth_file)
        ds_era = xr.open_dataset(truth_file)
        ds_speedy = xr.open_dataset(speedy_file)
        ds_hybrid = xr.open_dataset(hybrid_file)
        ds_parallel = xr.open_dataset(parallel_file)
        try:
           all_era_presistence_error[i,:],all_speedy_error[i,:],all_hybrid_error[i,:],all_parallel_error[i,:] = logp_rmse(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_parallel,plot_times,region)
        except Exception as e:
           print(e)
           print('problem with file',hybrid_file)
           pass
        i += 1

    #all_res_error = ma.masked_where(all_res_error >= np.amax(all_era_presistence_error)*1.5, all_res_error)

    average_era_presistence_error = np.average(all_era_presistence_error,axis=0)
    average_speedy_error = np.average(all_speedy_error,axis=0)
    average_hybrid_error = np.average(all_hybrid_error,axis=0)
    average_parallel_error = np.average(all_parallel_error,axis=0)

    return average_era_presistence_error,average_speedy_error,average_hybrid_error,average_parallel_error

def average_mean_and_var_error(var,backup_var,truth_files,speedy_files,hybrid_files,reservoir_files,plot_times,region):
    numlevels = 8
    endtime = 70 
    endtime_speedy = 70

    all_speedy_mean_error = np.zeros((len(speedy_files),numlevels,endtime_speedy))
    all_speedy_var_error = np.zeros((len(speedy_files),numlevels,endtime_speedy))

    all_res_mean_error = np.zeros((len(reservoir_files),numlevels,endtime))
    all_res_var_error = np.zeros((len(reservoir_files),numlevels,endtime))

    all_persistence_mean_error = np.zeros((len(reservoir_files),numlevels,endtime))
    all_persistence_var_error = np.zeros((len(reservoir_files),numlevels,endtime))

    all_hybrid_mean_error = np.zeros((len(reservoir_files),numlevels,endtime))
    all_hybrid_var_error = np.zeros((len(reservoir_files),numlevels,endtime))

    i = 0

    for truth_file,speedy_file,hybrid_file,reservoir_file in zip(truth_files,speedy_files,hybrid_files,reservoir_files):
        ds_era = xr.open_dataset(truth_file)
        ds_speedy = xr.open_dataset(speedy_file)
        ds_reservoir = xr.open_dataset(reservoir_file)
        ds_hybrid = xr.open_dataset(hybrid_file)
        try:
           all_persistence_mean_error[i,:,:],all_persistence_var_error[i,:,:],all_speedy_mean_error[i,:,:],all_speedy_var_error[i,:,:],all_res_mean_error[i,:,:],all_res_var_error[i,:,:],all_hybrid_mean_error[i,:,:],all_hybrid_var_error[i,:,:] = mean_error(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_reservoir,plot_times,region)
        except Exception as e:
           print(e)
           print('problem with file',reservoir_file)
           pass
        i += 1


    average_speedy_mean_error = np.average(all_speedy_mean_error,axis=0)
    average_res_mean_error = np.average(all_res_mean_error,axis=0)
    average_persistence_mean_error = np.average(all_persistence_mean_error,axis=0)
    average_hybrid_mean_error = np.average(all_hybrid_mean_error,axis=0)
     
    var_speedy = np.average(all_speedy_var_error,axis=0)
    var_res = np.average(all_res_var_error,axis=0)
    var_persistence = np.average(all_persistence_var_error,axis=0)
    var_hybrid = np.average(all_hybrid_var_error,axis=0)
    
    return average_persistence_mean_error,average_speedy_mean_error,average_res_mean_error,average_hybrid_mean_error,var_persistence,var_speedy,var_res,var_hybrid

@jit()
def average_mean_and_var_error_better(variables,backup_vars,truth_files,forecast_files,plot_times,region,stride=1):
    numlevels = 8
    endtime = plot_times[-1]+1

    all_forecast_mean_error = np.zeros((len(variables),len(forecast_files),numlevels,endtime))
    all_forecast_var_error = np.zeros((len(variables),len(forecast_files),numlevels,endtime))

    all_persistence_mean_error = np.zeros((len(variables),len(truth_files),numlevels,endtime))
    all_persistence_var_error = np.zeros((len(variables),len(truth_files),numlevels,endtime))

    i = 0

    for truth_file,forecast_file in zip(truth_files,forecast_files):
        print(forecast_file)
        j = 0
        for var,backup_var in zip(variables,backup_vars): 
            ds_era = xr.open_dataset(truth_file)
            ds_forecast = xr.open_dataset(forecast_file)
            all_persistence_mean_error[j,i,:,:],all_persistence_var_error[j,i,:,:],all_forecast_mean_error[j,i,:,:],all_forecast_var_error[j,i,:,:] = mean_error_better(var,backup_var,ds_era,ds_forecast,plot_times,region,stride=stride)
            j += 1
        i += 1


    average_forecast_mean_error = np.average(all_forecast_mean_error,axis=1)
    average_persistence_mean_error = np.average(all_persistence_mean_error,axis=1)

    var_forecast = np.average(all_forecast_var_error,axis=1)
    var_persistence = np.average(all_persistence_var_error,axis=1)

    return average_persistence_mean_error,average_forecast_mean_error,var_persistence,var_forecast


def era_file_truth(date,f_length,stride=1):
    start_year = date.year
    enddate = date + timedelta(hours=f_length) 

    currentdate = date
    while currentdate.year <= enddate.year:
        ds_era = xr.open_dataset(f'/scratch/user/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi_fixed_var_gcc.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d-H%")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_era = ds_era.assign_coords({"Timestep": ("Timestep", ds_era.Timestep.values, attrs)})
        ds_era = xr.decode_cf(ds_era)

        if start_year == currentdate.year:
           ds_merged = ds_era
        else:
           ds_merged = xr.merge([ds_merged,ds_era])

        currentdate = currentdate + timedelta(hours=ds_era.sizes['Timestep'])

    time_slice = slice(date.strftime("%Y-%m-%d-T%H"),enddate.strftime("%Y-%m-%d-T%H"),stride)
    return ds_merged.sel(Timestep=time_slice)

def average_mean_and_var_error_better_era(variables,backup_vars,truth_files,forecast_files,plot_times,region,stride=1):
    numlevels = 8
    endtime = plot_times[-1]+1

    all_forecast_mean_error = np.zeros((len(variables),len(forecast_files),numlevels,endtime))
    all_forecast_var_error = np.zeros((len(variables),len(forecast_files),numlevels,endtime))

    all_persistence_mean_error = np.zeros((len(variables),len(truth_files),numlevels,endtime))
    all_persistence_var_error = np.zeros((len(variables),len(truth_files),numlevels,endtime))

    analysis_mean_error = np.zeros((len(variables),len(truth_files),numlevels))
    analysis_var_error = np.zeros((len(variables),len(truth_files),numlevels))

    i = 0

    for truth_file,forecast_file in zip(truth_files,forecast_files):
        print(forecast_file)
        j = 0
        date = get_date_from_file(truth_file,truth_file[-35:-16])
        print(date)
        for var,backup_var in zip(variables,backup_vars):
            ds_era = era_file_truth(date,int(endtime*6),stride=6)
            ds_analysis = xr.open_dataset(truth_file)
            ds_forecast = xr.open_dataset(forecast_file)
            all_persistence_mean_error[j,i,:,:],all_persistence_var_error[j,i,:,:],all_forecast_mean_error[j,i,:,:],all_forecast_var_error[j,i,:,:] = mean_error_better_era(var,backup_var,ds_era,ds_forecast,ds_analysis,plot_times,region,stride=stride)
            j += 1
        i += 1


    average_forecast_mean_error = np.average(all_forecast_mean_error,axis=1)
    average_persistence_mean_error = np.average(all_persistence_mean_error,axis=1)

    var_forecast = np.average(all_forecast_var_error,axis=1)
    var_persistence = np.average(all_persistence_var_error,axis=1)

    return average_persistence_mean_error,average_forecast_mean_error,var_persistence,var_forecast

#@jit()

def mean_error_better_era(var,backup_var,ds_era,ds_forecast,ds_analysis,times,region,stride=1):
    level = slice(0,8)
    numlevels = 8
    endtime = 1500*stride
    time_slice = slice(0,endtime)
    mid_lat_NH_slice = slice(30,70)
    mid_lat_SH_slice = slice(-70,-30)
    trop_lat_slice = slice(-20,20)
    global_slice = slice(-90,90)
    lon_slice = slice(0,360)

    pressure_levels = np.array([25,95,200,350,500,680,850,950])

    if region == 'NH':
       region_slice = mid_lat_NH_slice
    elif region == 'SH':
       region_slice = mid_lat_SH_slice
    elif region == 'Tropics':
       region_slice = trop_lat_slice
    elif region == 'Global':
       region_slice = global_slice

    else:
       print('Wrong use of rmse only regions are NH,SH,Tropic,Global')

    try: 
        data_era = ds_era[var].sel(Sigma_Level=level,Lon=lon_slice,Lat=region_slice) 
    except:
        data_era = ds_era[backup_var].sel(Sigma_Level=level,Lon=lon_slice,Lat=region_slice) 

    if var == 'Specific-Humidity':
       data_era = data_era * 1000.0

    ps_era = ds_era['logp'].sel(Lon=lon_slice,Lat=region_slice)
    ps_era = np.exp(ps_era) * 1000.0
    
    try:
        data_forecast = ds_forecast[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)

        ps_forecast = ds_forecast['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_forecast = np.exp(ps_forecast) * 1000.0

    except:
        data_forecast = ds_forecast[backup_var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)

        ps_forecast = ds_forecast['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_forecast = np.exp(ps_forecast) * 1000.0

    try:
        data_analysis = ds_analysis[var].sel(Lon=lon_slice,Lat=region_slice)
    except:
        data_analysis = ds_analysis[backup_var].sel(Lon=lon_slice,Lat=region_slice)


    ps_analysis = ds_analysis['logp'].sel(Lon=lon_slice,Lat=region_slice)
    ps_analysis = np.exp(ps_analysis) * 1000.0

    forecast_mean_error = np.zeros((numlevels,times[-1]+1))
    persistence_mean_error = np.zeros((numlevels,times[-1]+1))

    forecast_var_error = np.zeros((numlevels,times[-1]+1))
    persistence_var_error = np.zeros((numlevels,times[-1]+1))

    lats = ps_forecast.Lat.values

    #data_pressure_era_persistence = lin_interp(data_era[0,:,:,:].values,ps_era[0,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])

    for i in times:
        data_pressure_era = lin_interp(data_era[i,:,:,:].values,ps_era[i,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
        data_pressure_era_persistence = lin_interp(data_analysis[i,:,:,:].values,ps_analysis[i,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
        if stride == 1:
           data_pressure_forecast = lin_interp(data_forecast[i,:,:,:].values,ps_forecast[i,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
        else:
           data_pressure_forecast = lin_interp(data_forecast[i*stride-1,:,:,:].values,ps_forecast[i*stride-1,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])

        persistence_mean_error[:,i],persistence_var_error[:,i] = latituded_weighted_error_bias_and_var(data_pressure_era,data_pressure_era_persistence,lats)

        #plt.contourf(data_pressure_era[7,:,:])
        #plt.show()
        #plt.contourf(data_pressure_forecast[7,:,:])
        #plt.show()

        #plt.contourf(data_pressure_forecast[7,:,:] - data_pressure_era[7,:,:],levels=np.arange(-10,11,1),cmap='seismic')
        #plt.colorbar()
        #plt.show()
    
        try:
            forecast_mean_error[:,i],forecast_var_error[:,i] = latituded_weighted_error_bias_and_var(data_pressure_era,data_pressure_forecast,lats)
        except Exception as e:
           print(e)
           forecast_mean_error[:,i] = np.nan
           forecast_var_error[:,i] = np.nan  #latituded_weighted_error_bias_and_var(data_era[i-1,j,:,:],data_forecast[(i-1)**stride,j,:,:])
           persistence_mean_error[:,i] = np.nan
           persistence_var_error[:,i] = np.nan #latituded_weighted_error_bias_and_var(data_era[i,j,:,:],data_era[0,j,:,:])

    return persistence_mean_error,np.sqrt(persistence_var_error),forecast_mean_error,np.sqrt(forecast_var_error)

def mean_error_better(var,backup_var,ds_era,ds_forecast,times,region,stride=1):
    level = slice(0,8)
    numlevels = 8
    endtime = 1500*stride
    time_slice = slice(0,endtime)
    mid_lat_NH_slice = slice(30,70)
    mid_lat_SH_slice = slice(-70,-30)
    trop_lat_slice = slice(-20,20)
    global_slice = slice(-90,90)
    lon_slice = slice(0,360)

    pressure_levels = np.array([25,95,200,350,500,680,850,950])

    if region == 'NH':
       region_slice = mid_lat_NH_slice
    elif region == 'SH':
       region_slice = mid_lat_SH_slice
    elif region == 'Tropics':
       region_slice = trop_lat_slice
    elif region == 'Global':
       region_slice = global_slice

    else:
       print('Wrong use of rmse only regions are NH,SH,Tropic,Global')

    try:
        data_era = ds_era[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_forecast = ds_forecast[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        
        ps_era = ds_era['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_era = np.exp(ps_era) * 1000.0

        ps_forecast = ds_forecast['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_forecast = np.exp(ps_forecast) * 1000.0
      
    except:
        data_era = ds_era[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_forecast = ds_forecast[backup_var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)

        ps_era = ds_era['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_era = np.exp(ps_era) * 1000.0

        ps_forecast = ds_forecast['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_forecast = np.exp(ps_forecast) * 1000.0

    forecast_mean_error = np.zeros((numlevels,times[-1]+1))
    persistence_mean_error = np.zeros((numlevels,times[-1]+1))

    forecast_var_error = np.zeros((numlevels,times[-1]+1))
    persistence_var_error = np.zeros((numlevels,times[-1]+1))

    lats = ps_forecast.Lat.values

    data_pressure_era_persistence = lin_interp(data_era[0,:,:,:].values,ps_era[0,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])

    for i in times:  
        data_pressure_era = lin_interp(data_era[i,:,:,:].values,ps_era[i,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
      
        if stride == 1:
           data_pressure_forecast = lin_interp(data_forecast[i-1,:,:,:].values,ps_forecast[i-1,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
        else:
           data_pressure_forecast = lin_interp(data_forecast[i*stride-1,:,:,:].values,ps_forecast[i*stride-1,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2]) 

        persistence_mean_error[:,i],persistence_var_error[:,i] = latituded_weighted_error_bias_and_var(data_pressure_era,data_pressure_era_persistence,lats)

        try:
            forecast_mean_error[:,i],forecast_var_error[:,i] = latituded_weighted_error_bias_and_var(data_pressure_era,data_pressure_forecast,lats)
        except Exception as e:
           print(e)
           forecast_mean_error[:,i] = np.nan
           forecast_var_error[:,i] = np.nan  #latituded_weighted_error_bias_and_var(data_era[i-1,j,:,:],data_forecast[(i-1)**stride,j,:,:])
           persistence_mean_error[:,i] = np.nan
           persistence_var_error[:,i] = np.nan #latituded_weighted_error_bias_and_var(data_era[i,j,:,:],data_era[0,j,:,:])
      
    return persistence_mean_error,np.sqrt(persistence_var_error),forecast_mean_error,np.sqrt(forecast_var_error)



def mean_error(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_reservoir,times,region):
    level = slice(0,8)
    numlevels = 8
    endtime = 72
    endtime_speedy = 72

    time_slice = slice(1,endtime)
    time_slice_speedy = slice(1,endtime_speedy)

    mid_lat_NH_slice = slice(30,70)
    mid_lat_SH_slice = slice(-70,-30)
    trop_lat_slice = slice(-20,20)
    global_slice = slice(-90,90)

    lon_slice = slice(0,360)

    if region == 'NH':
       region_slice = mid_lat_NH_slice
    elif region == 'SH':
       region_slice = mid_lat_SH_slice
    elif region == 'Tropics':
       region_slice = trop_lat_slice
    elif region == 'Global':
       region_slice = global_slice

    else:
       print('Wrong use of rmse only regions are NH,SH,Tropic,Global')

    try:
        data_era = ds_era[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_speedy = ds_speedy[var].sel(Timestep=time_slice_speedy, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_res = ds_reservoir[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
    except:
        data_era = ds_era[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_speedy = ds_speedy[backup_var].sel(Timestep=time_slice_speedy, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_res = ds_reservoir[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)

    print(np.shape(data_era)[0], endtime,'test')
    if np.shape(data_era)[0] < endtime-3: #or np.shape(data_speedy)[0] < endtime-3 or np.shape(data_res)[0] < endtime-3:
       print('Data length is less than endtime')
       raise

    speedy_mean_error = np.zeros((numlevels,np.shape(data_speedy)[0]-1))
    res_mean_error = np.zeros((numlevels,np.shape(data_era)[0]-1))
    persistence_mean_error = np.zeros((numlevels,np.shape(data_era)[0]-1))
    hybrid_mean_error = np.zeros((numlevels,np.shape(data_era)[0]-1))

    speedy_var_error = np.zeros((numlevels,np.shape(data_speedy)[0]-1))
    res_var_error = np.zeros((numlevels,np.shape(data_era)[0]-1))
    persistence_var_error = np.zeros((numlevels,np.shape(data_era)[0]-1))
    hybrid_var_error = np.zeros((numlevels,np.shape(data_era)[0]-1))    

    for i in times:
        for j in range(numlevels):
            if(i < endtime_speedy ):
              speedy_mean_error[j,i],speedy_var_error[j,i] = latituded_weighted_error_bias_and_var(data_era[i-1,j,:,:],data_speedy[i-1,j,:,:])

            res_mean_error[j,i],res_var_error[j,i] = latituded_weighted_error_bias_and_var(data_era[i-1,j,:,:],data_res[i-1,j,:,:])

            persistence_mean_error[j,i],persistence_var_error[j,i] = latituded_weighted_error_bias_and_var(data_era[i,j,:,:],data_era[0,j,:,:])

            hybrid_mean_error[j,i],hybrid_var_error[j,i] = latituded_weighted_error_bias_and_var(data_era[i,j,:,:],data_hybrid[i-1,j,:,:])

    return persistence_mean_error,np.sqrt(persistence_var_error),speedy_mean_error,np.sqrt(speedy_var_error),res_mean_error,np.sqrt(res_var_error),hybrid_mean_error,np.sqrt(hybrid_var_error)


def rmse(var,backup_var,ds_era,ds_speedy,ds_hybrid,ds_parallel,times,region):
    level = slice(0,8)
    numlevels = 8
    endtime = 104
    time_slice = slice(1,endtime)
    mid_lat_NH_slice = slice(30,70)
    mid_lat_SH_slice = slice(-70,-30)
    trop_lat_slice = slice(-20,20)
    global_slice = slice(-90,90)
    lon_slice = slice(0,360)

    if region == 'NH':
       region_slice = mid_lat_NH_slice
    elif region == 'SH':
       region_slice = mid_lat_SH_slice
    elif region == 'Tropics':
       region_slice = trop_lat_slice
    elif region == 'Global':
       region_slice = global_slice
    else:
       print('Wrong use of rmse only regions are NH,SH,Tropic')

    try:
        data_era = ds_era[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_speedy = ds_speedy[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_parallel = ds_parallel[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
    except:
        data_era = ds_era[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_speedy = ds_speedy[backup_var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_hybrid = ds_hybrid[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        data_parallel = ds_parallel[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)

    print(np.shape(data_era)[0])
    if np.shape(data_era)[0] < times[-1] or np.shape(data_speedy)[0] < times[-1] or np.shape(data_hybrid)[0] < times[-1]:
       print('Data length is less than requested')
       print(f'Date length {times[-1]} was requested but data is only {np.shape(data_speedy)[0]} long') 
       raise 

    era_presistence_error = np.zeros((numlevels,times[-1]+1))
    speedy_error = np.zeros((numlevels,times[-1]+1))
    hybrid_error = np.zeros((numlevels,times[-1]+1))
    parallel_error = np.zeros((numlevels,times[-1]+1))

    for i in times:
        for j in range(numlevels):
            era_presistence_error[j,i] = rms(data_era[i,j,:,:],data_era[0,j,:,:])
            speedy_error[j,i] = rms(data_era[i-1,j,:,:],data_speedy[i-1,j,:,:])
            hybrid_error[j,i] = rms(data_era[i-1,j,:,:],data_hybrid[i-1,j,:,:])
            parallel_error[j,i] = rms(data_era[i-1,j,:,:],data_parallel[i-1,j,:,:])
    
    return era_presistence_error,speedy_error,hybrid_error,parallel_error
    
#@jit()
def rmse_better(var,backup_var,ds_era,ds_forecast,times,region,stride=1,climo=None):
    level = slice(0,8)
    numlevels = 8
    endtime = 1500*stride
    time_slice = slice(0,endtime)
    mid_lat_NH_slice = slice(30,70)
    mid_lat_SH_slice = slice(-70,-30)
    trop_lat_slice = slice(-20,20)
    global_slice = slice(-90,90)
    lon_slice = slice(0,360)
    pressure_levels = np.array([25,95,200,350,500,680,850,950])

    if region == 'NH':
       region_slice = mid_lat_NH_slice
    elif region == 'SH':
       region_slice = mid_lat_SH_slice
    elif region == 'Tropics':
       region_slice = trop_lat_slice
    elif region == 'Global':
       region_slice = global_slice
    else:
       print('Wrong use of rmse only regions are NH,SH,Tropic')

    try:
        data_era = ds_era[var].sel(Timestep=time_slice,Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        ps_era = ds_era['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_era = np.exp(ps_era) * 1000.0
    except:
        data_era = ds_era[backup_var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        ps_era = ds_era['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_era = np.exp(ps_era) * 1000.0

    try:
        data_forecast = ds_forecast[var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        ps_forecast = ds_forecast['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_forecast = np.exp(ps_forecast) * 1000.0
    except:
        data_forecast = ds_forecast[backup_var].sel(Timestep=time_slice, Sigma_Level=level,Lon=lon_slice,Lat=region_slice)
        ps_forecast = ds_forecast['logp'].sel(Timestep=time_slice,Lon=lon_slice,Lat=region_slice)
        ps_forecast = np.exp(ps_forecast) * 1000.0

    #print('np.shape(data_era)[0]',np.shape(data_era)[0])
    #print(data_forecast.Timestep)
    era_presistence_error = np.zeros((numlevels,times[-1]+1))
    forecast_error = np.zeros((numlevels,times[-1]+1))

    if climo is not None:
       climo_error = np.zeros((numlevels,times[-1]+1))
  
    lats = ps_forecast.Lat.values

    data_pressure_era_persistence = lin_interp(data_era[0,:,:,:].values,ps_era[0,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
    for i in times:
        data_pressure_era = lin_interp(data_era[i,:,:,:].values,ps_era[i,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])

        if stride == 1:
           data_pressure_forecast = lin_interp(data_forecast[i-1,:,:,:].values,ps_forecast[i-1,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])
        else:
           data_pressure_forecast = lin_interp(data_forecast[i*stride-1,:,:,:].values,ps_forecast[i*stride-1,:,:].values,pressure_levels,np.shape(data_era[0,:,:,:])[1],np.shape(data_era[0,:,:,:])[2])

        era_presistence_error[:,i] = latituded_weighted_rmse(data_pressure_era[:,:,:],data_pressure_era_persistence[:,:,:],lats)
        if climo is not None:
           climo_error[:,i] = latituded_weighted_rmse(data_era[i,:,:,:],climo[:,:,:],lats)

        try:
           forecast_error[:,i] = latituded_weighted_rmse(data_pressure_era[:,:,:],data_pressure_forecast[:,:,:],lats)
        except: #Exception as e:
           #print(e)
           forecast_error[:,i] = np.nan

    if climo is not None:
       return era_presistence_error,forecast_error,climo_error
    else:
       return era_presistence_error,forecast_error

def panel_3by3_region(speedy_file,truth_file,reservoir_file,region,variables,backup_vars,units,plot_times,trialname,xlimit_array):
    ds_era = xr.open_dataset(truth_file)
    ds_speedy = xr.open_dataset(speedy_file)
    ds_reservoir = xr.open_dataset(reservoir_file) 
 
    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/speedyintegration/plots/' 

    panel_aug1 = 3
    panel_aug2 = 3
    panel_num = 1

    fig, axs = plt.subplots(3,3,figsize=(15,15))

    for var,backup_var,unit,xlim in zip(variables,backup_vars,units,xlimit_array):
        era_presistence_error,speedy_error,res_error = rmse(var,backup_var,ds_era,ds_speedy,ds_reservoir,plot_times,region)
   
        for time in plot_times:
            ax = plt.subplot(f'{panel_aug1}{panel_aug2}{panel_num}')
          
            #Set y-scale to be log since pressure decreases exponentially with height
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Set limits, tickmarks, and ticklabels for y-axis
            ax.set_ylim([1000, 25])
            ax.set_yticks(list(range(1000, 20, -200))+list(range(50,20,-25)))
            ax.set_yticklabels(list(range(1000, 20, -200))+list(range(50,20,-25)),fontsize=8) 
           
            ax.plot(era_presistence_error[:,time],pressure_levels,color='red',ls='-',linewidth=2.0,label='ERA Persistance')
            ax.plot(res_error[:,time],pressure_levels,color='navy',ls='-',linewidth=2.0,label='Parallel Reservoir')
            ax.plot(speedy_error[:,time],pressure_levels,color='g',ls='-',linewidth=2.0,label='SPEEDY') 
              
            ax.set_xlim(xlim)
            ax.set_title(f"{region} {var} {time}H",fontsize=12,fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_ylabel('hPa',fontsize=10)
            ax.set_xlabel(f'RMSE ({unit})',fontsize=8) 

            panel_num += 1

    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45, wspace=0.25)
    #plt.savefig(f'{plotdir}{region}_three_panel_{trialname}.png')
    plt.show()

def panel_3by3_region_average(speedy_files,truth_files,hybrid_files,parallel_files,region,variables,backup_vars,units,plot_times,trialname,xlimit_array):
    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    panel_aug1 = 3
    panel_aug2 = 3
    panel_num = 1

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(3,3,figsize=(15,15))

    for var,backup_var,unit,xlim in zip(variables,backup_vars,units,xlimit_array):
        if len(truth_files) > 1:
           era_presistence_error,speedy_error,hybrid_error,parallel_error = rmse_average(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region)
        else:
           ds_era = xr.open_dataset(truth_files[0])
           ds_speedy = xr.open_dataset(speedy_files[0])
           ds_reservoir = xr.open_dataset(reservoir_files[0])
           era_presistence_error,speedy_error,hybrid_error = rmse(var,backup_var,ds_era,ds_speedy,ds_reservoir,plot_times,region) 

        for time in plot_times:
            ax = plt.subplot(f'{panel_aug1}{panel_aug2}{panel_num}')

            #Set y-scale to be log since pressure decreases exponentially with height
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Set limits, tickmarks, and ticklabels for y-axis
            ax.set_ylim([1000, 25])
            ax.set_yticks(list(range(1000, 20, -200))+list(range(100,20,-50))+list(range(25,20,-25)))
            ax.set_yticklabels(list(range(1000, 20, -200))+list(range(100,20,-50))+list(range(25,20,-25)),fontsize=14)

            ax.plot(era_presistence_error[:,time],pressure_levels,color='#e41a1c',ls='-',linewidth=2.0,label='ERA Persistence')
            ax.plot(hybrid_error[:,time],pressure_levels,color='#377eb8',ls='-',linewidth=2.0,label='Hybrid')
            ax.plot(speedy_error[:,time],pressure_levels,color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY')
            ax.plot(parallel_error[:,time],pressure_levels,color='#984ea3',ls='-',linewidth=2.0,label='ML Model')

            ax.set_xlim(xlim)
            ax.tick_params(axis="x", labelsize=14)

            ax.set_title(f"{region} {var} {time} h",fontsize=16,fontweight="bold")

            ax.legend(fontsize=12)

            ax.set_ylabel('hPa',fontsize=16)
            ax.set_xlabel(f'RMSE ({unit})',fontsize=14)

            panel_num += 1

    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.25)
    plt.tight_layout()
    #plt.show()
    
    plt.savefig(f'{plotdir}{region}_hybrid_three_panel_{trialname}_badremoved.png')
    plt.savefig(f'{plotdir}{region}_hybrid_three_panel_{trialname}_badremoved.pdf')
    plt.close("all")

def timeseries_rmse_plots(speedy_files,truth_files,hybrid_files,parallel_files,region,variable,backup_var,unit,height_level,trialname,xlimit_array):
    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    plot_times = np.arange(0,90)
    forecast_times = np.arange(1,90)

    xtick_times = np.arange(0,90,24)
    era_presistence_error,speedy_error,hybrid_error,parallel_error = rmse_average(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region) 
    
   # era_presistence_error,speedy_error,hybrid_error,parallel_error = master_logp_rmse(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region)
   
     
    plt.plot(plot_times,[0]+list(era_presistence_error[height_level,forecast_times]),color='#e41a1c',ls='-',linewidth=2.0,label='ERA Persistence')
    plt.plot(plot_times,[0]+list(hybrid_error[height_level,forecast_times]),color='#377eb8',ls='-',linewidth=2.0,label='Hybrid')
    plt.plot(plot_times,[0]+list(speedy_error[height_level,forecast_times]),color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY')
    plt.plot(plot_times,[0]+list(parallel_error[height_level,forecast_times]),color='#984ea3',ls='-',linewidth=2.0,label='ML Model') 
     
    #plt.plot(plot_times,[0]+list(era_presistence_error[forecast_times]),color='#e41a1c',ls='-',linewidth=2.0,label='ERA Persistance')
    #plt.plot(plot_times,[0]+list(hybrid_error[forecast_times]),color='#377eb8',ls='-',linewidth=2.0,label='Hybrid')
    #plt.plot(plot_times,[0]+list(speedy_error[forecast_times]),color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY')
    #plt.plot(plot_times,[0]+list(parallel_error[forecast_times]),color='#984ea3',ls='-',linewidth=2.0,label='Parellel Only')

    plt.title(f"{region} RMSE \n{pressure_levels[height_level]} hPa {variable}")

    plt.legend() 

    plt.xticks(xtick_times)
    plt.xlabel('Forecast Hour') 
    plt.xlim(0,plot_times[-1])
    plt.ylim(0,np.max([np.max(era_presistence_error[height_level,forecast_times]),np.max(speedy_error[height_level,forecast_times]),np.max(parallel_error[height_level,forecast_times]),np.max(hybrid_error[height_level,forecast_times])]))

    plt.ylabel(f'RMSE ({unit})')

    plt.savefig(f'{plotdir}{region}_averaged_rmse_{pressure_levels[height_level]}hpa_{variable}_{trialname}_badremoved.png')
    plt.close("all") 

def average_bias_and_sd_error_map(speedy_files,truth_files,hybrid_files,parallel_files,variable,backup_var,unit,height_level,plot_time,trialname,xlimit_array,error_max,diff_max):
    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'
    mean_error_speedy, mean_error_parallel, mean_error_hybrid, error_sd_speedy, error_sd_parallel, error_sd_hybrid = mean_and_bias_error_grid(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,6,height_level)
 
    nc = Dataset(truth_files[0])

    lats = nc['Lat'][:]
    lons = nc['Lon'][:]

    cyclic_data, cyclic_lons = add_cyclic_point(error_sd_speedy - error_sd_parallel, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(4,4,figsize=(10,15))

    ax1 = plt.subplot(4,1,1,projection=ccrs.PlateCarree())
    ax1.coastlines()

    plot = ax1.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')

    cbar = fig.colorbar(plot, ax=ax1)
    cbar.set_label(f'{unit}',fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax1.set_title(f"SD Error Difference (SPEEDY - ML)\n{pressure_levels[height_level]} hPa {variable} 06 h",fontsize=18,fontweight="bold")

    ###########
    cyclic_data, cyclic_lons = add_cyclic_point(error_sd_parallel - error_sd_hybrid, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax2 = plt.subplot(4,1,2,projection=ccrs.PlateCarree())
    ax2.coastlines()

    plot_2 = ax2.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    cbar2 = fig.colorbar(plot_2, ax=ax2)
    cbar2.set_label(f'{unit}',fontsize=16)
    cbar2.ax.tick_params(labelsize=16)

    ax2.set_title(f"SD Error Difference (ML - Hybrid)\n{pressure_levels[height_level]} hPa {variable} 06 h",fontsize=18,fontweight="bold")

    ###########
    cyclic_data, cyclic_lons = add_cyclic_point(abs(mean_error_speedy) - abs(mean_error_parallel), coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax3 = plt.subplot(4,1,3,projection=ccrs.PlateCarree())
    ax3.coastlines()

    plot_3 = ax3.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    cbar3 = fig.colorbar(plot_3, ax=ax3)
    cbar3.set_label(f'{unit}',fontsize=16)
    cbar3.ax.tick_params(labelsize=16)
    ax3.set_title(f"Absolute Mean Error Difference (SPEEDY - ML)\n{pressure_levels[height_level]} hPa {variable} 06 h",fontsize=18,fontweight="bold")

    ###########
    cyclic_data, cyclic_lons = add_cyclic_point(abs(mean_error_parallel) - abs(mean_error_hybrid), coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax4 = plt.subplot(4,1,4,projection=ccrs.PlateCarree())
    ax4.coastlines()

    plot_4 = ax4.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    cbar4 = fig.colorbar(plot_4, ax=ax4)
    cbar4.set_label(f'{unit}',fontsize=16)
    cbar4.ax.tick_params(labelsize=16)
    ax4.set_title(f"Absolute Mean Error Difference (ML - Hybrid)\n{pressure_levels[height_level]} hPa {variable} 06 h",fontsize=18,fontweight="bold")

    plt.subplots_adjust(top=0.946, bottom=0.032, left=0.015, right=0.985, hspace=0.253, wspace=0.33)

    plt.savefig(f'{plotdir}averaged_2d_mean_and_sd_error_{pressure_levels[height_level]}hpa_{variable}_{trialname}_h{plot_time}badremoved.png')
    plt.close("all")

def average_error_map(speedy_files,truth_files,hybrid_files,parallel_files,variable,backup_var,unit,height_level,plot_time,trialname,xlimit_array,error_max,diff_max):
    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    rmse_grid_speedy_6,rmse_grid_parallel_6,rmse_grid_hybrid_6 = rmse_grid(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,6,height_level)
    rmse_grid_speedy_12,rmse_grid_parallel_12,rmse_grid_hybrid_12 = rmse_grid(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,12,height_level)
    rmse_grid_speedy_24,rmse_grid_parallel_24,rmse_grid_hybrid_24 = rmse_grid(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,24,height_level)
    rmse_grid_speedy_36,rmse_grid_parallel_36,rmse_grid_hybrid_36 = rmse_grid(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,36,height_level)

    nc = Dataset(truth_files[0])

    lats = nc['Lat'][:]
    lons = nc['Lon'][:]

    cyclic_data, cyclic_lons = add_cyclic_point(rmse_grid_speedy_12.values - rmse_grid_hybrid_12.values, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(3,3,figsize=(10,15))

    ax1 = plt.subplot(3,1,1,projection=ccrs.PlateCarree())
    ax1.coastlines()

    plot = ax1.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')

    cbar = fig.colorbar(plot, ax=ax1)
    cbar.set_label(f'{unit}',fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax1.set_title(f"RMS Error Difference (SPEEDY - Hybrid)\n{pressure_levels[height_level]} hPa {variable} 12 h",fontsize=18,fontweight="bold")

    ###########
    cyclic_data, cyclic_lons = add_cyclic_point(rmse_grid_speedy_24.values - rmse_grid_hybrid_24.values, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax2 = plt.subplot(3,1,2,projection=ccrs.PlateCarree())
    ax2.coastlines()

    plot_2 = ax2.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    cbar2 = fig.colorbar(plot_2, ax=ax2)
    cbar2.set_label(f'{unit}',fontsize=16)
    cbar2.ax.tick_params(labelsize=16)

    ax2.set_title(f"RMS Error Difference (SPEEDY - Hybrid)\n{pressure_levels[height_level]} hPa {variable} 24 h",fontsize=18,fontweight="bold")
   
    ###########
    cyclic_data, cyclic_lons = add_cyclic_point(rmse_grid_speedy_36.values - rmse_grid_hybrid_36.values, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax3 = plt.subplot(3,1,3,projection=ccrs.PlateCarree())
    ax3.coastlines()

    plot_3 = ax3.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    cbar3 = fig.colorbar(plot_3, ax=ax3)
    cbar3.set_label(f'{unit}',fontsize=16)
    cbar3.ax.tick_params(labelsize=16)
    ax3.set_title(f"RMS Error Difference (SPEEDY - Hybrid)\n{pressure_levels[height_level]} hPa {variable} 36 h",fontsize=18,fontweight="bold")

    '''
    ###########
    cyclic_data, cyclic_lons = add_cyclic_point(rmse_grid_speedy_48.values - rmse_grid_parallel_48.values, coord=lons)
    lons2d,lats2d = np.meshgrid(cyclic_lons,lats)

    ax3 = plt.subplot(4,1,4,projection=ccrs.PlateCarree())
    ax3.coastlines()

    plot_3 = ax3.pcolormesh(lons2d,lats2d,cyclic_data,transform=ccrs.PlateCarree(),vmin=-1*diff_max,vmax=diff_max,cmap='seismic')
    cbar3 = fig.colorbar(plot_3, ax=ax3)
    cbar3.set_label(f'{unit}',fontsize=16)
    cbar3.ax.tick_params(labelsize=16)
    ax3.set_title(f"RMS Error Difference (SPEEDY - ML Model)\n{pressure_levels[height_level]} hPa {variable} 48 h",fontsize=18,fontweight="bold")
    '''

    plt.subplots_adjust(top=0.946, bottom=0.032, left=0.015, right=0.985, hspace=0.253, wspace=0.33)

    plt.savefig(f'{plotdir}averaged_2d_rmse_{pressure_levels[height_level]}hpa_{variable}_{trialname}_h{plot_time}badremoved.png')
    plt.close("all")

def plot_surface_tendecy(speedy_files,truth_files,hybrid_files,parallel_files,llr_files,variable,backup_var,unit):
 
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'
 
    var = variable
    back_up = backup_var
   
    mean_speedy_tendency, mean_hybrid_tendency, mean_parallel_tendency, mean_era_tendency, mean_llr_tendency= average_variable_tendency(var,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,llr_files)
  
    plt.rc('font', family='serif')

    plt.figure(figsize=(10,6))
 
    plot_times = np.arange(1,41,1)
    index_times = np.arange(0,40,1)
    xtick_times = np.arange(0,11,1)

    plot_times = plot_times/4

    plt.plot(plot_times,mean_hybrid_tendency[index_times]/6.0,color='#377eb8',ls='-',linewidth=2.0,label='Hybrid')
    plt.plot(plot_times,mean_speedy_tendency[index_times]/6.0,color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY')
    plt.plot(plot_times,mean_parallel_tendency[index_times]/6.0,color='#ff7f00',ls='-',linewidth=2.0,label='ML')
    plt.plot(plot_times,mean_era_tendency[index_times]/6.0,color='#e41a1c',ls='-',linewidth=2.0,label='ERA5')
    #plt.plot(plot_times,mean_llr_tendency[index_times]/6.0,color='#984ea3',ls='-',linewidth=2.0,label='SPEEDY-LLR')

    plt.title(f'Global Mean Surface Pressure Tendency',fontsize=18,fontweight="bold")
    plt.xlabel('Forecast Day',fontsize=18)
    plt.ylabel(f'Surface Pressure Tendency ({unit}/hr)',fontsize=18)
    plt.xticks(xtick_times,fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0.2,np.amax(mean_speedy_tendency)/6.0)
    plt.xlim(0.25,10)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{plotdir}mean_surface_pressure_tendency_6hr_newest.pdf')
    plt.close("all")
    #plt.show() 

    #plt.title('Difference Between ERA and Hybrid ICs',fontsize=18,fontweight="bold")
    #plt.xlabel('Forecast Hour',fontsize=14)
    #plt.ylabel('Difference In Surface Pressure Tendency (hPa/hour)',fontsize=14)
    #plt.plot(plot_times,mean_parallel_tendency-mean_speedy_tendency,linewidth=2.0)
    #plt.show()

def mean_and_var_error_timeseries(speedy_files,truth_files,hybrid_files,parallel_files,region,variable,backup_var,unit,height_level,trialname,xlimit_array):

    plt.rc('font', family='serif')

    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    plot_times = np.arange(1,70)
    xtick_times = np.arange(1,72,6)
    forecast_times = np.arange(1,70)
    forecast_times_speedy = np.arange(1,70)
    plot_times_speedy = np.arange(1,70)

    mean_persistence_error,mean_speedy_error,mean_res_error,mean_hybrid_error,var_persistence_error,var_speedy_error,var_res_error,var_hybrid_error = average_mean_and_var_error(variable,backup_var,truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region)

    for i in height_level:
        plt.figure(figsize=(10,6))

        plt.plot(plot_times_speedy,mean_speedy_error[i,forecast_times_speedy],color='#e41a1c',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
        plt.plot(plot_times_speedy,var_speedy_error[i,forecast_times_speedy],color='#e41a1c',ls='--',linewidth=2.0,label='SPEEDY SD Error')
        plt.plot(plot_times,mean_hybrid_error[i,forecast_times],color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
        plt.plot(plot_times,var_hybrid_error[i,forecast_times],color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
        plt.plot(plot_times,mean_persistence_error[i,forecast_times],color='k',ls='-',linewidth=2.0,label='Persistence Mean Error')
        plt.plot(plot_times,var_persistence_error[i,forecast_times],color='k',ls='--',linewidth=2.0,label='Persistence SD Error')
        plt.plot(plot_times,mean_res_error[i,forecast_times],color='g',ls='-',linewidth=2.0,label='ML Mean Error')
        plt.plot(plot_times,var_res_error[i,forecast_times],color='g',ls='--',linewidth=2.0,label='ML SD Error')
 
        
        plt.title(f"{region} Mean Error and SD \n {pressure_levels[i]} hPa {variable}")

        plt.legend()

        plt.xticks(xtick_times)
        plt.xlabel('Forecast Hour')
        plt.xlim(0,plot_times[-1])

        plt.ylabel(f'Error ({unit})')

        #trialname = '9000node_32_32_32_noise__degree6_cylcing12hr_sigma0.5_radius0.3_0.7_long_prediction_beta_0.000001trial_truncated'
        plt.savefig(f'{plotdir}{region}_mean_and_var_error_{pressure_levels[i]}hpa_{variable}_{trialname}_badremoved_lat_weighted.png')
        plt.close("all")
        #plt.show()

def mean_and_var_error_3panel(speedy_files,truth_files,hybrid_files,parallel_files,region,trialname,xlimit_array):

    plt.rc('font', family='serif')

    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    plot_times = np.arange(1,70)
    xtick_times = np.arange(0,73,3)
    forecast_times = np.arange(1,70)
    forecast_times_speedy = np.arange(1,70)
    plot_times_speedy = np.arange(1,70)

    mean_persistence_error_temp,mean_speedy_error_temp,mean_res_error_temp,mean_hybrid_error_temp,var_persistence_error_temp,var_speedy_error_temp,var_res_error_temp,var_hybrid_error_temp = average_mean_and_var_error('Temperature','Temperature',truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region)

    mean_persistence_error_sp,mean_speedy_error_sp,mean_res_error_sp,mean_hybrid_error_sp,var_persistence_error_sp,var_speedy_error_sp,var_res_error_sp,var_hybrid_error_sp = average_mean_and_var_error('Specific-Humidity','Specific_Humidity',truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region)

    mean_persistence_error_vd,mean_speedy_error_vd,mean_res_error_vd,mean_hybrid_error_vd,var_persistence_error_vd,var_speedy_error_vd,var_res_error_vd,var_hybrid_error_vd = average_mean_and_var_error('V-wind','V-wind',truth_files,speedy_files,hybrid_files,parallel_files,plot_times,region)

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(3,3,figsize=(10,15))

    ax1 = plt.subplot(3,1,1)

    ax1.plot(plot_times_speedy,mean_speedy_error_temp[-1,forecast_times_speedy],color='#e41a1c',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
    ax1.plot(plot_times_speedy,var_speedy_error_temp[-1,forecast_times_speedy],color='#e41a1c',ls='--',linewidth=2.0,label='SPEEDY SD Error')
    ax1.plot(plot_times,mean_hybrid_error_temp[-1,forecast_times],color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
    ax1.plot(plot_times,var_hybrid_error_temp[-1,forecast_times],color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
    ax1.plot(plot_times,mean_res_error_temp[-1,forecast_times],color='g',ls='-',linewidth=2.0,label='ML Mean Error')
    ax1.plot(plot_times,var_res_error_temp[-1,forecast_times],color='g',ls='--',linewidth=2.0,label='ML SD Error')


    ax1.set_title(f"{region} Mean Error and SD \n 950  hPa Temperature")

    ax1.legend(fontsize=14)

    ax1.set_xticks(xtick_times)
    ax1.set_xlabel('Forecast Hour',fontsize=14)
    ax1.set_xlim(1,24)#plot_times[-1])

    ax1.set_ylabel(f'Error (Kelvin)',fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14) 

    ax2 = plt.subplot(3,1,2)

    ax2.plot(plot_times_speedy,mean_speedy_error_sp[-1,forecast_times_speedy],color='#e41a1c',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
    ax2.plot(plot_times_speedy,var_speedy_error_sp[-1,forecast_times_speedy],color='#e41a1c',ls='--',linewidth=2.0,label='SPEEDY SD Error')
    ax2.plot(plot_times,mean_hybrid_error_sp[-1,forecast_times],color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
    ax2.plot(plot_times,var_hybrid_error_sp[-1,forecast_times],color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
    ax2.plot(plot_times,mean_res_error_sp[-1,forecast_times],color='g',ls='-',linewidth=2.0,label='ML Mean Error')
    ax2.plot(plot_times,var_res_error_sp[-1,forecast_times],color='g',ls='--',linewidth=2.0,label='ML SD Error')


    ax2.set_title(f"{region} Mean Error and SD \n 950  hPa Specific Humidity")

    ax2.legend(fontsize=14)

    ax2.set_xticks(xtick_times)
    ax2.set_xlabel('Forecast Hour',fontsize=14)
    ax2.set_xlim(1,24)#plot_times[-1])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel(f'Error (g/kg)',fontsize=14)

    ax3 = plt.subplot(3,1,3)

    ax3.plot(plot_times_speedy,mean_speedy_error_vd[2,forecast_times_speedy],color='#e41a1c',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
    ax3.plot(plot_times_speedy,var_speedy_error_vd[2,forecast_times_speedy],color='#e41a1c',ls='--',linewidth=2.0,label='SPEEDY SD Error')
    ax3.plot(plot_times,mean_hybrid_error_vd[2,forecast_times],color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
    ax3.plot(plot_times,var_hybrid_error_vd[2,forecast_times],color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
    ax3.plot(plot_times,mean_res_error_vd[2,forecast_times],color='g',ls='-',linewidth=2.0,label='ML Mean Error')
    ax3.plot(plot_times,var_res_error_vd[2,forecast_times],color='g',ls='--',linewidth=2.0,label='ML SD Error')


    ax3.set_title(f"{region} Mean Error and SD \n 200  hPa V-wind")

    ax3.legend(fontsize=14)

    ax3.set_xticks(xtick_times)
    ax3.set_xlabel('Forecast Hour',fontsize=14)
    ax3.set_xlim(1,24)#plot_times[-1])

    ax3.set_ylabel(f'Error (m/s)',fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)

    plt.subplots_adjust(top=0.946, bottom=0.032, left=0.075, right=0.985, hspace=0.253, wspace=0.33)

    plt.savefig(f'{plotdir}{region}_mean_and_var_error_3panel_{trialname}_badremoved_lat_weighted.png')
    plt.close("all")


def timeseries_rmse_plots_6hr(speedy_files,truth_files,hybrid_files,region,variables,backup_vars,units,height_levels,trialname,xlimit_array,parallel_files=None,llr_files=None):
    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    plot_times = np.arange(0,57)
    plot_times_hybrid = np.arange(0,57)
    plot_times_speedy = np.arange(0,57)
    plot_times_parallel = np.arange(0,57)#260)
    plot_times_llr = np.arange(0,45)

    forecast_times = np.arange(1,57)
    forecast_times_hybrid = np.arange(1,57)
    forecast_times_speedy = np.arange(1,57)
    forecast_times_parallel =  np.arange(1,57)#260)
    forecast_times_peristence = np.arange(0,57)
    forecast_times_llr =  np.arange(1,45)
    
    xtick_times = np.arange(0,15,1)

    era_presistence_error_better, hybrid_error = rmse_average_better(variables,backup_vars,truth_files,hybrid_files,plot_times,region)
    era_presistence_error, speedy_error, climo_error = rmse_average_better(variables,backup_vars,truth_files,speedy_files, plot_times_speedy,region,stride=6,climatology=True)

    if parallel_files is not None: 
       era_presistence_error_parallel, parallel_error = rmse_average_better(variables,backup_vars,truth_files,parallel_files,plot_times_parallel,region,stride=2)

    if llr_files is not None:
       era_presistence_error_parallel, llr_error = rmse_average_better(variables,backup_vars,truth_files,llr_files,plot_times_llr,region)

    print(np.shape(speedy_error))
    print(np.shape(hybrid_error))
    print(np.shape(era_presistence_error))

    plot_times = plot_times/4
    plot_times_hybrid = plot_times_hybrid/4
    plot_times_speedy = plot_times_speedy/4
    plot_times_parallel = plot_times_parallel/4#24
    plot_times_llr = plot_times_llr/4 


    i = 0

    plt.rc('font', family='serif') 
    for variable,back_var,unit in zip(variables,backup_vars,units):
        for height_level in height_levels:
            plt.figure(figsize=(12,8))

            plt.plot(plot_times,era_presistence_error_better[i,height_level,forecast_times_peristence],color='#e41a1c',ls='-',linewidth=2.5,label='ERA Persistence')
            plt.plot(plot_times_speedy,[0]+list(speedy_error[i,height_level,forecast_times_speedy]),color='#4daf4a',ls='-',linewidth=2.5,label='SPEEDY')
            plt.plot(plot_times_hybrid[forecast_times_hybrid],hybrid_error[i,height_level,forecast_times_hybrid],color='#377eb8',ls='-',linewidth=2.5,label='Hybrid') 
            if parallel_files is not None:
               plt.plot(plot_times_parallel[forecast_times_parallel],parallel_error[i,height_level,forecast_times_parallel],color='#ff7f00',ls='-',linewidth=2.5,label='ML')
            if llr_files is not None:
               plt.plot(plot_times_llr[forecast_times_llr],llr_error[i,height_level,forecast_times_llr],color='#984ea3',ls='-',linewidth=2.5,label='SPEEDY-LLR')
            plt.plot(plot_times,list(climo_error[i,height_level,forecast_times])+[climo_error[i,height_level,forecast_times[-1]]],color='k',ls='--',linewidth=2.5,label='Climatology')

            plt.title(f"{region} RMSE \n{pressure_levels[height_level]} hPa {variable}",fontsize=22,fontweight='bold')

            plt.legend(fontsize=20)

            plt.xticks(xtick_times,fontsize=22)
            plt.yticks(fontsize=22)
            plt.xlabel('Forecast Day',fontsize=26)
            plt.xlim(0,plot_times[-1])
            plt.ylim(0,np.max([np.max(era_presistence_error_better[i,height_level,forecast_times]),np.max(speedy_error[i,height_level,forecast_times_speedy]),np.max(hybrid_error[i,height_level,forecast_times])])*1.2)#,np.max(parallel_error[height_level,forecast_times_parallel[6:247:6]])]))

            plt.ylabel(f'RMSE ({unit})',fontsize=26)

            plt.tight_layout()

            #plt.savefig(f'{plotdir}{region}_averaged_rmse_{pressure_levels[height_level]}hpa_{variable}_{trialname}.png')
            plt.savefig(f'{plotdir}{region}_averaged_rmse_{pressure_levels[height_level]}hpa_{variable}_{trialname}.pdf')
            plt.close("all")
        i += 1

def mean_and_var_error_3panel_6hr(speedy_files,truth_files,hybrid_files,region,trialname,xlimit_array,parallel_files=None,llr_files=None):

    plt.rc('font', family='serif')

    pressure_levels = [25,95,200,350,500,680,850,950]
    plotdir = '/scratch/user/troyarcomano/letkf-hybrid-speedy/hybrid/plots/'

    plot_times = np.arange(0,5)#np.arange(0,57)
    plot_times_hybrid = np.arange(0,5)#np.arange(0,57)
    plot_times_speedy = np.arange(0,5)#np.arange(0,57)
    plot_times_llr = np.arange(0,5)#np.arange(0,44)
    plot_times_parallel = np.arange(0,5)#np.arange(0,57)

    forecast_times = np.arange(1,5)#np.arange(1,57)
    forecast_times_speedy = np.arange(1,5)#np.arange(1,57)
    forecast_times_llr =  np.arange(1,5)#np.arange(1,44)
    forecast_times_persistence = np.arange(0,5)#np.arange(0,57)
    forecast_times_parallel =  np.arange(1,5)#np.arange(1,57)

    xtick_times = np.arange(0,3,1)

    variables = ['Temperature','V-wind','Specific-Humidity']
    backup_vars = ['Temperature','V-wind','Specific_Humidity']

    ###Varibles Data ####
    mean_persistence_error,mean_speedy_error, var_persistence_error, var_speedy_error = average_mean_and_var_error_better(variables,backup_vars,truth_files,speedy_files,plot_times,region)
    mean_persistence_error,mean_hybrid_error, var_persistence_error, var_hybrid_error = average_mean_and_var_error_better(variables,backup_vars,truth_files,hybrid_files,plot_times_hybrid,region)
    #if parallel_files is not None:
    #   mean_persistence_error_par, mean_parallel_error, var_persistence_error_par, var_parallel_error = average_mean_and_var_error_better_era(variables,backup_vars,truth_files,parallel_files,plot_times_parallel,region,stride=2)
    if llr_files is not None:
       mean_persistence_error_par,mean_llr_error, var_persistence_error_par, var_llr_error = average_mean_and_var_error_better(variables,backup_vars,truth_files,llr_files,plot_times_llr,region)

    '''
    ###Varibles Data ####
    mean_persistence_error,mean_speedy_error, var_persistence_error, var_speedy_error = average_mean_and_var_error_better_era(variables,backup_vars,truth_files,speedy_files,plot_times,region)
    mean_persistence_error,mean_hybrid_error, var_persistence_error, var_hybrid_error = average_mean_and_var_error_better_era(variables,backup_vars,truth_files,hybrid_files,plot_times_hybrid,region)
    #if parallel_files is not None:
    #   mean_persistence_error_par, mean_parallel_error, var_persistence_error_par, var_parallel_error = average_mean_and_var_error_better_era(variables,backup_vars,truth_files,parallel_files,plot_times_parallel,region,stride=2)
    if llr_files is not None:
       mean_persistence_error_par,mean_llr_error, var_persistence_error_par, var_llr_error = average_mean_and_var_error_better_era(variables,backup_vars,truth_files,llr_files,plot_times_llr,region)
    '''


    plot_times = plot_times/4
    plot_times_speedy = plot_times_speedy/4
    plot_times_parallel = plot_times_parallel/4
    plot_times_llr =  plot_times_llr/4#2#4

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(3,3,figsize=(13.5,18))

    ax1 = plt.subplot(3,1,1)
 
   
    print('average rmse',np.sqrt(mean_persistence_error[0,-1,0]**2.0 + var_persistence_error[0,-1,0]**2.0)) 
    ax1.plot(plot_times_speedy,[mean_persistence_error[0,-1,0]]+list(mean_speedy_error[0,-1,forecast_times_speedy]),color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
    ax1.plot(plot_times_speedy,[var_persistence_error[0,-1,0]]+list(var_speedy_error[0,-1,forecast_times_speedy]),color='#4daf4a',ls='--',linewidth=2.0,label='SPEEDY SD Error')
    ax1.plot(plot_times_speedy,[mean_persistence_error[0,-1,0]]+list(mean_hybrid_error[0,-1,forecast_times_speedy]),color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
    ax1.plot(plot_times_speedy,[var_persistence_error[0,-1,0]]+list(var_hybrid_error[0,-1,forecast_times_speedy]),color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
    #ax1.plot(plot_times,mean_persistence_error[0,-1,forecast_times_persistence],color='#e41a1c',ls='-',linewidth=2.0,label='Persistence Mean Error')
    #ax1.plot(plot_times,var_persistence_error[0,-1,forecast_times_persistence],color='#e41a1c',ls='--',linewidth=2.0,label='Persistence SD Error')
    if llr_files is not None:
       ax1.plot(plot_times_llr[forecast_times_llr],mean_llr_error[0,-1,forecast_times_llr],color='#984ea3',ls='-',linewidth=2.0,label='LLR Mean Error')
       ax1.plot(plot_times_llr[forecast_times_llr],var_llr_error[0,-1,forecast_times_llr],color='#984ea3',ls='--',linewidth=2.0,label='LLR SD Error')
    #if parallel_files is not None:
       #ax1.plot(plot_times_parallel[forecast_times_parallel],mean_parallel_error[0,-1,forecast_times_parallel],color='#ff7f00',ls='-',linewidth=2.0,label='ML Model Mean Error')
       #ax1.plot(plot_times_parallel[forecast_times_parallel],var_parallel_error[0,-1,forecast_times_parallel],color='#ff7f00',ls='--',linewidth=2.0,label='ML Model SD Error')
    

    ax1.set_title(f"{region} Mean and Standard Deviation of the Error \n 950 hPa Temperature",fontsize=16, fontweight='bold')

    ax1.legend(fontsize=13)

    ax1.set_xticks(xtick_times)
    ax1.set_xlabel('Forecast Day',fontsize=14)
    ax1.set_xlim(0,plot_times[-1])

    ax1.set_ylabel(f'Error (Kelvin)',fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid()

    ax2 = plt.subplot(3,1,2)

    ax2.plot(plot_times_speedy,[mean_persistence_error[2,-1,0]]+list(mean_speedy_error[2,-1,forecast_times_speedy]),color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
    ax2.plot(plot_times_speedy,[var_persistence_error[2,-1,0]]+list(var_speedy_error[2,-1,forecast_times_speedy]),color='#4daf4a',ls='--',linewidth=2.0,label='SPEEDY SD Error')
    ax2.plot(plot_times_speedy,[mean_persistence_error[2,-1,0]]+list(mean_hybrid_error[2,-1,forecast_times_speedy]),color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
    ax2.plot(plot_times_speedy,[var_persistence_error[2,-1,0]]+list(var_hybrid_error[2,-1,forecast_times_speedy]),color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
    #ax2.plot(plot_times,mean_persistence_error[2,-1,forecast_times_persistence],color='#e41a1c',ls='-',linewidth=2.0,label='Persistence Mean Error')
    #ax2.plot(plot_times,var_persistence_error[2,-1,forecast_times_persistence],color='#e41a1c',ls='--',linewidth=2.0,label='Persistence SD Error')
    if llr_files is not None:
       ax2.plot(plot_times_llr[forecast_times_llr],mean_llr_error[2,-1,forecast_times_llr],color='#984ea3',ls='-',linewidth=2.0,label='LLR Mean Error')
       ax2.plot(plot_times_llr[forecast_times_llr],var_llr_error[2,-1,forecast_times_llr],color='#984ea3',ls='--',linewidth=2.0,label='LLR SD Error')
    #if parallel_files is not None: 
    #   ax2.plot(plot_times_parallel[forecast_times_parallel],mean_parallel_error[2,-1,forecast_times_parallel],color='#ff7f00',ls='-',linewidth=2.0,label='ML Model Mean Error')
    #   ax2.plot(plot_times_parallel[forecast_times_parallel],var_parallel_error[2,-1,forecast_times_parallel],color='#ff7f00',ls='--',linewidth=2.0,label='ML Model SD Error')

    ax2.set_title(f"{region} Mean and Standard Deviation of the Error \n 950 hPa Specific Humidity",fontsize=16, fontweight='bold')

    ax2.legend(fontsize=13)

    ax2.set_xticks(xtick_times)
    ax2.set_xlabel('Forecast Day',fontsize=14)
    ax2.set_xlim(0,plot_times[-1])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel(f'Error (g/kg)',fontsize=14)
    ax2.grid()

    ax3 = plt.subplot(3,1,3)

    ax3.plot(plot_times_speedy,[mean_persistence_error[1,2,0]]+list(mean_speedy_error[1,2,forecast_times_speedy]),color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY Mean Error')
    ax3.plot(plot_times_speedy,[var_persistence_error[1,2,0]]+list(var_speedy_error[1,2,forecast_times_speedy]),color='#4daf4a',ls='--',linewidth=2.0,label='SPEEDY SD Error')
    ax3.plot(plot_times_speedy,[mean_persistence_error[1,2,0]]+list(mean_hybrid_error[1,2,forecast_times_speedy]),color='#377eb8',ls='-',linewidth=2.0,label='Hybrid Mean Error')
    ax3.plot(plot_times_speedy,[var_persistence_error[1,2,0]]+list(var_hybrid_error[1,2,forecast_times_speedy]),color='#377eb8',ls='--',linewidth=2.0,label='Hybrid SD Error')
    #ax3.plot(plot_times,mean_persistence_error[1,2,forecast_times_persistence],color='#e41a1c',ls='-',linewidth=2.0,label='Persistence Mean Error')
    #ax3.plot(plot_times,var_persistence_error[1,2,forecast_times_persistence],color='#e41a1c',ls='--',linewidth=2.0,label='Persistence SD Error')
    if llr_files is not None:
       ax3.plot(plot_times_llr[forecast_times_llr],mean_llr_error[1,2,forecast_times_llr],color='#984ea3',ls='-',linewidth=2.0,label='LLR Mean Error')
       ax3.plot(plot_times_llr[forecast_times_llr],var_llr_error[1,2,forecast_times_llr],color='#984ea3',ls='--',linewidth=2.0,label='LLR SD Error')
    #if parallel_files is not None:
    #   ax3.plot(plot_times[forecast_times_parallel],mean_parallel_error[1,2,forecast_times_parallel],color='#ff7f00',ls='-',linewidth=2.0,label='ML Model Mean Error')
    #   ax3.plot(plot_times[forecast_times_parallel],var_parallel_error[1,2,forecast_times_parallel],color='#ff7f00',ls='--',linewidth=2.0,label='ML Model SD Error')

    ax3.set_title(f"{region} Mean and Standard Deviation of the Error \n 200 hPa V-wind",fontsize=16,fontweight='bold')

    ax3.legend(fontsize=13)

    ax3.set_xticks(xtick_times)
    ax3.set_xlabel('Forecast Day',fontsize=14)
    ax3.set_xlim(0,plot_times[-1])

    ax3.set_ylabel(f'Error (m/s)',fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.grid()

    plt.subplots_adjust(top=0.946, bottom=0.032, left=0.075, right=0.985, hspace=0.253, wspace=0.33)

    #plt.show()
    plt.savefig(f'{plotdir}{region}mean_and_var_error_3panel_{trialname}_badremoved_lat_weighted.png') 
    plt.savefig(f'{plotdir}{region}mean_and_var_error_3panel_{trialname}_badremoved_lat_weighted.pdf')
    plt.close("all")

def surface_pressure_at_grid_point_timeseries(truthfile,parallel_file,hybrid_file,speedy_file):

    variable = 'logp'
    grid_point_x = 55
    grid_point_y = 24
 
    nc_truth = Dataset(truthfile)
    nc_parallel = Dataset(parallel_file)
    nc_hybrid = Dataset(hybrid_file)
    nc_speedy = Dataset(speedy_file)

    truth_sp = nc_truth['logp'][:]
    truth_sp = np.exp(truth_sp)*1000.0

    parallel_sp = nc_parallel['logp'][:]
    parallel_sp = np.exp(parallel_sp)*1000.0 

    hybrid_sp = nc_hybrid['logp'][:]
    hybrid_sp = np.exp(hybrid_sp)*1000.0

    speedy_sp = nc_speedy['logp'][:]
    speedy_sp = np.exp(speedy_sp)*1000.0

    speedy_sp = speedy_sp[0:600:6,:,:]

    times = np.arange(0,600,6)
    plt.title("Tropical Pacific")
    plt.plot(times,speedy_sp[:,grid_point_y,grid_point_x],label='SPEEDY')
    plt.plot(times,hybrid_sp[:,grid_point_y,grid_point_x],label='Hybrid') 
    plt.plot(times,truth_sp[:,grid_point_y,grid_point_x],label='truth')
    plt.plot(times[4::],parallel_sp[4::,grid_point_y,grid_point_x],label='parallel')
    plt.xlabel("Forecast Hours")
    plt.ylabel('hPa') 
    plt.legend()
    plt.show()

def panel_3by3_region_average_better(speedy_files,truth_files,hybrid_files,parallel_files,llr_files,region,variables,backup_vars,units,plot_times,trialname,xlimit_array,llr_truth_files):
    pressure_levels = [25,95,200,350,500,680,850,950]
    pressure_levels_sp = [200,350,500,680,850,950]
    plotdir = '/home/troyarcomano/FortranReservoir/hybridspeedy/plots/'

    timestep = 12

    panel_aug1 = 3
    panel_aug2 = 3
    panel_num = 1

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(3,3,figsize=(15,15))


    era_presistence_error, hybrid_error = rmse_average_better(variables,backup_vars,truth_files,hybrid_files,plot_times,region)
    era_presistence_error_speedy, speedy_error, climo_error = rmse_average_better(variables,backup_vars,truth_files,speedy_files,plot_times,region,stride=6,climatology=True)
    era_presistence_error_parallel, parallel_error = rmse_average_better(variables,backup_vars,truth_files,parallel_files,plot_times,region,stride=2)
    #era_presistence_error_parallel, parallel_error = rmse_average_better(variables,backup_vars,truth_files,parallel_files,plot_times,region)
    era_presistence_error_parallel, llr_error = rmse_average_better(variables,backup_vars,llr_truth_files,llr_files,plot_times,region)#,stride=2)

    i = 0
    for var,backup_var,unit,xlim in zip(variables,backup_vars,units,xlimit_array):
        for time in plot_times:
            ax = plt.subplot(f'{int(panel_aug1)}{int(panel_aug2)}{int(panel_num)}')

            #Set y-scale to be log since pressure decreases exponentially with height
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Set limits, tickmarks, and ticklabels for y-axis
            if var == 'Specific-Humidity':
               ax.set_yticks(list(range(1000, 20, -200))+list(range(100,20,-50))+list(range(25,20,-25)))
               ax.set_yticklabels(list(range(1000, 20, -200))+list(range(100,20,-50))+list(range(25,20,-25)),fontsize=14)
               ax.set_ylim([1000, 200])

            else:
               ax.set_ylim([1000, 25])
               ax.set_yticks(list(range(1000, 20, -200))+list(range(100,20,-50))+list(range(25,20,-25)))
               ax.set_yticklabels(list(range(1000, 20, -200))+list(range(100,20,-50))+list(range(25,20,-25)),fontsize=14)
            
            if var == 'Specific-Humidity':
               #ax.plot(era_presistence_error[i,2::,time],pressure_levels_sp,color='#e41a1c',ls='-',linewidth=2.0,label='Persistence')
               ax.plot(hybrid_error[i,2::,time],pressure_levels_sp,color='#377eb8',ls='-',linewidth=2.0,label='Hybrid 6hr Climo SST')#color='#377eb8',ls='-',linewidth=2.0,label='Hybrid')#'Hybrid')
               #ax.plot(speedy_error[i,2::,time],pressure_levels_sp,color='k',ls='-',linewidth=2.0,label='SPEEDY')#color='4daf4a',ls='-',linewidth=2.0,label='SPEEDY')
               #ax.plot(parallel_error[i,2::,time],pressure_levels_sp,color='g',ls='-',linewidth=2.0,label='Parallel ML')#color='#ff7f00',ls='-',linewidth=2.0,label='Parallel ML') 
               #ax.plot(climo_error[i,2::,time],pressure_levels_sp,color='k',ls='-',linewidth=2.0,label='Climatology') 
               if time < 45:
                  ax.plot(llr_error[i,2::,time],pressure_levels_sp,color='#984ea3',ls='-',linewidth=2.0,label='Hybrid 6hr')#label='SPEEDY-LLR') 
            else:
               print(era_presistence_error[i,:,time])
               #ax.plot(era_presistence_error[i,:,time],pressure_levels,color='#e41a1c',ls='-',linewidth=2.0,label='Persistence')
               ax.plot(hybrid_error[i,:,time],pressure_levels,color='#377eb8',ls='-',linewidth=2.0,label='Hybrid 6hr Climo SST')#color='#377eb8',ls='-',linewidth=2.0,label='Hyrid')#'Hybrid')
               #ax.plot(speedy_error[i,:,time],pressure_levels,color='k',ls='-',linewidth=2.0,label='SPEEDY')#color='#4daf4a',ls='-',linewidth=2.0,label='SPEEDY')
               #ax.plot(parallel_error[i,:,time],pressure_levels,color='g',ls='-',linewidth=2.0,label='Parallel ML')#color='#ff7f00',ls='-',linewidth=2.0,label='Parallel ML')
               #ax.plot(climo_error[i,:,time],pressure_levels,color='k',ls='-',linewidth=2.0,label='Climatology')
               if time < 45*2:
               #   print('int(time/2)',int(time/2))
               #   print('llr_error[i,:,int(time/2)]',llr_error[i,:,int(time/2)])
                  ax.plot(llr_error[i,:,time],pressure_levels,color='#984ea3',ls='-',linewidth=2.0,label='Hybrid 6hr')#label='SPEEDY-LLR')

            ax.set_xlim(xlim)
            ax.tick_params(axis="x", labelsize=14)

            if var == 'Specific-Humidity':
               ax.set_title(f"{region} Specific Humidity Day {int((time*timestep)/24)}",fontsize=16,fontweight="bold") 
            else:
               ax.set_title(f"{region} {var} Day {int((time*timestep)/24)}",fontsize=16,fontweight="bold")

            ax.legend(fontsize=12)

            ax.set_ylabel('hPa',fontsize=16)
            ax.set_xlabel(f'RMSE ({unit})',fontsize=14)

            ax.minorticks_off()
  
            panel_num += 1

        i += 1

    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.25)
    plt.tight_layout()
    #plt.show()

    plt.savefig(f'{plotdir}{region}_hybrid_three_panel_istvan_ams2022{trialname}.png')
    plt.savefig(f'{plotdir}{region}_hybrid_three_panel_istvan_ams2022{trialname}.pdf')
    plt.close("all")


variables = ['Temperature','V-wind','Specific-Humidity']
backup_vars = ['Temperature','V-wind','Specific_Humidity']
units = ['Kelvin','m/s','g/kg']
#plot_times = np.array([4,12,20])
plot_times = np.array([2,6,10])
#plot_times = plot_times/6
height_level = 4#Sigma level
xlimit_array = [[0,6.1],[0,20.0],[0,2.2]]

#trialname = '6000_20_20_20_beta_res1.0_beta_model_0.001_prior_0.0_overlap1_vertlevel_1_speedy_letkf_analysis_9yr_training_fixed_truth_data_and_dates_fixed'
trialname = '6000_10_10_10_beta_res0.001_beta_model_0.001_prior_0.0_overlap1_vertlevel_1_speedy_letkf_analysis_9yr_trainingtrial'

#trialname_parallel = '9000node_28_28_28_noise_beta_0.00001_degree6_cylcing12hr_sigma0.5_radius0.3_0.7_long_prediction_year_simulationtrial'
#trialname_parallel = '9000node_28_28_28_noise_beta_0.001_1hrtimestep_full_data_tisr_test_falsetrial'
#trialname_parallel= '6000_20_20_20_beta_res0.1_beta_model_0.01_prior_0.0_overlap1_era5_6hrtimestep_21yearsnontruncated_data_year_trial'

#trialname_parallel = '9000node_28_28_28_noise_beta_0.00001_3hrtimestep_full_data_tisr_test_false_overlap1trial'
#trialname_parallel = '9000node_28_28_28_noise_beta_0.00001_3hrtimestep_full_data_tisr_test_false_overlap1trial'
trialname_parallel = '9000_30_30_30_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_speedy_letkf_analysis_9yr_training_fixed_truth_data_and_dates_fixedtrial'

#trialname_llr = 'no_reservoir_21yearsnontruncated_data_beta_model_200.0_prior_1.0_overlap1_era5_6hrtimesteptrial' #paper version
#trialname_llr = 'no_reservoir_21yearsnontruncated_data_beta_model_0.01_prior_1.0_0.05noise_speedy_states_overlap1_era5_6hrtimesteptrial'
## paper trialname_llr = 'no_reservoir_21yearsnontruncated_data_beta_model_40_prior_0.0_0.00noise_speedy_states_overlap1_era5_6hrtimesteptrial'
#trialname_llr = 'no_reservoir_21yearsnontruncated_data_beta_model_100_prior_0.0_0.00noise_speedy_states_overlap1_era5_6hrtimesteptrial'
trialname_llr = '6000_20_20_20_beta_res0.01_dp_beta_model_1_prior_0.0_overlap1_era5_6hrtimestep_tisr_full_years_0.00noise_speedy_statestrial'

truth_pattern = f'era_truth{trialname}*.nc'
hybrid_pattern = f'hybrid_prediction_era{trialname}*.nc'

parallel_pattern = f'ml_prediction_era{trialname_parallel}'
#parallel_pattern = f'res_prediction_era{trialname_parallel}*.nc'

llr_pattern = f'hybrid_prediction_era{trialname_llr}'
#llr_truth_pattern = f'era_truth{trialname_llr}*.nc'
llr_truth_pattern = f'era_truth{trialname}*.nc'

path = '/scratch/user/troyarcomano/Predictions/Hybrid/'
#path = '/tiered/user/troyarcomano/hybrid_prediction_data/'

truth_files = get_files(path,truth_pattern)
hybrid_files = get_files(path,hybrid_pattern)
speedy_files = get_speedy_files(sorted(truth_files))
ml_only_files = get_parallel_files(sorted(truth_files),parallel_pattern)

truth_files = sorted(truth_files)
hybrid_files = sorted(hybrid_files)
speedy_files = sorted(speedy_files)
ml_only_files = sorted(ml_only_files)

speedy_files = speedy_files[4::]
hybrid_files = hybrid_files[4::]#[1::]
truth_files = truth_files[4::]#[1::]
ml_only_files = ml_only_files[4::]

print(len(speedy_files))
print(len(hybrid_files))
print(len(truth_files))
print(len(ml_only_files))

print(speedy_files[0])
print(hybrid_files[0])
print(truth_files[0])

print(speedy_files[-1])
print(hybrid_files[-1])
print(truth_files[-1])

for speedy_file,hybrid_file,truth_file in zip(speedy_files,hybrid_files,truth_files):
    print(speedy_file)
    print(hybrid_file)
    print(truth_file)
#print(len(parallel_truth_files))

#plot_surface_tendecy(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],llr_files[:],'logp','logp','hPa')

#average_error_map(speedy_files,truth_files,hybrid_files,parallel_files,'V-wind','V-wind','m/s',height_level,36,trialname,xlimit_array,0.5,5)

#average_bias_and_sd_error_map(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],'Specific-Humidity','Specific_Humidity','g/kg',7,6,trialname,xlimit_array,0.2,1)
region = 'NH'
#panel_3by3_region_average_better(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],llr_files[:],region,variables,backup_vars,units,plot_times,trialname,xlimit_array,llr_truth_files)
region = 'SH'
xlimit_array = [[0,7.0],[0,28.0],[0,2.2]]
#panel_3by3_region_average_better(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],llr_files[:],region,variables,backup_vars,units,plot_times,trialname,xlimit_array,llr_truth_files)
region = 'Tropics'
xlimit_array = [[0,6.5],[0,15.0],[0,2.5]]
#panel_3by3_region_average_better(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],llr_files[:],region,variables,backup_vars,units,plot_times,trialname,xlimit_array,llr_truth_files)
#surface_pressure_at_grid_point_timeseries('era_truth6000_20_20_20_beta_res0.1_beta_model_0.01_prior_0.0_overlap1_era5_bug_fixed_6hr_timestep_moresteps_cant_die_shtrial_06_21_1990_00.nc','/scratch/user/troyarcomano/FortranReservoir/speedyintegration/res_prediction_era9000node_28_28_28_noise_beta_0.001_6hrtimestep_full_datatrial_06_22_1990_00.nc','hybrid_prediction_era6000_20_20_20_beta_res0.1_beta_model_0.01_prior_0.0_overlap1_era5_bug_fixed_6hr_timestep_moresteps_cant_die_shtrial_06_21_1990_00.nc','/tiered/user/troyarcomano/parallel_prediction_data/speedy_data/speedy_era_start06_21_1990_00.nc')

region = 'Global'
mean_and_var_error_3panel_6hr(speedy_files[:],truth_files[:],hybrid_files[:],region,trialname,xlimit_array,parallel_files=ml_only_files[:]) #,llr_files=llr_files[:])

height_levels = np.arange(0,8)
#timeseries_rmse_plots_6hr(speedy_files[:],truth_files[:],hybrid_files[:],region,variables,backup_vars,units,height_levels,trialname,xlimit_array,parallel_files=parallel_files[:],llr_files=llr_files[:])
#timeseries_rmse_plots_6hr(speedy_files[:],truth_files[:],hybrid_files[:],region,'Temperature','Temperature','Kelvin',height_levels,trialname,xlimit_array,parallel_files=parallel_files[:],llr_files=llr_files[:])#,parallel_files=parallel_files[:],parallel_truth_files=truth_files[:])
#timeseries_rmse_plots_6hr(speedy_files[:],truth_files[:],hybrid_files[:],region,'Specific-Humidity','Specific_Humidity','g/kg',height_levels,trialname,xlimit_array,parallel_files=parallel_files[:],llr_files=llr_files[:])#,parallel_files=parallel_files[:],parallel_truth_files=truth_files[:])#,parallel_files=parallel_files[:],parallel_truth_files=parallel_truth_files[:])
region = 'Tropics'
#timeseries_rmse_plots_6hr(speedy_files[:],truth_files[:],hybrid_files[:],region,'Specific-Humidity','Specific_Humidity','g/kg',height_levels,trialname,xlimit_array,parallel_files=parallel_files[:],llr_files=llr_files[:])
#region = 'NH'
#timeseries_rmse_plots_6hr(speedy_files[:],truth_files[:],hybrid_files[:],region,'V-wind','V-Wind','m/s',height_levels,trialname,xlimit_array)
#timeseries_rmse_plots_6hr(speedy_files,truth_files,hybrid_files,region,'Temperature','Temperature','Kelvin',height_levels,trialname,xlimit_array)


#mean_and_var_error_3panel(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],region,trialname,xlimit_array)
#panel_3by3_region_average(speedy_files,truth_files,hybrid_files,parallel_files,region,variables,backup_vars,units,plot_times,trialname,xlimit_array)

#timeseries_rmse_plots(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],region,'Temperature','Temperature','Kelvin',7,trialname,xlimit_array)
#timeseries_rmse_plots(speedy_files,truth_files,hybrid_files,parallel_files,region,'Specific-Humidity','Specific_Humidity','g/kg',7,trialname,xlimit_array)
'''
height_level = np.arange(0,8)
for region in ['Global']:#,'NH','SH','Tropics']: #['Global','NH','SH','Tropics']
    for i in range(len(variables)):
        mean_and_var_error_timeseries(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],region,variables[i],backup_vars[i],units[i],height_level,trialname,xlimit_array)
        #timeseries_rmse_plots(speedy_files[:],truth_files[:],hybrid_files[:],parallel_files[:],region,variables[i],backup_vars[i],units[i],height_level,trialname,xlimit_array)
region = 'Global'
#for region in ['Global','NH','SH','Tropics']:
#for var,backup_var,unit in zip(variables, backup_vars,units):
#    for i in range(8):
#        timeseries_rmse_plots(speedy_files,truth_files,hybrid_files,parallel_files,region,var,backup_var,unit,i,trialname,xlimit_array)

region = 'NH'
panel_3by3_region_average(speedy_files,truth_files,hybrid_files,parallel_files,region,variables,backup_vars,units,plot_times,trialname,xlimit_array)
#for i in range(len(truth_files)):
#    panel_3by3_region(speedy_files[i],truth_files[i],res_files[i],region,variables,backup_vars,units,plot_times,trialname,xlimit_array) 
#panel_3by3_region('/scratch/user/troyarcomano/FortranReservoir/speedyintegration/parallel_prediction_data/speedy_9000node_20_10_10_noise_beta_0.000001_radius_0.6_degree6_cylcing_regional_standardization_yeslogqtrial11_27_1989_00.nc','era_truth9000node_25_20_15_noise_beta_0.0001_degree9_cylcing12hr_sigma0.5_regional_standardization_regional_varying_radiustrial_11_27_1989_00.nc','res_prediction_era9000node_25_20_15_noise_beta_0.0001_degree9_cylcing12hr_sigma0.5_regional_standardization_regional_varying_radiustrial_11_27_1989_00.nc',region,variables,backup_vars,units,plot_times,trialname,xlimit_array)

region = 'SH'
panel_3by3_region_average(speedy_files,truth_files,hybrid_files,parallel_files,region,variables,backup_vars,units,plot_times,trialname,xlimit_array)

region = 'Tropics'
panel_3by3_region_average(speedy_files,truth_files,hybrid_files,parallel_files,region,variables,backup_vars,units,plot_times,trialname,xlimit_array)
'''
