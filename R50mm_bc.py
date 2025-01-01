import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np 
from scipy import stats
import cftime
import matplotlib.pyplot as plt 
import os 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
import glob

# ======================================================================================
# This script proceed bias-correction for R50mm/pd50 over Indonesia
# ======================================================================================

# Define data paths
modpath = "/Volumes/dec_Degree/etcddi_biascorrection_ID/data/raw/mod_cmip6/"
obspath = "/Volumes/dec_Degree/etcddi_biascorrection_ID/data/raw/obs_chirps/"
out_dir = '/Volumes/dec_Degree/etcddi_biascorrection_ID/data/processed/bias_corrected/'

model_list = ['CESM2-WACCM', 'CNRM-ESM2-1', 'MPI-ESM1-2-LR']
mpi_var = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1'] # One of the model includes three different variants

# Define several function
# -----------------------

# Create a function to read in pd50 index
def open_ncfile(filapaths):
    with xr.open_dataset(filapaths) as ds:
        print(ds.keys())
    return ds.precipitation_days_index_per_time_period

# Define a function to reindex all of the coordinate dimensions in obs_da to 
# avoid error in xarray.apply_ufunc()
def reindexing_dims(da_1, da_2): # da_1 for obs and da_2 for the model dataset
    tim_reindexed = da_1.reindex({"time": da_2.get_index('time')},)
    lat_reindexed = tim_reindexed.reindex({'latitude': da_2.get_index('latitude')},)
    lon_reindexed = lat_reindexed.reindex({'longitude': da_2.get_index('longitude')},)
    return lon_reindexed.fillna(da_1.values)

# Define Polynomial and Linear Regression model
qubic_feature = PolynomialFeatures(degree=3, include_bias=False)
polynom_model = LinearRegression(fit_intercept=False) # set the intercept to 0
linear_model = LinearRegression(fit_intercept=False)
# Define a function for baseline bias-correction
def biascorrection_baseline(obs, base, qubic_feature, polynom_model, linear_model):
    # Check for NaN values in ocean data
    S_no_nan = obs[~np.isnan(obs)]
    N = len(obs)
    N2 = len(S_no_nan)
    if ((N2/N) == 0):
        out = np.empty(len(obs))
        out[:] = np.nan
        return out#if the data point is in ocean than return np.nan for output

    # Event-based correction
    X_train = np.array(base).reshape(-1,1)
    y_train = np.array(obs)
    linear_model.fit(X_train, y_train)
    y_predict = linear_model.predict(X_train)

    # Distribution-based correction (Piani et.al 2010)
    obs_hist = np.histogram(obs, bins=100)
    obs_dist = stats.rv_histogram(obs_hist,density=True) # return distribustion using the given histogram
    mdl_hist = np.histogram(y_predict, bins=100)
    mdl_dist = stats.rv_histogram(mdl_hist,density=True)

    # Inverse CDF calculation
    prob_grid = np.linspace(0, 1, 100)
    obs_invcdf = obs_dist.ppf(prob_grid)
    mdl_invcdf = mdl_dist.ppf(prob_grid)

    # Fit the model
    mdl_to_train = mdl_invcdf.reshape(-1,1)
    mdl_train_qubic = qubic_feature.fit_transform(mdl_to_train)
    polynom_model.fit(mdl_train_qubic, obs_invcdf)

    # Correcting baseline data
    mdl_to_correct = np.array(y_predict).reshape(-1,1)
    mdl_test_qubic = qubic_feature.transform(mdl_to_correct)
    mdl_corrected = polynom_model.predict(mdl_test_qubic)

    return mdl_corrected

# Define a function for future bias-correction
def biascorrection_future(obs, base, future, qubic_feature, polynom_model, linear_model):
  # Check for NaN values in ocean data
  S_no_nan = obs[~np.isnan(obs)]
  N = len(obs)
  N2 = len(S_no_nan)
  if ((N2/N) == 0):
    out = np.empty(len(obs))
    out[:] = np.nan
    return out #if the data point is in ocean than return np.nan for output

  # Event-based correction
  X_train = np.array(base).reshape(-1,1)
  y_train = np.array(obs)
  linear_model.fit(X_train, y_train)
  y_predict = linear_model.predict(X_train)

  # Distribution-based correction (Piani et.al 2010)
  obs_hist = np.histogram(obs, bins=100)
  obs_dist = stats.rv_histogram(obs_hist,density=True)
  mdl_hist = np.histogram(y_predict, bins=100)
  mdl_dist = stats.rv_histogram(mdl_hist,density=True)

  # Inverse CDF calculation
  prob_grid = np.linspace(0, 1, 100)
  obs_invcdf = obs_dist.ppf(prob_grid)
  mdl_invcdf = mdl_dist.ppf(prob_grid)

  # Fit the model
  mdl_to_train = mdl_invcdf.reshape(-1,1)
  mdl_train_qubic = qubic_feature.fit_transform(mdl_to_train)
  polynom_model.fit(mdl_train_qubic, obs_invcdf)

  # Correcting future data
  future_predict = linear_model.predict(np.array(future).reshape(-1,1))
  mdl_to_correct = np.array(future_predict).reshape(-1,1)
  mdl_test_qubic = qubic_feature.transform(mdl_to_correct)
  mdl_corrected = polynom_model.predict(mdl_test_qubic)

  return mdl_corrected

# Define the directories of baseline and future dataset
all_bpath = []
all_fpath = []
# Get all of data path for baseline and future
for model in model_list:
    print(model)
    model_name = model
    if model_name != model_list[-1]:
        base_dir = modpath+'baseline/'+model_name+'/'
        #print("Here is baseline data directory:\n", base_dir)
        future_dir = modpath+'future/'+model_name+'/'
        #print("Here is future data directory:\n", future_dir)
        all_bpath.append(base_dir)
        all_fpath.append(future_dir)
    else:
        for model_var in mpi_var:
            base_dir = modpath+'baseline/'+model_name+'/'+model_var+'/'
            #print("Here is baseline data directory for MPI-ESM1-2-LR model:\n", base_dir)
            future_dir = modpath+'future/'+model_name+'/'+model_var+'/'
            #print("Here is future data directory for MPI-ESM1-2-LR model:\n", future_dir)
            all_bpath.append(base_dir)
            all_fpath.append(future_dir)

# Define the file pattern to match
base_name = "*pd50*.nc"
solar_name = "*G6solar*pd50*.nc"
sulfur_name = "*G6sulfur*pd50*.nc"
ssp245_name = "*ssp245*pd50*.nc"
ssp585_name = "*ssp585*pd50*.nc"

# Read in observation dataset
obs_da = open_ncfile(glob.glob(obspath+base_name)[0]).sel(
        time=slice("1990","2014"))

# Begin the bias-correction process and loop over each model directory
# --------------------------------------------------------------------
for baseline,future in zip(all_bpath,all_fpath):
    # Construct the full pattern with the directory path
    # and read in all of the data
    base_pattern = os.path.join(baseline, base_name)
    base_da = open_ncfile(glob.glob(base_pattern)[0]).sel(
        time=slice("1990","2014")) #print(base_file)
    
    solar_pattern = os.path.join(future, solar_name)
    solar_da = open_ncfile(glob.glob(solar_pattern)[0]) #print(solar_file)
    
    sulfur_pattern = os.path.join(future, sulfur_name)
    sulfur_da = open_ncfile(glob.glob(sulfur_pattern)[0]) #print(sulfur_file)
    
    ssp245_pattern = os.path.join(future, ssp245_name)
    ssp245_da = open_ncfile(glob.glob(ssp245_pattern)[0]) #print(ssp245_file)

    ssp585_pattern = os.path.join(future, ssp585_name)
    ssp585_da = open_ncfile(glob.glob(ssp585_pattern)[0]) #print(ssp585_file)
    
    # select future period for G6solar dataset
    solar_1 = solar_da.isel(time=slice(0,25))
    solar_2 = solar_da.isel(time=slice(25,50))
    solar_3 = solar_da.isel(time=slice(50,75))
    solar_4 = solar_da.isel(time=slice(-26,-1)) # get the last 25 data series
    solar_list = [solar_1, solar_2, solar_3,solar_4]
    
    # select future period for G6sulfur dataset
    sulfur_1 = sulfur_da.isel(time=slice(0,25))
    sulfur_2 = sulfur_da.isel(time=slice(25,50))
    sulfur_3 = sulfur_da.isel(time=slice(50,75))
    sulfur_4 = sulfur_da.isel(time=slice(-26,-1))
    sulfur_list = [sulfur_1, sulfur_2, sulfur_3, sulfur_4]
    
    # select future period for ssp245 dataset
    ssp245_1 = ssp245_da.isel(time=slice(0,25))
    ssp245_2 = ssp245_da.isel(time=slice(25,50))
    ssp245_3 = ssp245_da.isel(time=slice(50,75))
    ssp245_4 = ssp245_da.isel(time=slice(-26,-1))
    ssp245_list = [ssp245_1,ssp245_2,ssp245_3,ssp245_4]
    
    # select future period for ssp585 dataset
    ssp585_1 = ssp585_da.isel(time=slice(0,25))
    ssp585_2 = ssp585_da.isel(time=slice(25,50))
    ssp585_3 = ssp585_da.isel(time=slice(50,75))
    ssp585_4 = ssp585_da.isel(time=slice(-26,-1))
    ssp585_list = [ssp585_1,ssp585_2,ssp585_3,ssp585_4]
    
    # Reindex obs dataarray
    reindexed_obs_da = reindexing_dims(obs_da,base_da)
    
    # Reindex future dataarray
    reindexed_solar_list = []
    for item in solar_list:
        reindexed_da = reindexing_dims(item, base_da)
        reindexed_solar_list.append(reindexed_da)
    
    reindexed_sulfur_list = []
    for item in ssp245_list:
        reindexed_da = reindexing_dims(item, base_da)
        reindexed_sulfur_list.append(reindexed_da)
        
    reindexed_ssp245_list = []
    for item in sulfur_list:
        reindexed_da = reindexing_dims(item, base_da)
        reindexed_ssp245_list.append(reindexed_da)

    reindexed_ssp585_list = []
    for item in ssp585_list:
        reindexed_da = reindexing_dims(item, base_da)
        reindexed_ssp585_list.append(reindexed_da)

    # Apply bias-correction function for baseline data
    baseline_corrected = xr.apply_ufunc(
        biascorrection_baseline,
        reindexed_obs_da,
        base_da, # baseline periode
        qubic_feature,
        polynom_model,
        linear_model,
        input_core_dims=[["time"],["time"], [], [], []],
        output_core_dims=[['time']],
        vectorize=True,)
    bias_corrected_baseline = baseline_corrected.transpose('time', ...)
    
    # Apply bias-correction function for future data
    # G6solar simulation
    bc_solar_list = []
    for da in reindexed_solar_list:
        bc = xr.apply_ufunc(biascorrection_future,
                            reindexed_obs_da,
                            base_da,
                            da,
                            qubic_feature,
                            polynom_model,
                            linear_model,
                            input_core_dims=[["time"],["time"],["time"], [], [], []],
                            output_core_dims=[['time']],
                            vectorize=True,)
        bc_T = bc.transpose('time', ...)
        bc_solar_list.append(bc_T)
    # G6sulfur simulation
    bc_sulfur_list = []
    for da in reindexed_sulfur_list:
        bc = xr.apply_ufunc(biascorrection_future,
                            reindexed_obs_da,
                            base_da, # baseline periode
                            da,
                            qubic_feature,
                            polynom_model,
                            linear_model,
                            input_core_dims=[["time"],["time"],["time"], [], [], []],
                            output_core_dims=[['time']],
                            vectorize=True,)
        bc_T = bc.transpose('time', ...)
        bc_sulfur_list.append(bc_T)
    # SSP2-4.5 simulation
    bc_ssp245_list = []
    for da in reindexed_ssp245_list:
        bc = xr.apply_ufunc(biascorrection_future,                        
                            reindexed_obs_da,
                            base_da, # baseline periode
                            da,
                            qubic_feature,
                            polynom_model,
                            linear_model,
                            input_core_dims=[["time"],["time"],["time"], [], [], []],
                            output_core_dims=[['time']],
                            vectorize=True,)
        bc_T = bc.transpose('time', ...)
        bc_ssp245_list.append(bc_T)
    # SSP5-8.5 simulation
    bc_ssp585_list = []
    for da in reindexed_ssp585_list:
        bc = xr.apply_ufunc(biascorrection_future,
                            reindexed_obs_da,
                            base_da, # baseline periode
                            da,
                            qubic_feature,
                            polynom_model,
                            linear_model,
                            input_core_dims=[["time"],["time"],["time"], [], [], []],
                            output_core_dims=[['time']],
                            vectorize=True,)
        bc_T = bc.transpose('time', ...)
        bc_ssp585_list.append(bc_T)
    
    # Get time attributs
    time_att = base_da.time.attrs
    
    # Define a function to assign dataarray time attributes
    def assign_attrs(da): 
        da['time'] = da.time.assign_attrs(time_att)
        return da
    
    # Assign new time attributes to bias-corrected data
    bias_corrected_baseline = assign_attrs(
        bias_corrected_baseline)
    
    solar_with_attrs = []
    for da in bc_solar_list:
        solar_with_attrs.append(assign_attrs(da))

    sulfur_with_attrs = []
    for da in bc_sulfur_list:
        sulfur_with_attrs.append(assign_attrs(da))

    ssp245_with_attrs = []
    for da in bc_ssp245_list:
        ssp245_with_attrs.append(assign_attrs(da))

    ssp585_with_attrs = []
    for da in bc_ssp585_list:
        ssp585_with_attrs.append(assign_attrs(da))
    
    # Define a new function to return the original time index of future dataset
    def return_indextime(da_b, da_f): # baseline and future dataarray
        returned_index = da_b.reindex({"time": da_f.get_index('time')},)
        return returned_index.fillna(da_b.values)
    
    # Return the original index of future data
    bc_solar_reindex = []
    for i in range(len(solar_with_attrs)):
        reindex_da = return_indextime(solar_with_attrs[i], solar_list[i])
        bc_solar_reindex.append(reindex_da)

    bc_sulfur_reindex = []
    for i in range(len(sulfur_with_attrs)):
        reindex_da = return_indextime(sulfur_with_attrs[i], sulfur_list[i])
        bc_sulfur_reindex.append(reindex_da)

    bc_ssp245_reindex = []
    for i in range(len(ssp245_with_attrs)):
        reindex_da = return_indextime(ssp245_with_attrs[i], ssp245_list[i])
        bc_ssp245_reindex.append(reindex_da)

    bc_ssp585_reindex = []
    for i in range(len(ssp585_with_attrs)):
        reindex_da = return_indextime(ssp585_with_attrs[i], ssp585_list[i])
        bc_ssp585_reindex.append(reindex_da)
    
    # Combining data arrays along a single dimension (time),
    # and drop duplicate values along the dimension
    solar_bcorr = xr.concat([bc_solar_reindex[0], bc_solar_reindex[1],
                        bc_solar_reindex[2], bc_solar_reindex[3]], dim="time")
    solar_bcorr = solar_bcorr.drop_duplicates(dim="time") 
    
    sulfur_bcorr = xr.concat([bc_sulfur_reindex[0], bc_sulfur_reindex[1],
                        bc_sulfur_reindex[2], bc_sulfur_reindex[3]], dim="time")
    sulfur_bcorr = sulfur_bcorr.drop_duplicates(dim="time") 
    
    ssp245_bcorr = xr.concat([bc_ssp245_reindex[0], bc_ssp245_reindex[1],
                        bc_ssp245_reindex[2], bc_ssp245_reindex[3]], dim="time")
    ssp245_bcorr = ssp245_bcorr.drop_duplicates(dim="time")
    
    ssp585_bcorr = xr.concat([bc_ssp585_reindex[0], bc_ssp585_reindex[1],
                        bc_ssp585_reindex[2], bc_ssp585_reindex[3]], dim="time")
    ssp585_bcorr = ssp585_bcorr.drop_duplicates(dim="time")
    
    # Convert data array into dataset form
    bc_baseline_ds = bias_corrected_baseline.to_dataset(dim=None,
                                name='precipitation_days_index_per_time_period')

    bc_solar_ds = solar_bcorr.to_dataset(dim=None, 
                            name='precipitation_days_index_per_time_period')
        

    bc_sulfur_ds = sulfur_bcorr.to_dataset(dim=None,
                            name='precipitation_days_index_per_time_period')

    bc_ssp245_ds = ssp245_bcorr.to_dataset(dim=None, 
                            name='precipitation_days_index_per_time_period')

    bc_ssp585_ds = ssp585_bcorr.to_dataset(dim=None, 
                            name='precipitation_days_index_per_time_period')
    
    # Save the dataset to a new netcdf file
    base_filename = os.path.basename(glob.glob(base_pattern)[0])
    bc_baseline_ds.to_netcdf(
        out_dir+'bias-corrected_'+ base_filename,
        format = "NETCDF4",
        engine ="netcdf4",
        encoding= {"precipitation_days_index_per_time_period": {"dtype": "float32"}},
        unlimited_dims='time')
    print(f"Exporting files:{base_filename}")
    
    solar_filename = os.path.basename(glob.glob(solar_pattern)[0])
    bc_solar_ds.to_netcdf(
        out_dir+'bias-corrected_'+ solar_filename,
        format = "NETCDF4",
        engine ="netcdf4",
        encoding= {"precipitation_days_index_per_time_period": {"dtype": "float32"}},
        unlimited_dims='time')
    print(f"Exporting files:{solar_filename}")
    
    sulfur_filename = os.path.basename(glob.glob(sulfur_pattern)[0])
    bc_solar_ds.to_netcdf(
        out_dir+'bias-corrected_'+ sulfur_filename,
        format = "NETCDF4",
        engine ="netcdf4",
        encoding= {"precipitation_days_index_per_time_period": {"dtype": "float32"}},
        unlimited_dims='time')
    print(f"Exporting files:{sulfur_filename}")
    
    ssp245_filename = os.path.basename(glob.glob(ssp245_pattern)[0])
    bc_solar_ds.to_netcdf(
        out_dir+'bias-corrected_'+ ssp245_filename,
        format = "NETCDF4",
        engine ="netcdf4",
        encoding= {"precipitation_days_index_per_time_period": {"dtype": "float32"}},
        unlimited_dims='time')
    print(f"Exporting files:{ssp245_filename}")
    
    ssp585_filename = os.path.basename(glob.glob(ssp585_pattern)[0])
    bc_solar_ds.to_netcdf(
        out_dir+'bias-corrected_'+ ssp585_filename,
        format = "NETCDF4",
        engine ="netcdf4",
        encoding= {"precipitation_days_index_per_time_period": {"dtype": "float32"}},
        unlimited_dims='time')
    print(f"Exporting files:{ssp585_filename}")
    
print("This is the end of bias-correction process for pd50 index")
# -----------------------------------------------------------------