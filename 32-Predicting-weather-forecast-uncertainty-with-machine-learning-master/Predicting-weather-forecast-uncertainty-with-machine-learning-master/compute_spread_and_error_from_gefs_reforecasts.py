#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:47:09 2017

compute forecast RMSE and spread averaged over Europe for the 
GEFS reforecasts and save to .csv files

@author: sebastian
"""
import os
import matplotlib
matplotlib.use('agg')

import xarray as xr
import pandas as pd
import numpy as np
from pylab import plt
from dask.diagnostics import ProgressBar
ProgressBar().register()

def sellonlatbox(data,west,north,east,south):

    if north == south and west==east:
        return selclosestpoint(data,west,north)
    else:

        indexers = {'lon':slice(west,east), 'lat':slice(north,south)}

        sub = data.sel(**indexers)

        return sub

def selclosestpoint(data,lon,lat):
    sub = data.sel(lat=lat,lon=lon,method='nearest')
    print('selcolestepoint')
    return sub

def wrap360_to180(ds, lon='lon'):
    """
    wrap longitude coordinates from 0..360 to -180..180

    Parameters
    ----------
    ds : Dataset
        object with longitude coordinates
    lon : string
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.

    from https://github.com/pydata/xarray/issues/577
    """

    # wrap 0..359 to  -179...180
    new = np.array([e if e < 180 else e-360 for e in ds[lon].values])


    ds = ds.assign(**{ lon : new})
    ds = ds.reindex({lon: np.sort(new)})

    return ds
def standardize_dataset(ds):
    ''' change dimension names to standard names (lat,lon,lev,time),

    return: None
    '''
    pairs = {'latitude':'lat',
             'longitude':'lon',
             'level':'lev'}
    for key in pairs.keys():
        if key in ds.dims.keys():
            ds = ds.rename({key:pairs[key]})

    # extract variable name
    var_keys = ds.data_vars.keys()
    assert ( len(var_keys) ==1)
    for e in var_keys:
        name = e
    ds.attrs['varname'] = name

    #  #if necessary, warp coordindates
    # if np.any(ds.lon > 180):
    #     ds = wrap360_to180(ds)

    return ds




def read_gefs_reforecast_ctrl(date):
    date = date.strftime('%Y%m%d')
    path = '/climstorage/sebastian/GEFS_reforecast/hgt_pres/netcdf/'

    ifile = path+'/'+'hgt_pres_'+date+'00_c00.grib2.nc'

    if not os.path.exists(ifile):
        print(ifile,' not available, skipping!')
        raise FileNotFoundError()


    ds = xr.open_dataset(ifile,autoclose=True)
    ds = standardize_dataset(ds)
    ds = wrap360_to180(ds)['gh']


    z = sellonlatbox(ds,*subreg)

    return z



subreg= [-20,80,50,20]





#date=pd.to_datetime('20010101')
startdate = '19850101'
enddate = '20161231'



# read in spread
# the spread data was donwloadedin 3 different files, split along forecast hour
spread_all = xr.open_mfdataset(['/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_sprd_19841201_20161231_lead1.nc',
                                '/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_sprd_19841201_20161231_lead2.nc',
                                '/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_sprd_19841201_20161231_lead3.nc'], chunks={'time':1})

ctrl_all = xr.open_mfdataset(['/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_c00_19841201_20161231_lead1.nc',
                                '/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_c00_19841201_20161231_lead2.nc',
                                '/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_c00_19841201_20161231_lead3.nc'], chunks={'time':1})

# ensmean_all = xr.open_mfdataset(['/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_mean_19841201_20161231_lead1.nc',
#                                 '/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_mean_19841201_20161231_lead2.nc',
#                                 '/climstorage/sebastian/GEFS_reforecast/ordered_data/hgt_pres_latlon_mean_19841201_20161231_lead3.nc'], chunks={'time':1})





spread_all = wrap360_to180(spread_all)['Geopotential_height']
spread_all = sellonlatbox(spread_all.isel(lat=slice(None,None,-1)), *subreg).sel(pressure=500)

ctrl_all = wrap360_to180(ctrl_all)['Geopotential_height']
ctrl_all = sellonlatbox(ctrl_all.isel(lat=slice(None,None,-1)), *subreg).sel(pressure=500)

# ensmean_all = wrap360_to180(ensmean_all)['Geopotential_height']
# ensmean_all = sellonlatbox(ensmean_all.isel(lat=slice(None,None,-1)), *subreg).sel(pressure=500)

# ensmean_all.load()
ctrl_all.load()
spread_all.load()

#%%


lead_days  = np.arange(1,14)


dates = {lead_day:[] for lead_day in lead_days}
rmse_ctrl_res = {lead_day:[] for lead_day in lead_days}
# rmse_ensmean_res = {lead_day:[] for lead_day in lead_days}
spread_res = {lead_day:[] for lead_day in lead_days}

failed_dates = []
for date in pd.date_range(startdate,enddate):
    print(date)


    try:

        for lead_day in lead_days:
            date_truth = date + pd.Timedelta(str(lead_day)+' Days')
            # select timeslice after lead_day days
            ctrl = ctrl_all.sel(time=date, fhour=pd.Timedelta(str(lead_day)+'d'))

            spread = spread_all.sel(time=date, fhour=pd.Timedelta(str(lead_day)+'d'))
            assert(ctrl.shape==spread.shape)
            spread  = spread.mean(('lat','lon')).values
            # ensmean = ensmean_all.sel(time=date_truth, fhour=pd.Timedelta(str(lead_day)+'d'))

            # read "truth"

            truth = read_gefs_reforecast_ctrl(date_truth).sel(lev=50000)

            truth = truth.sel(time=date_truth)



            assert(truth.shape==ctrl.shape)
            #assert(truth.shape==ensmean.shape)


            rmse_ctrl = np.sqrt(np.mean((ctrl - truth)**2)).values
            # rmse_ensmean = np.sqrt(np.mean((ensmean - truth)**2)).values


            rmse_ctrl_res[lead_day].append(rmse_ctrl)
            # rmse_ensmean_res[lead_day].append(rmse_ensmean)
            spread_res[lead_day].append(spread)
            dates[lead_day].append(date)

    except (KeyError, FileNotFoundError):
        print(date,'_failed')
        failed_dates.append(date)
        continue

print('the following dates failed',failed_dates)

for lead_day in lead_days:


    rmse_ctrl = np.array(rmse_ctrl_res[lead_day])
    # rmse_ensmean = np.array(rmse_ensmean_res[lead_day])
    spread = np.array(spread_res[lead_day])


    res_df = pd.DataFrame({'ctrl':rmse_ctrl,
    #'ensmean':rmse_ensmean,
    'spread':spread}, index=pd.DatetimeIndex(dates[lead_day]))

    res_df.to_csv('RMSE_gefsreforecasts_ctrl_and_spread_from_ordered_'+str(subreg)+'_'+str(startdate)+'_'+str(enddate)+'_day'+str(lead_day)+'.csv')
