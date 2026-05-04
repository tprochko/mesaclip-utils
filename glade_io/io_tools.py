"""
These functions are ease-of-use functions for accessing MESACLIP datasets on NCAR's 
glade system. There are a few inconsistencies with the organization of the directories
and naming conventions, so these functions smooth the process for the user. 
Understanding of the glade file system is required.
"""
import os
import glob
from functools import partial
import warnings

import numpy as np
import xarray as xr
import cftime

warnings.filterwarnings('always')


"""
===============================================================================================
===============================================================================================
===================================== preprocess functions ====================================
===============================================================================================
===============================================================================================
"""

def time_set_midmonth(ds, time_name, deep=False):
    """
    Courtesy Steve Yeager (https://github.com/sgyeager/POP_MOC/)
    Return copy of ds with values of ds[time_name] replaced with mid-month
    values (day=15) rather than end-month values.
    
    """
    #ds_out = ds.copy(deep)
    year = ds[time_name].dt.year
    month = ds[time_name].dt.month
    year = xr.where(month==1,year-1,year)
    month = xr.where(month==1,12,month-1)
    nmonths = len(month)
    newtime = [cftime.DatetimeNoLeap(year[i], month[i], 15) for i in range(nmonths)]
    ds[time_name] = newtime
    return ds

def _preprocessor_month_1(ds0,keepvars,isel,sel,drop_cell,shift_time):
    """
    Based on Steve Yeager's preprocessor (https://github.com/sgyeager/POP_MOC/)
    Preprocessor for xarray xr.open_dataset() function that adjusts the
    time coordinates of monthly POP2 output from the first second of the 
    month following the averaging period to the middle day of the month 
    of the averaging period (e.g. output for mean March 2002 is saved as
    2002-04-01 00:00:00. The preprocessor sets this to 2002-03-15 00:00:00.)
    Has the optional input of selecting only desired variables.
    --- Input ---
        ds0: input dataset, this is a preprocess step so this is not an object
        keepvars: str or list of str, indicating desired variables. Defaults
                                      to None.
    --- Output ---
        None, internal to xr.open_dataset() function.
    """
    if 'z_t' in ds0.data_vars:
        ds0['z_t'] = ds0.data_vars['z_t']
        ds0.z_t.attrs = {}              
        ds0 = ds0.set_coords('z_t')
    
    if keepvars != None:
        ds0 = ds0[[keepvars]]

    if isel != None:
        ds0 = ds0.isel(isel)
    if sel != None:
        ds0 = ds0.sel(sel)

    if drop_cell == True:
        ds0 = ds0.drop_vars(['TLAT','TLONG','ULAT','ULONG'])

    if shift_time == True:
        ds0 = time_set_midmonth(ds0,'time')

    # Not all file time coordinates were saved in the start-of-next-month format.
    # This step checks if we need to shift, or if the time coordinate starts at 1. 
    if shift_time == 'auto':
        if ds0.time.dt.month.values[0] == 2:
            print('Shifting time coordinate.')
            ds0 = time_set_midmonth(ds0,'time')
        else:
            newtime = ds0.indexes['time'].shift(14, freq='D')
            ds0['time'] = newtime

    
    return ds0

def _preprocessor_day_1(ds0,res,keepvars,isel,sel,drop_cell,shift_time):
    """
    Preprocessor for xarray xr.open_dataset() function that adjusts the
    time coordinates of daily POP2 output from the first second of the 
    day following the averaging period to the first second of the day
    of the averaging period (e.g. output for mean 01 March 2002 is saved
    as 2002-03-02 00:00:00. The preprocessor sets this to 2002-03-01 
    00:00:00.) Since the daily output files contain overlaps at the edges
    of each time-period, the preprocess step also drops the duplicates.
    Has the optional input of selecting only desired variables.
    --- Input ---
        ds0: input dataset, this is a preprocess step so this is not an object
        keepvars: str or list of str, indicating desired variables. Defaults
                                      to None.
    --- Output ---
        None, internal to xr.open_dataset() function.
    """
    if shift_time == True:
        newtime = ds0.indexes['time'].shift(-1, freq='D')
        ds0['time'] = newtime
        
    if keepvars != None:
        ds0 = ds0[[keepvars]]

    if res == 'LR' and ds0.time.dt.year[0] not in [2000,2096]:
        ds0 = ds0.isel(time=slice(None,-31))

    if isel != None:
        ds0 = ds0.isel(isel)
    if sel != None:
        ds0 = ds0.sel(sel)

    if drop_cell == True:
        ds0 = ds0.drop_vars(['TLAT','TLONG','ULAT','ULONG'])
    
    return ds0

"""
===============================================================================================
===============================================================================================
====================================== get_CESM_variable ======================================
===============================================================================================
===============================================================================================
"""

BRCPs = ['BRCP26','BRCP45','BRCP60','BRCP85']

def get_CESM_variable(resolution,variable,start,end,scenario,ensemble,component,temporal,
                      isel=None,sel=None,keepvars=None,drop_cell=False,shift_time='auto',
                      chunks='auto',parallel=False,out_list=False):
    """
    Calls either get_CESMLR_variable or get_CESMHR_variable functions depending on resolution
    selection. Passes all arguments and returns the desired dataset or filepaths list.
    --- Input ---
    resolution: str, indicating desired resolution (LR or HR)
    variable : str, indicating the desired variable (e.g. 'VVEL')
    start : str, indicating the start year (e.g. '1979')
    end : str, indicating the end year (e.g. '2017')
    scenario : str, indicating the desired BRCP, if analyzing years past 2005 
                    (e.g. BRCP85) ; there is only BRCP85 for CESM-LR
    ensemble : str, indicating desired ensemble member; input as a str of
                    length 3 (e.g. '002' for ensemble #002) or as 'all'.
    component : str, indicating desired model component; input CESM abbreviation
                    (e.g. 'ocn')
    temporal : str, indicating temporal resolution ('day_1' or 'month_1')
    isel : dict, indicating dimension and indices to subset
    sel : dict, indicating dimension and coordinate tick labels to subset
    drop_cell : bool, indicating whether or not to drop grid cell coordinate variables
    shift_time : bool or str, indicating how to shift the time coordinate labels
                    depending on the cell_method used to save each dataset (VERY IMPORTANT)
    chunks : dict or str, indicating desired chunks for lazy loading (passes xarray.open_dataset()
                    chunk argument)
    parallel : bool, indicates whether or not to load data lazily with dask
--- Output ---
    result : xarray Dataset object or list of str, if path_list is True, then return just the 
                    filepaths, otherwise return an xarray Dataset object
    """
    if resolution not in ['LR','HR']:
        raise Exception('Please input one of "LR" or "HR" for your desired simulation.')    

    if resolution == 'LR':
        result = get_CESMLR_variable(variable,start,end,scenario,ensemble,component,temporal,
                                 isel,sel,keepvars,drop_cell,shift_time,chunks,parallel,out_list)

    elif resolution == 'HR':
        result = get_CESMHR_variable(variable,start,end,scenario,ensemble,component,temporal,
                                 isel,sel,keepvars,drop_cell,shift_time,chunks,parallel,out_list)

    if out_list == True:
        # result is a list of paths to the datasets
        return result
    else:
        # result is the xarray Dataset
        return result.sel(time=slice(str(start).zfill(4),str(end).zfill(4)))

"""
===============================================================================================
===============================================================================================
====================================== CESM-HR functions ======================================
===============================================================================================
===============================================================================================
"""

def get_CESMHR_base_directory(scenario):
    """
    Returns base directory of CESM-HR MESACLIP datasets as a string.
    CESM-HR MESACLIP was run with the full CMIP Deck Set (Chang et al. 2020,
    2023, 2025)
    --- Input ---
        scenario : str, indicating desired simulation time scenario; options
                      'PIcntl', 'BHIST', 'BRCP26', 'BRCP45', 'BRCP60', or 'BRCP85'
    --- Output ---
        bdir : str, defining the base directory name
    """
    if scenario not in ['PIcntl','BHIST','BRCP26','BRCP45','BRCP60','BRCP85']:
        raise Exception('Please input one of "PIcntl", "BHIST", "BRCP26", "BRCP45", "BRCP60", or "BRCP85",for your desired scenario.')
        
    if scenario == 'PIcntl':
        bdir = '/glade/campaign/collections/rda/data/d651029/'
    elif scenario == 'BHIST':
        bdir = '/glade/campaign/collections/rda/data/d651007'
    elif scenario == 'BRCP60':
        bdir = '/glade/campaign/collections/rda/data/d651008/'
    elif scenario == 'BRCP85':
        bdir = '/glade/campaign/collections/rda/data/d651009/'
    # elif 'XXX'
    # include other scenarios in future version

    return bdir

def get_CESMHR_ensemble_directory_names(scenario):
    """
    Returns simulation names of CESM-HR MESACLIP datasets as a string or
    list of strings.
    --- Input ---
        scenario : str, indicating desired simulation scenario; options
                      'PIcntl', 'BHIST', 'BRCP85', etc.
    --- Output ---
        nlist : list of str, indicating all simulation names for the selected
                             scenario
    """
    if scenario not in ['PIcntl','BHIST','BRCP60','BRCP85']:
        raise Exception('Please input one of "PIcntl","BHIST", or "BRCP85" for your desired scenario.')
        
    if scenario == 'PIcntl':
        nlist = ['B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02']
        
    elif scenario == 'BHIST':
        nlist = ['b.e13.BHISTC5.ne120_t12.cesm-ihesp-sehires38-1850-2005.001',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2005.002',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2005.003',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.44-1920-2005.004',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.44-1920-2005.005',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.45-1920-2005.006',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.007',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.008',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.009',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.010']
        
    elif scenario == 'BRCP60':
        nlist = ['b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.001',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.002',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.003',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.004',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.005',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.006',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.007',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.008',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.009',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.010']

    elif scenario == 'BRCP85':
        nlist = ['b.e13.BRCP85C5.ne120_t12.cesm-ihesp-sehires38-2006-2100.001',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30-2006-2100.002',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31-2006-2100.003',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.004',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.005',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.006',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.007',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.008',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.009',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.010']
        
    return nlist

def get_CESMHR_ensemble_directory(scenario,ensemble='all',component='ocn',temporal='month_1'):
    """
    Returns ensemble directory for a specified component of CESM-HR MESACLIP 
    datasets as a string or list of strings.
    --- Input ---
        scenario : str, indicating desired simulation scenario; options
                      'PIcntl', 'BHIST', 'BRCP85', etc.
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
    --- Output ---
        dir_list : list of str, containing paths to desired ensemble datasets
    """
    if scenario not in ['PIcntl','BHIST','BRCP85']:
        raise Exception('Keyword error, scenario: Please input one of "PIcntl","BHIST", or "BRCP85".')
    ens_num = [str(i).zfill(3) for i in range(1,11)]
    ens_num.append('all')
    if scenario != 'PIcntl' and ensemble not in ens_num:
        raise Exception('Keyword error, ensemble: Please select either "all" ensembles or in the format "00X".')
    if component not in ['atm','ice','lnd','ocn','rof']:
        raise Exception('Keyword error, component: Please input on of "atm", "ice", "lnd", "ocn", or "rof".')
    if temporal not in ['day_1','month_1']:
        raise Exception('Keyword error, temporal: Please input either "day_1" or "month_1".')
    if component in ['ice'] and temporal == 'day_1':
        raise Exception('CESM CICE output (component = "ice") is only available in monthly means (temporal = "month_1").') 
        
    bdir = get_CESMHR_base_directory(scenario)
    cdir = os.path.join(component,'proc/tseries/',temporal,'')
    
    ens_list = get_CESMHR_ensemble_directory_names(scenario)
    
    if ensemble != 'all':
        enum = int(ensemble) - 1
        ens_list = [ens_list[enum]]
    dir_list = [os.path.join(bdir,ens,cdir) for ens in ens_list]

    return dir_list

def get_CESMHR_filename_base(scenario,ensemble='all',component='ocn',temporal='month_1'):
    """
    Returns list of filename bases for the desired scenario (if utilizing BRCPs), 
    the ensemble members, the component, and the temporal frequency.
    --- Input ---
        scenario : str, indicating the desired BRCP, if analyzing years past 2005 
                        (e.g. BRCP85) ; there is only BRCP85 for CESM-LR
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
                        (e.g. 'ocn')
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
    --- Output ---
        name_list : list of str, containing filename bases of desired variables
    """
    if scenario == 'PIcntl':
        blist = ['B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.']
        
    elif scenario == 'BHIST':
        blist = ['b.e13.BHISTC5.ne120_t12.cesm-ihesp-sehires38-1850-2005.001.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2005.002.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2005.003.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.44-1920-2005.004.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.44-1920-2005.005.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.45-1920-2005.006.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.007.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.008.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.009.',
                'b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.010.']

    elif scenario == 'BRCP60':
        blist = ['b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.001.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.002.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.003.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.004.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.005.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.006.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.007.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.008.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.009.',
                 'b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.010.']
        
    elif scenario == 'BRCP85':
        blist = ['b.e13.BRCP85C5.ne120_t12.cesm-ihesp-sehires38-2006-2100.001.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30-2006-2100.002.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31-2006-2100.003.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.004.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.005.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.006.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.007.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.008.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.009.',
                'b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.010.']

    if ensemble != 'all' and scenario != 'PIcntl':
        enum = int(ensemble) - 1
        blist = [blist[enum]]

    if (component == 'ocn') and (temporal == 'day_1'):
        comp_name = 'pop.h.nday1.'
    elif (component == 'ocn') and (temporal == 'month_1'):
        comp_name = 'pop.h.'
    elif (component == 'atm') and (temporal == 'day_1'):
        comp_name = 'cam.h1.'
    elif (component == 'atm') and (temporal == 'month_1'):
        comp_name = 'cam.h0.'
    elif component == 'ice':
        comp_name = 'cice.h.'
    ## include other components in future

    blist0 = [''.join([b,comp_name]) for b in blist]
    
    return blist0

def get_CESMHR_variable(variable,start,end,scenario,ensemble,component,temporal,
                        isel=None,sel=None,keepvars=None,drop_cell=False,shift_time=True,
                        chunks='auto',parallel=False,out_list=False):
    """
    Returns the xarray Dataset or a list of paths for the specified variable.
    --- Input ---
        variable : str, indicating the desired variable (e.g. 'VVEL')
        start : str, indicating the start year (e.g. '1979')
        end : str, indicating the end year (e.g. '2017')
        scenario : str, indicating the desired BRCP, if analyzing years past 2005 
                        (e.g. BRCP85) ; there is only BRCP85 for CESM-LR
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
                        (e.g. 'ocn')
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
        isel : dict, indicating dimension and indices to subset
        sel : dict, indicating dimension and coordinate tick labels to subset
        drop_cell : bool, indicating whether or not to drop grid cell coordinate variables
        shift_time : bool or str, indicating how to shift the time coordinate labels
                        depending on the cell_method used to save each dataset (VERY IMPORTANT)
        chunks : dict or str, indicating desired chunks for lazy loading (passes xarray.open_dataset()
                        chunk argument)
        parallel : bool, indicates whether or not to load data lazily with dask
--- Output ---
        result : xarray Dataset object or list of str, if path_list is True, then return just the 
                        filepaths, otherwise return an xarray Dataset object
    """
    start = str(start).zfill(4) ; end = str(end).zfill(4)
    if ensemble == 'all':
        nens = 10
    else:
        nens = 1
        
    if (scenario == 'PIcntl') and (start > '0519' or end > '0519'):
        raise Exception('Time selection error: PIcntl runs from year 0001 to 0500. Please select a start and end time within those bounds.')
    elif (scenario == 'BHIST'):
        if (ensemble != '001') and ((start < '1920' or start > '2005') or (end < '1920' or end > '2005')):
            raise Exception('Time selection error: Aside from ensemble member #001, BHIST runs from year 1920 to 2005. Please select a start and end time within those bounds.')
        elif (ensemble == '001') and ((start < '1850' or start > '2005') or (end < '1920' or end > '2005')):
            raise Exception('Time selection error: Ensemble member #001 BHIST runs from year 1850 to 2005. Please select a start and end time within those bounds.')
    elif (scenario in BRCPs) and (end < '2006' or end > '2100'):
        raise Exception('Time selection error: BRCPs run from 2006 to 2100. Please select at least and end time within those bounds. Note: If you choose a start time before 2006, BHIST will be used for those dates.')
    elif (scenario == 'BRCP85') and (ensemble == 'all' or '009') and (component == 'atm') and (temporal == 'day_1'):
        warnings.warn('There are no daily output atmosphere files for ensemble member 009 on the glade file system. There are 6-hour mean files (from which daily can be derived), but those files are not compiled by this subroutine as of yet.', UserWarning)

    if scenario != 'PIcntl':
        
        bhist_dir_list = get_CESMHR_ensemble_directory('BHIST',ensemble,component,temporal)
        bhist_base_list = get_CESMHR_filename_base('BHIST',ensemble,component,temporal)
        bhist_base_list = [x + y for x,y in zip(bhist_dir_list,bhist_base_list)]
        
        brcp_dir_list = get_CESMHR_ensemble_directory(scenario,ensemble,component,temporal)
        brcp_base_list = get_CESMHR_filename_base(scenario,ensemble,component,temporal)
        brcp_base_list = [x + y for x,y in zip(brcp_dir_list,brcp_base_list)]

        bhist_fpaths = [get_CESM_filepath(start,end,variable,x) for x in bhist_base_list]
        brcp_fpaths = [get_CESM_filepath(start,end,variable,x) for x in brcp_base_list]

        if scenario == 'BHIST':
            path_list = bhist_fpaths
        elif (scenario in BRCPs) and (int(start) < 2006):
            path_list = [np.concatenate((x,y)) for x,y in zip(bhist_fpaths,brcp_fpaths)]
        else:
            path_list = brcp_fpaths

    elif scenario == 'PIcntl':
    
        dir_list = get_CESMHR_ensemble_directory(scenario,ensemble,component,temporal)
        base_list = get_CESMHR_filename_base(scenario,ensemble,component,temporal)
        base_list = [x + y for x,y in zip(dir_list,base_list)]
        path_list = [get_CESM_filepath(start,end,variable,x) for x in base_list]

    if out_list == True:
        result = path_list
    else:
        result = get_xarray_dataset(path_list,variable,start,end,scenario,ensemble,component,temporal,
                                res='HR',isel=isel,sel=sel,keepvars=keepvars,drop_cell=drop_cell,
                                shift_time=shift_time,chunks=chunks,parallel=parallel)

    return result

"""
===============================================================================================
===============================================================================================
====================================== CESM-LR functions ======================================
===============================================================================================
===============================================================================================
"""

def get_CESMLR_base_directory(scenario):
    """
    Returns base directory of CESM-LR MESACLIP datasets as a string.
    CESM-LR PIcntl, BHIST, and BRCP8.5 have the same base directory,
    and the scenario is a subdirectory
    --- Input ---
        scenario : str, indicating desired simulation scenario; options
                        are 'PIcntl', 'BHIST', or 'BRCP85'.
    --- Output ---
        bdir : str, defining the base directory name
    """
    if scenario not in ['PIcntl','BHIST','BRCP85']:
        raise Exception('Please input one of "PIcntl","BHIST", or "BRCP85" for your desired scenario.')
        
    bdir = '/glade/campaign/collections/rda/data/d651030/'
    bdir = os.path.join(bdir,scenario)
    
    return bdir

def get_CESMLR_ensemble_directory_names(scenario):
    """
    Returns simulation names of CESM-LR MESACLIP datasets as a string or
    list of strings.
    --- Input ---
        scenario : str, indicating desired simulation scenario; options
                      'PIcntl', 'BHIST', 'BRCP85', etc.
    --- Output ---
        nlist : list of str, indicating all simulation names for the selected
                             scenario
    """
    if scenario not in ['PIcntl','BHIST','BRCP85']:
        raise Exception('Please input one of "PIcntl","BHIST", or "BRCP85" for your desired scenario.')
        
    if scenario == 'PIcntl':
        nlist = ['B.E.13.B1850C5.ne30_g16.sehires38.003.sunway']
    elif scenario == 'BHIST':
        nlist = ['b.e13.BHISTC5.ne30_g16.cesm-ihesp-sehires38-1850-2005.001',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.002',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.003',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.004',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.005',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.006',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.007',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.008',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.009',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.010']
    elif scenario == 'BRCP85':
        nlist = ['b.e13.BRCP85C5.ne30_g16.cesm-ihesp-sehires38-2006-2100.001',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.002',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.003',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.004',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.005',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.006',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.007',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.008',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.009',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.010'] 

    return nlist

def get_CESMLR_ensemble_directory(scenario,ensemble='all',component='ocn',temporal='month_1'):
    """
    Returns ensemble directory for a specified component of CESM-LR MESACLIP 
    datasets as a string or list of strings.
    --- Input ---
        scenario : str, indicating desired simulation scenario; options
                      'PIcntl', 'BHIST', 'BRCP85', etc.
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
    --- Output ---
        dir_list : list of str, containing paths to desired ensemble datasets
    """
    if scenario not in ['PIcntl','BHIST','BRCP85']:
        raise Exception('Keyword error, scenario: Please input one of "PIcntl","BHIST", or "BRCP85".')
    ens_num = [str(i).zfill(3) for i in range(1,11)]
    ens_num.append('all')
    if ensemble not in ens_num:
        raise Exception('Keyword error, ensemble: Please select either "all" ensembles or in the format "00X".')
    if component not in ['atm','ice','lnd','ocn','rof']:
        raise Exception('Keyword error, component: Please input on of "atm", "ice", "lnd", "ocn", or "rof".')
    if temporal not in ['day_1','month_1']:
        raise Exception('Keyword error, temporal: Please input either "day_1" or "month_1".')
    
    bdir = get_CESMLR_base_directory(scenario)
    cdir = os.path.join(component,'proc/tseries/',temporal,'')
    
    ens_list = get_CESMLR_ensemble_directory_names(scenario)
    
    if ensemble != 'all' and scenario != 'PIcntl':
        enum = int(ensemble) - 1
        ens_list = [ens_list[enum]]
    dir_list = [os.path.join(bdir,ens,cdir) for ens in ens_list]

    return dir_list

def get_CESMLR_filename_base(scenario,ensemble='all',component='ocn',temporal='month_1'):
    """
    Returns list of filename bases for the desired scenario (if utilizing BRCPs), 
    the ensemble members, the component, and the temporal frequency.
    --- Input ---
        scenario : str, indicating the desired BRCP, if analyzing years past 2005 
                        (e.g. BRCP85) ; there is only BRCP85 for CESM-LR
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
                        (e.g. 'ocn')
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
    --- Output ---
        name_list : list of str, containing filename bases of desired variables
    """
    if scenario == 'PIcntl':
        blist = ['B.E.13.B1850C5.ne30g16.sehires38.003.sunway.']
        
    elif scenario == 'BHIST':
        blist = ['b.e13.BHISTC5.ne30_g16.cesm-ihesp-sehires38-1850-2005.001.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.002.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2100.003.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2100.004.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2100.005.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2100.006.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2100.007.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2100.008.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.009.',
                'b.e13.BHISTC5.ne30_g16.cesm-ihesp-hires1.0.42-1920-2005.010.']
    elif scenario == 'BRCP85':
        blist = ['b.e13.BRCP85C5.ne30_g16.cesm-ihesp-sehires38-2006-2100.001.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.002.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.003.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.004.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.005.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.006.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.007.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.008.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.009.',
                'b.e13.BRCP85C5.ne30_g16.cesm-ihesp-hires1.0.42-2006-2100.010.']

    if ensemble != 'all' and scenario != 'PIcntl':
        enum = int(ensemble) - 1
        blist = [blist[enum]]

    if (component == 'ocn') and (temporal == 'day_1'):
        comp_name = 'pop.h.nday1.'
    elif (component == 'ocn') and (temporal == 'month_1'):
        comp_name = 'pop.h.'
    elif (component == 'atm') and (temporal == 'day_1'):
        comp_name = 'cam.h1.'
    elif (component == 'atm') and (temporal == 'month_1'):
        comp_name = 'cam.h0.'
    ## include other components in future

    blist0 = [''.join([b,comp_name]) for b in blist]
    
    return blist0

def get_CESMLR_variable(variable,start,end,scenario,ensemble,component,temporal,
                        isel=None,sel=None,keepvars=None,drop_cell=False,shift_time=True,
                        chunks='auto',parallel=False,out_list=False):
    """
    Returns the xarray Dataset or a list of paths for the specified variable.
    --- Input ---
        resolution: str, indicating desired resolution (LR or HR)
        variable : str, indicating the desired variable (e.g. 'VVEL')
        start : str, indicating the start year (e.g. '1979')
        end : str, indicating the end year (e.g. '2017')
        scenario : str, indicating the desired BRCP, if analyzing years past 2005 
                        (e.g. BRCP85) ; there is only BRCP85 for CESM-LR
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
                        (e.g. 'ocn')
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
        isel : dict, indicating dimension and indices to subset
        sel : dict, indicating dimension and coordinate tick labels to subset
        drop_cell : bool, indicating whether or not to drop grid cell coordinate variables
        shift_time : bool or str, indicating how to shift the time coordinate labels
                        depending on the cell_method used to save each dataset (VERY IMPORTANT)
        chunks : dict or str, indicating desired chunks for lazy loading (passes xarray.open_dataset()
                        chunk argument)
        parallel : bool, indicates whether or not to load data lazily with dask
    --- Output ---
        result : xarray Dataset object or list of str, if path_list is True, then return just the 
                        filepaths, otherwise return an xarray Dataset object
    """
    start = str(start).zfill(4) ; end = str(end).zfill(4)
    if ensemble == 'all':
        nens = 10
    else:
        nens = 1
        
    if (scenario == 'PIcntl') and (start > '0500' or end > '0500'):
        raise Exception('Time selection error: PIcntl runs from year 0001 to 0500. Please select a start and end time within those bounds.')
    elif (scenario == 'BHIST'):
        if (ensemble != '001') and ((start < '1920' or start > '2005') or (end < '1920' or end > '2005')):
            raise Exception('Time selection error: Aside from ensemble member #001, BHIST runs from year 1920 to 2005. Please select a start and end time within those bounds.')
        elif (ensemble == '001') and ((start < '1850' or start > '2005') or (end < '1920' or end > '2005')):
            raise Exception('Time selection error: Ensemble member #001 BHIST runs from year 1850 to 2005. Please select a start and end time within those bounds.')
    elif (scenario == 'BRCP85' or scenario == 'BRCP60') and (end < '2006' or end > '2100'):
        raise Exception('Time selection error: BRCPs run from 2006 to 2100. Please select at least and end time within those bounds. Note: If you choose a start time before 2006, BHIST will be used for those dates.')
   
    if scenario != 'PIcntl':
        
        bhist_dir_list = get_CESMLR_ensemble_directory('BHIST',ensemble,component,temporal)
        bhist_base_list = get_CESMLR_filename_base('BHIST',ensemble,component,temporal)
        bhist_base_list = [x + y for x,y in zip(bhist_dir_list,bhist_base_list)]
        
        brcp_dir_list = get_CESMLR_ensemble_directory(scenario,ensemble,component,temporal)
        brcp_base_list = get_CESMLR_filename_base(scenario,ensemble,component,temporal)
        brcp_base_list = [x + y for x,y in zip(brcp_dir_list,brcp_base_list)]

        bhist_fpaths = [get_CESM_filepath(start,end,variable,x) for x in bhist_base_list]
        brcp_fpaths = [get_CESM_filepath(start,end,variable,x) for x in brcp_base_list]

        if scenario == 'BHIST':
            path_list = bhist_fpaths
        elif (scenario == 'BRCP85' or scenario == 'BRCP60') and (int(start) < 2006):
            path_list = [np.concatenate((x,y)) for x,y in zip(bhist_fpaths,brcp_fpaths)]
        else:
            path_list = brcp_fpaths

    elif scenario == 'PIcntl':
    
        dir_list = get_CESMLR_ensemble_directory(scenario,ensemble,component,temporal)
        base_list = get_CESMLR_filename_base(scenario,ensemble,component,temporal)
        base_list = [x + y for x,y in zip(dir_list,base_list)]
        path_list = [get_CESM_filepath(start,end,variable,x) for x in base_list]

    if out_list == True:
        result = path_list
    else:
        result = get_xarray_dataset(path_list,variable,start,end,scenario,ensemble,component,temporal,
                                res='LR',isel=isel,sel=sel,keepvars=keepvars,drop_cell=drop_cell,
                                shift_time=shift_time,chunks=chunks,parallel=parallel)

    return result

def get_CESM_filepath(start,end,variable,fbase):
    """
    Returns list of time strings for each filename base, constrained by 
    start and end times, the variable selected, and the file directory.
    --- Input ---
        variable : str, indicating the desired variable (e.g. 'VVEL')
        start : str, indicating the start year (e.g. '1979')
        end : str, indicating the end year (e.g. '2017')
        fdir : str, indicating file base directory
        fname_base : str, indicating filename base
    --- Output ---
        time_str : list of str, containing time strings to append to filename base
    """
    start = str(start).zfill(4) ; end = str(end).zfill(4)

    years = np.arange(int(start),int(end)+1,1)

    flist = np.sort(glob.glob(fbase+'{}.*.nc'.format(variable)))

    if len(flist) > 0:
        bin_edges = []
        time_str = []
        for f in flist:
            # split = f.split('.'+variable+'.')
            split = f.rsplit('.'+variable+'.',1)
            split0 = split[-1].split('-')[0].strip('.').strip('nc')
            bin_edges.append(int(split0[:4]))
        
        bin_edges = np.array(bin_edges)
        filled_bins_idx = np.unique(np.digitize(years, bin_edges)) - 1
        flist0 = flist[filled_bins_idx]
    
        return flist0

    else:
        raise Exception('Path error: The paths for {} could not be found. Please check your selected path variables.'.format(fbase+'{}.*.nc'.format(variable)))
        return

def get_xarray_dataset(path_list,variable,start,end,scenario,ensemble,component,temporal,
                        res,isel=None,sel=None,keepvars=None,drop_cell=False,shift_time=True,
                        chunks='auto',parallel=False):
    """
    Returns the xarray Dataset for the specified variable.
    --- Input ---
        resolution: str, indicating desired resolution (LR or HR)
        variable : str, indicating the desired variable (e.g. 'VVEL')
        start : str, indicating the start year (e.g. '1979')
        end : str, indicating the end year (e.g. '2017')
        scenario : str, indicating the desired BRCP, if analyzing years past 2005 
                        (e.g. BRCP85) ; there is only BRCP85 for CESM-LR
        ensemble : str, indicating desired ensemble member; input as a str of
                        length 3 (e.g. '002' for ensemble #002) or as 'all'.
        component : str, indicating desired model component; input CESM abbreviation
                        (e.g. 'ocn')
        temporal : str, indicating temporal resolution ('day_1' or 'month_1')
        isel : dict, indicating dimension and indices to subset
        sel : dict, indicating dimension and coordinate tick labels to subset
        drop_cell : bool, indicating whether or not to drop grid cell coordinate variables
        shift_time : bool or str, indicating how to shift the time coordinate labels
                        depending on the cell_method used to save each dataset (VERY IMPORTANT)
        chunks : dict or str, indicating desired chunks for lazy loading (passes xarray.open_dataset()
                        chunk argument)
        parallel : bool, indicates whether or not to load data lazily with dask
    --- Output ---
        result : xarray Dataset object, if path_list is True, then return just the 
                        filepaths, otherwise return an xarray Dataset object
    """
    if temporal == 'month_1':
        pp = partial(_preprocessor_month_1,isel=isel,sel=sel,keepvars=keepvars,drop_cell=drop_cell,shift_time=shift_time)
    else:
        pp = partial(_preprocessor_day_1,res=res,isel=isel,sel=sel,keepvars=keepvars,drop_cell=drop_cell,shift_time=shift_time)

    if scenario != 'PIcntl' and ensemble == 'all':
        ds_list = []
        for path in path_list:
            ds0 = xr.open_mfdataset(path,preprocess=pp,parallel=parallel,chunks=chunks,engine="h5netcdf",
                                   coords='minimal',compat='override',data_vars='minimal')
            ds_list.append(ds0)
        ds = xr.concat(ds_list,dim='ensemble')
    else:
        ds = xr.open_mfdataset(path_list[0],preprocess=pp,parallel=parallel,chunks=chunks,engine="h5netcdf",
                                   coords='minimal',compat='override',data_vars='minimal')
                              
    return ds
