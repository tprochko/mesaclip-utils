# mesaclip-utils

A Python utility package for reading Community Earth System Model (CESM) output on NCAR's GLADE filesystem. This package focuses on easy-to-use, xarray and dask capable reading of netcdf4 datasets, specifically for data saved from the MESACLIP project (https://project.cgd.ucar.edu/projects/MESACLIP/). There are many subtle differences in different model component outputs or cases, so this package seemlessly handles these without the headache.

## Features
* **Automated I/O:** Automated path resolution for CESM history files on GLADE (slight bias to ocean model output, POP2).
* **Metadata Cleanup:** Ensures coordinate names and attributes are consistent across different model streams.
* **Simulation Case and Ensemble Concatenation:** Seemlessly concatenates historical (BHIST) and RCP (e.g. RCP8.5) datasets and their ensembles depending on user selection.
* **Xarray Integration:** Optimized for `open_mfdataset` with Dask for handling massive model outputs.

## Installation

Since this is a package to be used on GLADE/Casper/Derecho, it is recommended to install it in "editable" mode within your Conda environment.

1. Activate your environment (e.g., NPL or a custom clone):
   ```bash
   module load conda
   conda activate your-env-name
2. After cloning and activating this repo, run:
   ```bash
   cd /path/to/mesaclip-utils
   pip install -e . --user

## Quick start
  ```bash
  import mesaclip-utils as mu
  
  variable = 'TEMP'
  resolution = 'HR'
  scenario = 'BHIST'
  ensemble = 'all'
  component = 'ocn'
  temporal = 'month_1'
  
  years = np.arange(1920, 2005, 1)
  start = years[0]
  end = years[-1]
  
  chunks = {'time':1,'z_t': -1, 'nlat': 100, 'nlon': 100}
  
  ds = mu.get_CESM_variable(resolution, variable, start, end, scenario, ensemble, component, temporal,
                            drop_cell=True, chunks=chunks, parallel=True, shift_time='auto')
```

## Contact
Travis Prochko - trp2@tamu.edu | travpro18@outlook.com \
Department of Oceanography, Texas A&M University, College Station, TX

## Acknowledgements
Much inspiration from Stephen Yeager's (https://github.com/sgyeager) scripts and repositories.


