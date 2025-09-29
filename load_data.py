from utilities import *

import pathlib

import xarray as xr
from rasterio.enums import Resampling

SCALING = 1/10

S2CCI_2016_PATH = {
    "path": pathlib.Path("data/s2cci_2016"),
    "files": sorted(list(pathlib.Path("data/s2cci_2016").glob("*h39v21*JD.tif")))
}
S2CCI_2019_PATH = {
    "path": pathlib.Path("data/s2cci_2019"),
    "files": sorted(list(pathlib.Path("data/s2cci_2019").glob("*h39v21*JD.tif")))
}
MODIS_BA_2016 = {
    "path": pathlib.Path("data/modis_ba_2016"),
    "files": list(pathlib.Path("data/modis_ba_2016").glob("MCD64A1.061_Burn_Date_doy*.tif"))
}
MODIS_BA_2019 = {
    "path": pathlib.Path("data/modis_ba_2019"),
    "files": list(pathlib.Path("data/modis_ba_2019").glob("MCD64A1.061_Burn_Date_doy*.tif"))
}
MODIS_CCI_2016 = {
    "path": pathlib.Path("data/modis_cci_2016"),
    "files": list(pathlib.Path("data/modis_cci_2016").glob("*AREA_5*JD.tif"))
}
MODIS_CCI_2019 = {
    "path": pathlib.Path("data/modis_cci_2019"),
    "files": list(pathlib.Path("data/modis_cci_2019").glob("*AREA_5*JD.tif"))
}


# Create empty template with desired shape and resolution
# from Sentinel-2 CCI data sample
s2_sample = S2CCI_2016_PATH['files'][0]
s2_da = xr.open_dataarray(s2_sample, engine='rasterio')
s2_small = centered_subset(s2_da, fraction=SCALING)
TEMPLATE = xr.zeros_like(s2_small.isel(band=0, drop=True))


def load_s2_cci(year: int) -> xr.Dataset:
    # Load all monthly files and combine to get annual burn date composite
    data_arrays = []
    if year == 2016:
        path_info = S2CCI_2016_PATH
    elif year == 2019:
        path_info = S2CCI_2019_PATH
    else:
        raise ValueError("Year must be 2016 or 2019")
    
    for file_path in path_info['files']:
        da = xr.open_dataarray(file_path, engine='rasterio')
        subset = centered_subset(da, fraction=1/10)
        if year == 2019:
            subset = subset.rio.reproject_match(TEMPLATE, resampling=Resampling.nearest)
        data_arrays.append(subset)

    # Stack all monthly data
    stacked = xr.concat(data_arrays, dim='month')

    # Create composite: take the maximum DOY value across all months
    # This gives you the latest burn date detected for each pixel
    annual_burn_composite = stacked.max(dim='month')

    # Alternative: Create a mask of burned vs unburned areas
    burned_mask = (stacked > 0).any(dim='month')

    # Convert to Dataset
    s2cci = xr.Dataset({
        'burn_date_doy': annual_burn_composite,
        'burned_mask': burned_mask
    })

    return s2cci


def load_modis_ba(year: int) -> xr.Dataset:
    data_arrays = []

    if year == 2016:    
        path_info = MODIS_BA_2016
    elif year == 2019:
        path_info = MODIS_BA_2019
    else:
        raise ValueError("Year must be 2016 or 2019")

    for file in path_info['files']:
        da = xr.open_dataarray(file, engine='rasterio')
        resampled = da.rio.reproject_match(TEMPLATE, resampling=Resampling.nearest)
        data_arrays.append(resampled)

    stacked = xr.concat(data_arrays, dim='month')
    annual_burn_composite = stacked.max(dim='month')
    burned_mask = (stacked > 0).any(dim='month')

    modis_ba = xr.Dataset({
        'burn_date_doy': annual_burn_composite,
        'burned_mask': burned_mask
    })

    return modis_ba

def load_modis_cci(year: int) -> xr.Dataset:
    data_arrays = []

    if year == 2016:    
        path_info = MODIS_CCI_2016
    elif year == 2019:
        path_info = MODIS_CCI_2019
    else:
        raise ValueError("Year must be 2016 or 2019")

    data_arrays = []

    for file in path_info['files']:
        da = xr.open_dataarray(file, engine='rasterio')
        resampled = da.rio.reproject_match(TEMPLATE, resampling=Resampling.nearest)
        data_arrays.append(resampled)

    stacked = xr.concat(data_arrays, dim='month')
    annual_burn_composite = stacked.max(dim='month')
    burned_mask = (stacked > 0).any(dim='month')

    modis_cci = xr.Dataset({
        'burn_date_doy': annual_burn_composite,
        'burned_mask': burned_mask
    })

    return modis_cci
