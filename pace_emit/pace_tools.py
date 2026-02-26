"""
This file contains functions for working with PACE data, including opening, gridding,
    etc. Xarray and rasterio are heavily used to process PACE data. 

Author: Skye Caplan (NASA, SSAI)
Last updated: 02/26/2026

TODO: 
- Generalize the grid_data fcn to work for EMIT
"""
# Packages
import xarray as xr 
import numpy as np 
import cartopy
import cartopy.crs as ccrs
import cf_xarray  # noqa: F401
import matplotlib.pyplot as plt
import rasterio
import rioxarray as rio
from rasterio.enums import Resampling
from rasterio.crs import CRS

def open_l2(fpath):
    """ 
    Opens a PACE L2 file as an xarray dataset, assigning lat/lons (and wavelength, 
        if the dataset is 3D) as coordinates.
    Args:
        fpath - path to a L2 PACE file 
    Returns:
        ds - xarray dataset
    """
    dt = xr.open_datatree(fpath, decode_timedelta=False)
    try:
        ds = xr.merge((
            dt.ds,
            dt["geophysical_data"].to_dataset(),
            dt["sensor_band_parameters"].coords,
            dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
            )
        )
        ds = ds.set_xindex(("latitude", "longitude"), xr.indexes.NDPointIndex)
    except:
        ds = xr.merge((
            dt.ds,
            dt["geophysical_data"].to_dataset(),
            dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
            )
        )
        ds = ds.set_xindex(("latitude", "longitude"), xr.indexes.NDPointIndex)
    return ds

def mask_ds(ds, flag="CLDICE", reverse=False):
    """
    Mask a PACE OCI dataset for L2 flags. Default is to mask for clouds only
    Args:
        ds - xarray dataset containing "l2_flags" variable
        flag - str or list of l2 flag to mask for (see https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/)
        reverse - boolean or list of booleans to keep only pixels with the desired flag. 
                  Default is False. E.g., set True to use "LAND" flag to mask water pixels. 
    Returns:
        Masked dataset
    """
    # Make sure flags are recognized by the package
    if ds["l2_flags"].cf.is_flag_variable:
        # If multiple flags, make sure reverse is also a list and then iterate
        if type(flag)==list:
            if type(reverse)!=list:
                reverse = [reverse for i in range(len(flag))]
            for f,r in zip(flag, reverse):
                if r == False:
                    ds = ds.where(~(ds["l2_flags"].cf == f))
                else:
                    ds = ds.where(ds["l2_flags"].cf == f)
                print(f"{f} mask applied")
            return ds
        else:
            if type(reverse)==list:
                reverse = reverse[0]
            if reverse == False:
                ds = ds.where(~(ds["l2_flags"].cf == flag))
            else:
                ds = ds.where((ds["l2_flags"].cf == flag))
            print(f"{flag} mask applied")
            return ds
    else:
        print("l2_flags not recognized as flag variable")
        return ds

def grid_data(src, resolution, dst_crs="epsg:4326", src_crs="epsg:4326", resampling=Resampling.nearest):
    """
    Grid a PACE OCI L2 dataset. Makes sure 3D variables are in (Z, Y, X) 
        dimension order, and all variables have spatial dims/crs assigned.
    Args:
        src - an xarray dataset or dataarray to reproject
        resolution - resolution of the output grid, in dst_crs units
        dst_crs - CRS of the output data
        resampling - resampling method (see rasterio.enums)
    Returns:
        dst - projected xr dataset
    """
    # TODO: combine with regrid_data fcn - check for spatial dims, if none, 
    #   do the following lines, if found, skip
    if (len(list(src.dims)) == 3) and (list(src.dims)[0] != "wavelength_3d"):
        src = src.transpose("wavelength_3d", ...)
    src = src.rio.set_spatial_dims("pixels_per_line", "number_of_lines")
    src = src.rio.write_crs(src_crs)

    # Calculating the default affine transform
    defaults = rasterio.warp.calculate_default_transform(
        src.rio.crs,
        dst_crs,
        src.rio.width,
        src.rio.height,
        left=src.attrs["geospatial_lon_min"],#left=np.nanmin(src.longitude.data),
        bottom=src.attrs["geospatial_lat_min"],#bottom=np.nanmin(src.latitude.data),
        right=src.attrs["geospatial_lon_max"],#right=np.nanmax(src.longitude.data),
        top=src.attrs["geospatial_lat_max"],#top=np.nanmax(src.latitude.data),
    )
    
    # Aligning that transform to our desired resolution
    transform, width, height = rasterio.warp.aligned_target(*defaults, resolution)
    
    dst = src.rio.reproject(
        dst_crs=dst_crs,
        shape=(height, width),
        transform=transform,
        src_geoloc_array=(
            src["longitude"],
            src["latitude"],
        ),
        nodata=np.nan,
        resample=resampling,
    )
    dst["x"] = dst["x"].round(9)
    dst["y"] = dst["y"].round(9)
    
    return dst.rename({"x":"longitude", "y":"latitude"})

def regrid_data(src, resolution, dst_crs="epsg:4326", resampling=Resampling.nearest):
    """
    Reproject a dataset to match an input grid. Makes sure 3D variables are
        in (Z, Y, X) dimension order, and all variables have spatial dims/crs 
        assigned.
    Args:
        src - an xarray dataset or dataarray to reproject
        resolution - resolution of the output grid, in dst_crs units
        dst_crs - CRS of the output data
        resampling - resampling method (see rasterio.enums)
    Returns:
        dst - projected xr dataset
    """
    # TODO: Generalize
    if (len(list(src.dims)) == 3) and ((list(src.dims)[0] != "wavelengths") or (list(src.dims)[0] != "wavelength_3d")):
        try:
            src = src.transpose("wavelengths", ...)
        except:
            src = src.transpose("wavelength_3d", ...)

    # Aligning that transform to our desired resolution
    transform, width, height = rasterio.warp.aligned_target(transform=src.rio.transform(), 
                                                            width=src.rio.width, 
                                                            height=src.rio.height,
                                                            resolution=resolution)
    
    dst = src.rio.reproject(
        dst_crs=dst_crs,
        shape=(height, width),
        transform=transform,
        nodata=np.nan,
        resample=resampling,
    )
    dst["x"] = dst["x"].round(9)
    dst["y"] = dst["y"].round(9)
    
    return dst.rename({"x":"longitude", "y":"latitude"})