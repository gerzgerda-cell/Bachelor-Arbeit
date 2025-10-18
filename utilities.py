import numpy as np
import xarray as xr
import geopandas as gpd
import re
from shapely.geometry import box
import rasterio
from sklearn.neighbors import KDTree
from collections import deque


def resample_dataset(ds, target_res_m=20):
    """Partly developed with ChatGPT."""
    # Check if CRS is available
    if not hasattr(ds.rio, 'crs') or ds.rio.crs is None:
        raise ValueError("Dataset has no CRS information")
    
    # Get current resolution and CRS
    current_res = ds.rio.resolution()
    crs = ds.rio.crs
    bounds = ds.rio.bounds()
    x_min, y_min, x_max, y_max = bounds
    
    
    # Check if CRS is in degrees (geographic) or meters (projected)
    is_geographic = crs.is_geographic
    
    if is_geographic:
        # For geographic coordinates, convert meters to degrees
        # Approximate conversion: 1 degree ≈ 111,000 meters at equator
        target_res_deg = target_res_m / 111000
        
        # Create new coordinate arrays at target resolution in degrees
        new_x = np.arange(x_min, x_max, target_res_deg)
        new_y = np.arange(y_max, y_min, -target_res_deg)
    else:
        # For projected coordinates, use meters directly
        
        new_x = np.arange(x_min, x_max, target_res_m)
        new_y = np.arange(y_max, y_min, -target_res_m)
    
    # Interpolate to new resolution using nearest neighbor
    resampled_ds = ds.interp(
        x=new_x,
        y=new_y,
        method='nearest'
    )
    
    return resampled_ds



def centered_subset(ds, dims=("y", "x"), fraction=1/1000):
    subset = ds
    ndims = len(dims)

    frac_per_dim = fraction ** (1 / ndims)

    for dim in dims:
        n = ds.sizes[dim]
        center = n // 2
        half_width = max(1, int(n * frac_per_dim / 2))
        subset = subset.isel({dim: slice(center - half_width, center + half_width)})

    return subset


def crop_da_to_bbox(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """Partly developed with ChatGPT.
    Fast bbox crop of an xarray.DataArray to a bounding box (EPSG:4326).
    No CRS conversion, no polygon masking—just a rectangular cut.
    """
    # bbox = (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bbox

    # ensure coords are ascending so slice works as expected
    if da["x"][0] > da["x"][-1]:
        da = da.sortby("x")
    if da["y"][0] > da["y"][-1]:
        da = da.sortby("y")

    # rectangular selection
    return da.sel({"x": slice(minx, maxx), "y": slice(maxy, miny)})

def crop_da_to_gdf_bbox(
    da: xr.DataArray,
    gdf: gpd.GeoDataFrame,
) -> xr.DataArray:
    """
    Fast bbox crop of an xarray.DataArray to a GeoDataFrame's extent (EPSG:4326).
    No CRS conversion, no polygon masking—just a rectangular cut.
    """
    # bbox = (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = gdf.total_bounds

    # ensure coords are ascending so slice works as expected
    if da["x"][0] > da["x"][-1]:
        da = da.sortby("x")
    if da["y"][0] > da["y"][-1]:
        da = da.sortby("y")

    # rectangular selection
    cropped = da.sel({"x": slice(minx, maxx), "y": slice(miny, maxy)})

    # mask out everything outside the gdf geometry (make them NaN)
    geoms = gdf.geometry.values
    return cropped.rio.clip(geoms, gdf.crs, drop=False)


def filter_files_by_aoi(file_list, aoi_gdf):
    """
    Filter Sentinel-2 CCI files based on spatial intersection with AOI.
    Assumes files follow naming convention with tile information.
    """
    relevant_files = []
    
    for file_path in file_list:
        try:
            # Open the file to get its bounds
            with rasterio.open(file_path) as src:
                # Get the bounds of the raster
                bounds = src.bounds
                # Create a bounding box geometry
                file_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                
                # Check if the file's bounding box intersects with any geometry in the AOI
                if aoi_gdf.geometry.intersects(file_bbox).any():
                    relevant_files.append(file_path)                    
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    return relevant_files


def cluster_by_area_and_time(data: xr.Dataset, max_area: float = 3, max_time: int = 3) -> xr.DataArray:
    """Partly developed with ChatGPT."""

    def day_diff(a, b, days_in_year=366):
        d = abs(int(a) - int(b))
        return min(d, days_in_year - d)


    doy_da   = data['burn_date_doy'].squeeze('band')
    mask_da  = data['burned_mask'].squeeze('band')

    # Valid burned pixels with a proper DOY
    valid = (mask_da == True) & (doy_da >= 1) & (doy_da <= 366)

    # Get pixel indices of valid points (row=y_index, col=x_index)
    iy, ix = np.where(valid.values)   # arrays of length N (N = #burned pixels)

    # Build X from pixel indices => distances are in pixels
    X = np.column_stack([ix.astype(float), iy.astype(float)])  # shape (N, 2)

    # Extract matching DOYs as int array
    T = doy_da.values[iy, ix].astype(int)                      # shape (N,)

    tree = KDTree(X)
    nbrs = tree.query_radius(X, r=max_area, return_distance=False)  # 3 pixels

    adj = [[] for _ in range(len(X))]
    for i, neigh in enumerate(nbrs):
        ti = T[i]
        for j in neigh:
            if i == j: 
                continue
            if day_diff(ti, T[j]) <= max_time:   # ≤3 days
                adj[i].append(j)

    # Connected components (BFS)
    labels = np.full(len(X), -1, dtype=int)
    cid = 0
    for i in range(len(X)):
        if labels[i] != -1:
            continue
        labels[i] = cid
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = cid
                    q.append(v)
        cid += 1

    H, W = doy_da.shape

    # initialise with -1 (meaning "no burn" / background)
    labels_full = np.full((H, W), -1, dtype=int)

    # fill in the burned pixels with their cluster ids
    labels_full[iy, ix] = labels

    return xr.DataArray(
        labels_full,
        coords={"y": doy_da.y, "x": doy_da.x},
        dims=("y", "x"),
        name="fire_labels"
    )
