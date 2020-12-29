"""
Spatial operations extend fiona and rasterio.
Collection of spatial operation which i occasionally use written in python:
 - Interpolation with IDW (Inverse Distance Weighting) Shepard
 - Drape vector to raster
 - Spatial join between two vector
 - Raster wrapper, for better experience. ie: math operation between two raster, resize and resample
"""
from . import interpolate
from ._geosardine import (
    drape2raster,
    drape_geojson,
    drape_shapely,
    rowcol2xy,
    spatial_join,
    xy2rowcol,
)
from ._utility import harvesine_distance, vincenty_distance
from .raster import Raster

__all__ = [
    "rowcol2xy",
    "xy2rowcol",
    "drape2raster",
    "spatial_join",
    "drape_shapely",
    "drape_geojson",
    "interpolate",
    "harvesine_distance",
    "vincenty_distance",
    "Raster",
]

__version__ = "0.10.2"
__author__ = "Sahit Tuntas Sadono"
