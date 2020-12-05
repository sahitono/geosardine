"""
Spatial operations extend fiona and rasterio
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

__version__ = "0.8.0"
__author__ = "Sahit Tuntas Sadono"
