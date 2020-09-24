import os
import warnings
from functools import singledispatch
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import fiona
import numba
import numpy as np
from rasterio.crs import CRS

from geosardine._utility import calc_extent, calc_distance
from geosardine.interpolate._utility import InterpolationResults


@numba.njit(parallel=True)
def _idw(
    known_coordinates: np.ndarray,
    value: np.ndarray,
    unknown_coordinates: np.ndarray,
    distance_function: Callable,
    power: Union[float, int] = 2,
) -> np.ndarray:
    interpolated = np.zeros(unknown_coordinates.shape[0])
    for i in numba.prange(interpolated.shape[0]):
        distances = np.array(
            [
                distance_function(known_coordinates[j], unknown_coordinates[i])
                for j in range(known_coordinates.shape[0])
            ]
        )
        distances[distances == 0] = np.nan
        mask = ~np.isnan(distances)
        weight = distances[mask] ** -power
        interpolated[i] = (weight * value[mask]).sum() / weight.sum()
    return interpolated


@singledispatch
def idw(
    points: Union[str, np.ndarray],
    value: np.ndarray,
    spatial_res: Tuple[float, float],
    epsg: int = 4326,
    column_name: Optional[str] = None,
    longlat_distance: str = "harvesine",
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Union[float, int] = 2,
) -> Optional[InterpolationResults]:
    """
    create interpolated raster from point by using Inverse Distance Weighting (Shepard)

    Parameters
    ----------
    points : numpy array, str.
        list of points coordinate as numpy array or address of vector file
        i.e shapefile or geojson
        * if numpy array, then value input needed
        * if str, then value is not needed. value will be created from file
    value : numpy array
        list of points value as numpy array, not needed if vector file used as input
    spatial_res : tuple or list of float
        spatial resolution in x and y axis
    column_name : str, default None
        column name needed to obtain value from attribute data of vector file
        * If str, value will be read from respective column name
        * If None, first column will be used as value
    epsg : int, default 4326
        EPSG code of reference system
        * If 4326, WGS 1984 geographic system
        * If int, epsg will be parsed
    longlat_distance: str harvesine or vincenty, default harvesine
        method used to calculate distance in spherical / ellipsoidal
        * If harvesine, calculation will be faster but less accurate
        * If vincenty, calculation will be slower but more accurate
    extent: tuple of float, default None
        how wide the raster will be
        * If None, extent will be calculated from points input
        * If tuple of float, user input of extent will be used
    power: float, default 2
        how smooth the interpolation will be

    Returns
    -------
    InterpolationResults

    """
    print("only support numpy array or vector file such as shapefile and geojson")
    return None


@idw.register
def _idw_array(
    points: np.ndarray,
    value: np.ndarray,
    spatial_res: Tuple[float, float],
    epsg: int = 4326,
    longlat_distance: str = "harvesine",
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Union[float, int] = 2,
    source: Optional[Union[str, Path]] = None,
) -> InterpolationResults:

    if extent is None:
        x_min, y_min, x_max, y_max = calc_extent(points)
        warnings.warn("using points' extent")
    else:
        x_min, y_min, x_max, y_max = extent

    crs = CRS.from_epsg(epsg)

    distance_calculation = longlat_distance
    if crs.is_projected:
        distance_calculation = "projected"

    x_res, y_res = spatial_res
    rows = int((y_max - y_min) / y_res)
    columns = int((x_max - x_min) / x_res)

    x_dist, y_dist = np.meshgrid(
        np.arange(columns, dtype=np.float32), np.arange(rows, dtype=np.float32)
    )

    x_dist *= x_res
    x_dist += x_min

    y_dist *= -y_res
    y_dist += y_max
    interpolated_coordinate = np.stack([x_dist, y_dist], axis=2)

    interpolated_value = _idw(
        points,
        value,
        interpolated_coordinate.reshape(rows * columns, 2),
        distance_function=calc_distance[distance_calculation],
        power=power,
    ).reshape(rows, columns)

    return InterpolationResults(
        interpolated_value,
        interpolated_coordinate,
        crs,
        (x_min, y_min, x_max, y_max),
        source=source,
    )


@idw.register
def _idw_file(
    file_name: str,
    spatial_res: Tuple[float, float],
    epsg: Optional[int] = None,
    column_name: Optional[str] = None,
    longlat_distance: str = "harvesine",
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Union[float, int] = 2,
) -> InterpolationResults:
    if os.path.exists(file_name):
        with fiona.open(file_name) as file:
            if "Point" in file.schema["geometry"]:
                if column_name is None:
                    column_name = list(file.schema["properties"].items())[0][0]
                    warnings.warn(f"using first column: {column_name} as data input")
                epsg = int(file.crs["init"][5:])
                points: List[List[float]] = []
                value: List[float] = []
                for layer in file:
                    points.append(layer["geometry"]["coordinates"])
                    value.append(float(layer["properties"][column_name]))

        return _idw_array(
            np.array(points),
            np.array(value),
            spatial_res,
            epsg,
            longlat_distance,
            extent,
            power,
            source=file_name,
        )


@idw.register
def _idw_file_path(
    file_name: Path,
    spatial_res: Tuple[float, float],
    epsg: Optional[int] = None,
    column_name: Optional[str] = None,
    longlat_distance: str = "harvesine",
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Union[float, int] = 2,
):
    return _idw_file(
        file_name, spatial_res, epsg, column_name, longlat_distance, extent, power
    )
