import os
import warnings
from functools import singledispatch
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import fiona
import numba
import numpy as np
from rasterio.crs import CRS

from .._utility import calc_distance, calc_extent
from ._utility import InterpolationResult


@numba.njit(parallel=True)
def _idw(
    known_coordinates: np.ndarray,
    value: np.ndarray,
    unknown_coordinates: np.ndarray,
    distance_function: Callable,
    power: Union[float, int] = 2,
    distance_limit: float = 0.0,
) -> np.ndarray:
    interpolated = np.zeros(unknown_coordinates.shape[0])
    for i in numba.prange(interpolated.shape[0]):
        distances = np.array(
            [
                distance_function(known_coordinates[j], unknown_coordinates[i])
                for j in range(known_coordinates.shape[0])
            ]
        )
        distances[distances <= distance_limit] = np.nan
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
    distance_limit: float = 0.0,
) -> Optional[InterpolationResult]:
    """
    create interpolated raster from point by using Inverse Distance Weighting (Shepard)

    Parameters
    ----------
    points : numpy array, str
        list of points coordinate as numpy array or address of vector file  <br/>
        i.e shapefile or geojson  <br/>
        * if numpy array, then value input needed  <br/>
        * if str, then value is not needed instead will be created from file
    value : numpy array
        list of points value as numpy array, not needed if vector file used as input
    spatial_res : tuple or list of float
        spatial resolution in x and y axis
    column_name : str, default None
        column name needed to obtain value from attribute data of vector file  <br/>
        * If str, value will be read from respective column name  <br/>
        * If None, first column will be used as value
    epsg : int, default 4326
        EPSG code of reference system  <br/>
        * If 4326, WGS 1984 geographic system  <br/>
        * If int, epsg will be parsed
    longlat_distance: str harvesine or vincenty, default harvesine
        method used to calculate distance in spherical / ellipsoidal  <br/>
        * If harvesine, calculation will be faster but less accurate  <br/>
        * If vincenty, calculation will be slower but more accurate
    extent: tuple of float, default None
        how wide the raster will be  <br/>
        * If None, extent will be calculated from points input  <br/>
        * If tuple of float, user input of extent will be used
    power: float, default 2
        how smooth the interpolation will be
    distance_limit: float, default 0
        maximum distance to be interpolated, can't be negative

    Returns
    -------
    InterpolationResult

    Examples
    --------

    >>> xy = np.array([[106.8358,  -6.585 ],
    ...     [106.6039,  -6.7226],
    ...     [106.7589,  -6.4053],
    ...     [106.9674,  -6.7092],
    ...     [106.7956,  -6.5988]
    ... ])

    >>> values = np.array([132., 127.,  37.,  90., 182.])

    >>> idw(xy, values, spatial_res=(0.01,0.01), epsg=4326)

    >>> print(interpolated.array)
    [[ 88.63769859  86.24219616  83.60463194 ... 101.98185127 103.37001289 104.54621272]
     [ 90.12053232  87.79279317  85.22030848 ... 103.77118852 105.01425289 106.05302554]
     [ 91.82987695  89.60855271  87.14722258 ... 105.70090081 106.76928067 107.64635337]
     ...
     [127.21214817 127.33208302 127.53878268 ...  97.80436475  94.96247196 93.12113458]
     [127.11315081 127.18465002 127.33444124 ...  95.86455668  93.19212577 91.51135399]
     [127.0435062  127.0827023  127.19214624 ...  94.80175756  92.30685734 90.75707134]]
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
    distance_limit: float = 0.0,
) -> InterpolationResult:

    if extent is None:
        x_min, y_min, x_max, y_max = calc_extent(points)
        print("using points' extent")
    else:
        x_min, y_min, x_max, y_max = extent
        if x_min > x_max:
            raise ValueError(f"x_min {x_min} must be smaller than x_max {x_max}")
        elif y_min > y_max:
            raise ValueError(f"y_min {y_min} must be smaller than x_max {y_max}")

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
        distance_limit=distance_limit,
    ).reshape(rows, columns)

    return InterpolationResult(
        interpolated_value,
        interpolated_coordinate,
        crs,
        (x_min, y_min, x_max, y_max),
        source=source,
    )


def idw_single(
    point: List[float],
    known_coordinates: np.ndarray,
    known_value: np.ndarray,
    epsg: int = 4326,
    longlat_distance: str = "harvesine",
    power: Union[float, int] = 2,
    distance_limit: float = 0.0,
) -> float:
    """

    Parameters
    ----------
    point : list
        list of single point to be interpolated
    known_coordinates : numpy array
        list of points coordinate as numpy array
    known_value: numpy array
        list of points value as numpy array, not needed if vector file used as input
    epsg : int, default 4326
        EPSG code of reference system  <br/>
        * If 4326, WGS 1984 geographic system  <br/>
        * If int, epsg will be parsed
    longlat_distance: str harvesine or vincenty, default harvesine
        method used to calculate distance in spherical / ellipsoidal  <br/>
        * If harvesine, calculation will be faster but less accurate  <br/>
        * If vincenty, calculation will be slower but more accurate
    power: float, default 2
        how smooth the interpolation will be
    distance_limit: float, default 0
        maximum distance to be interpolated, can't be negative

    Returns
    -------
    float
        interpolated value

    Examples
    --------
    >>> from geosardine.interpolate import idw_single

    >>> result = idw_single(
    ...     [860209, 9295740],
    ...     np.array([[767984, 9261620], [838926, 9234594]]),
    ...     np.array([[101.1, 102.2]]),
    ...     epsg=32748,
    ...     distance_limit=0
    ... )

    >>> print(result)
    101.86735169471324

    """
    if len(point) > 2:
        raise ValueError("only for single point, input can't be more than 2 items")
    crs = CRS.from_epsg(epsg)
    distance_calculation = longlat_distance
    if crs.is_projected:
        distance_calculation = "projected"

    interpolated = _idw(
        known_coordinates,
        known_value,
        np.array([point]),
        calc_distance[distance_calculation],
        power,
        distance_limit,
    )
    return interpolated[0]


@idw.register
def _idw_file(
    file_name: str,
    spatial_res: Tuple[float, float],
    epsg: Optional[int] = None,
    column_name: Optional[str] = None,
    longlat_distance: str = "harvesine",
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Union[float, int] = 2,
    distance_limit: float = 0.0,
) -> InterpolationResult:
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
            distance_limit=distance_limit,
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
    distance_limit: float = 0.0,
):
    return _idw_file(
        file_name,
        spatial_res,
        epsg,
        column_name,
        longlat_distance,
        extent,
        power,
        distance_limit=distance_limit,
    )
