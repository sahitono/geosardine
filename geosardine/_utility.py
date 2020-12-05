import math
from functools import singledispatch
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numba
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


def calc_affine(coordinate_array: np.ndarray) -> Affine:
    x_res = coordinate_array[0, 1, 0] - coordinate_array[0, 0, 0]
    y_res = coordinate_array[1, 0, 1] - coordinate_array[0, 0, 1]
    affine = Affine.translation(*coordinate_array[0, 0]) * Affine.scale(x_res, y_res)
    return affine


def save_raster(
    file_name: Union[str, Path],
    value_array: np.ndarray,
    crs: Union[CRS, int],
    coordinate_array: Optional[np.ndarray] = None,
    affine: Optional[Affine] = None,
):
    height, width = value_array.shape
    layers = 1
    if len(value_array.shape) == 3:
        height, width, layers = value_array.shape
    if layers == 1:
        value_array = value_array.reshape(height, width, layers)

    if affine is None:
        if coordinate_array is None:
            raise ValueError("please, provide array of coordinate per pixel")
        affine = calc_affine(coordinate_array)

    if type(crs) == int:
        crs = CRS.from_epsg(crs)

    with rasterio.open(
        file_name,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=layers + 1,
        dtype=value_array.dtype,
        crs=crs,
        transform=affine,
    ) as raster:
        for layer in range(layers):
            raster.write(value_array[:, :, layer], layer + 1)
    print(f"{file_name} saved")


@singledispatch
def harvesine_distance(
    long_lat1: Union[np.ndarray, Tuple[float, float], List[float]],
    long_lat2: Union[np.ndarray, Tuple[float, float], List[float]],
) -> Optional[float]:
    """
    Calculate distance in ellipsoid by harvesine method
    faster, less accurate

    Parameters
    ----------
    long_lat1 : tuple, list, numpy array
        first point coordinate in longitude, latitude
    long_lat2 : tuple, list, numpy array
        second point coordinate in longitude, latitude

    Returns
    -------
    float
        distance

    Notes
    -------
    https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
    """

    print("only accept numpy array, list and tuple")
    return None


@harvesine_distance.register(np.ndarray)
@numba.njit()
def _harvesine_distance(long_lat1: np.ndarray, long_lat2: np.ndarray) -> float:
    radians = math.pi / 180
    long1, lat1 = long_lat1
    long2, lat2 = long_lat2

    long1 *= radians
    long2 *= radians
    lat1 *= radians
    lat2 *= radians

    earth_radius_equator = 6378137.0  # earth average radius at equador (km)
    long_diff = long2 - long1
    lat_diff = lat2 - lat1
    a = (
        math.sin(lat_diff / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(long_diff / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius_equator * c
    return np.abs(distance)


@harvesine_distance.register(list)
def __harvesine_distance(long_lat1: List[float], long_lat2: List[float]):
    return _harvesine_distance(np.array(long_lat1), np.array(long_lat2))


@harvesine_distance.register(tuple)
def __harvesine_distance(
    long_lat1: Tuple[float, float], long_lat2: Tuple[float, float]
):
    return _harvesine_distance(np.array(long_lat1), np.array(long_lat2))


@singledispatch
def vincenty_distance(
    long_lat1: Union[np.ndarray, Tuple[float, float], List[float]],
    long_lat2: Union[np.ndarray, Tuple[float, float], List[float]],
) -> Optional[float]:
    """
    Calculate distance in ellipsoid by vincenty method
    slower, more accurate

    Parameters
    ----------
    long_lat1 : tuple, list
        first point coordinate in longitude, latitude
    long_lat2 : tuple, list
        second point coordinate in longitude, latitude

    Returns
    -------
    distance

    Notes
    -------
    https://www.johndcook.com/blog/2018/11/24/spheroid-distance/
    """

    print("only accept numpy array, list and tuple")
    return None


@vincenty_distance.register(np.ndarray)
@numba.njit()
def _vincenty_distance(long_lat1: np.ndarray, long_lat2: np.ndarray) -> float:
    # WGS 1984
    earth_radius_equator = 6378137.0  # equatorial radius in meters
    flattening = 1 / 298.257223563
    earth_radius_poles = (1 - flattening) * earth_radius_equator
    tolerance = 1e-11  # to stop iteration

    radians = math.pi / 180
    long1, lat1 = long_lat1
    long2, lat2 = long_lat2

    long1 *= radians
    long2 *= radians
    lat1 *= radians
    lat2 *= radians

    distance = 0.0

    if long1 != long2 and lat1 != lat2:
        phi1, phi2 = lat1, lat2
        u1 = math.atan((1 - flattening) * math.tan(phi1))
        u2 = math.atan((1 - flattening) * math.tan(phi2))
        long_diff = long2 - long1

        lambda_old = long_diff + 0

        while True:
            t = (math.cos(u2) * math.sin(lambda_old)) ** 2
            t += (
                math.cos(u1) * math.sin(u2)
                - math.sin(u1) * math.cos(u2) * math.cos(lambda_old)
            ) ** 2
            sin_sigma = t ** 0.5
            cos_sigma = math.sin(u1) * math.sin(u2) + math.cos(u1) * math.cos(
                u2
            ) * math.cos(lambda_old)
            sigma = math.atan2(sin_sigma, cos_sigma)

            sin_alpha = math.cos(u1) * math.cos(u2) * math.sin(lambda_old) / sin_sigma
            cos_sq_alpha = 1 - sin_alpha ** 2
            cos_2sigma_m = cos_sigma - 2 * math.sin(u1) * math.sin(u2) / cos_sq_alpha
            c = (
                flattening
                * cos_sq_alpha
                * (4 + flattening * (4 - 3 * cos_sq_alpha))
                / 16
            )

            t = sigma + c * sin_sigma * (
                cos_2sigma_m + c * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
            )
            lambda_new = long_diff + (1 - c) * flattening * sin_alpha * t
            if abs(lambda_new - lambda_old) <= tolerance:
                break
            else:
                lambda_old = lambda_new

        u2 = cos_sq_alpha * (
            (earth_radius_equator ** 2 - earth_radius_poles ** 2)
            / earth_radius_poles ** 2
        )
        A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        t = cos_2sigma_m + 0.25 * B * (cos_sigma * (-1 + 2 * cos_2sigma_m ** 2))
        t -= (
            (B / 6)
            * cos_2sigma_m
            * (-3 + 4 * sin_sigma ** 2)
            * (-3 + 4 * cos_2sigma_m ** 2)
        )
        delta_sigma = B * sin_sigma * t
        distance = earth_radius_poles * A * (sigma - delta_sigma)

    return np.abs(distance)


@vincenty_distance.register(list)
def __vincenty_distance(long_lat1: List[float], long_lat2: List[float]):
    return _vincenty_distance(np.array(long_lat1), np.array(long_lat2))


@vincenty_distance.register(tuple)
def __vincenty_distance(long_lat1: Tuple[float, float], long_lat2: Tuple[float, float]):
    return _vincenty_distance(np.array(long_lat1), np.array(long_lat2))


@numba.njit()
def projected_distance(
    xy1: Union[Tuple[float, float], List[float]],
    xy2: Union[Tuple[float, float], List[float]],
) -> float:
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)


def calc_extent(points: np.ndarray) -> Tuple[float, float, float, float]:
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    return x_min, y_min, x_max, y_max


calc_distance = {
    "harvesine": _harvesine_distance,
    "vincenty": _vincenty_distance,
    "projected": projected_distance,
}
