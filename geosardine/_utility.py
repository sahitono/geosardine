import math
from functools import singledispatch
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numba
import numpy as np
import rasterio
from affine import Affine
from osgeo import osr
from rasterio.crs import CRS


def calc_affine(coordinate_array: np.ndarray) -> Affine:
    x_res = coordinate_array[0, 1, 0] - coordinate_array[0, 0, 0]
    y_res = coordinate_array[1, 0, 1] - coordinate_array[0, 0, 1]
    x_min, y_max = coordinate_array[0, 0]
    x_min -= x_res / 2
    y_max -= y_res / 2
    affine = Affine.translation(*coordinate_array[0, 0]) * Affine.scale(x_res, y_res)
    return affine


def get_ellipsoid_par(epsg: int) -> Tuple[float, float, float]:
    crs = CRS.from_epsg(epsg)
    semi_major = float(
        osr.SpatialReference(wkt=crs.to_wkt()).GetAttrValue("SPHEROID", 1)
    )
    inverse_flattening = float(
        osr.SpatialReference(wkt=crs.to_wkt()).GetAttrValue("SPHEROID", 2)
    )
    semi_minor: float = (1 - (1 / inverse_flattening)) * semi_major
    return semi_major, semi_minor, inverse_flattening


def save_raster(
    file_name: Union[str, Path],
    value_array: np.ndarray,
    crs: Union[CRS, int],
    coordinate_array: Optional[np.ndarray] = None,
    affine: Optional[Affine] = None,
    nodata: Union[None, float, int] = None,
    compress: bool = False,
) -> None:

    if len(value_array.shape) == 3:
        height, width, layers = value_array.shape
    else:
        height, width = value_array.shape
        layers = 1

        value_array = value_array.reshape(height, width, layers)

    _compress = None
    if compress:
        _compress = "lzw"

    if affine is None:
        if coordinate_array is None:
            raise ValueError("please, provide array of coordinate per pixel")
        affine = calc_affine(coordinate_array)

    if type(crs) == int:
        crs = CRS.from_epsg(crs)
    if nodata is not None:
        with rasterio.open(
            file_name,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=layers,
            dtype=value_array.dtype,
            crs=crs,
            transform=affine,
            nodata=nodata,
            compress=_compress,
        ) as raster:
            for layer in range(layers):
                raster.write(value_array[:, :, layer], layer + 1)
    else:
        with rasterio.open(
            file_name,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=layers,
            dtype=value_array.dtype,
            crs=crs,
            transform=affine,
            compress=_compress,
        ) as raster:
            for layer in range(layers):
                raster.write(value_array[:, :, layer], layer + 1)
    print(f"{file_name} saved")


def harvesine_distance(
    long_lat1: Union[np.ndarray, Tuple[float, float], List[float]],
    long_lat2: Union[np.ndarray, Tuple[float, float], List[float]],
    epsg: int = 4326,
) -> Optional[float]:

    """Calculate distance in ellipsoid by harvesine method
    faster, less accurate

    Parameters
    ----------
    long_lat1 : tuple, list
        first point coordinate in longitude, latitude
    long_lat2 : tuple, list
        second point coordinate in longitude, latitude
    epsg : int, optional
        epsg code of the spatial reference system, by default 4326

    Returns
    -------
    Optional[float]
        distance, if None then input is not np.ndarray, tuple or list

    Notes
    -------
    https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
    """

    semi_major, semi_minor, i_flattening = get_ellipsoid_par(epsg)

    return _harvesine_distance_dispatch(
        long_lat1,
        long_lat2,
        semi_major=semi_major,
        semi_minor=semi_minor,
        i_flattening=i_flattening,
    )


@singledispatch
def _harvesine_distance_dispatch(
    long_lat1: Union[np.ndarray, Tuple[float, float], List[float]],
    long_lat2: Union[np.ndarray, Tuple[float, float], List[float]],
    semi_major: float,
    semi_minor: float,
    i_flattening: float,
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
    semi_major : float
        ellipsoid's semi major axes
    semi_minor : float
        ellipsoid's semi minor axes
    i_flattening : float
        ellipsoid's inverse flattening

    Returns
    -------
    Optional[float]
        distance, if None then input is not np.ndarray, tuple or list

    Notes
    -------
    https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
    """

    print("only accept numpy array, list and tuple")
    return None


@_harvesine_distance_dispatch.register(np.ndarray)
@numba.njit()
def _harvesine_distance(
    long_lat1: np.ndarray,
    long_lat2: np.ndarray,
    semi_major: float = 6378137.0,
    semi_minor: float = 6356752.314245179,
    i_flattening=298.257223563,
) -> float:
    long1, lat1 = np.radians(long_lat1)
    long2, lat2 = np.radians(long_lat2)

    long_diff = long2 - long1
    lat_diff = lat2 - lat1
    a = (
        math.sin(lat_diff / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(long_diff / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return np.abs(semi_major * c)


@_harvesine_distance_dispatch.register(list)
def __harvesine_distance(
    long_lat1: List[float],
    long_lat2: List[float],
    semi_major: float,
    semi_minor: float,
    i_flattening: float,
):
    return _harvesine_distance(
        np.array(long_lat1),
        np.array(long_lat2),
        semi_major,
        semi_minor,
        i_flattening,
    )


@_harvesine_distance_dispatch.register(tuple)
def __harvesine_distance(
    long_lat1: Tuple[float, float],
    long_lat2: Tuple[float, float],
    semi_major: float,
    semi_minor: float,
    i_flattening: float,
):
    return _harvesine_distance(
        np.array(long_lat1),
        np.array(long_lat2),
        semi_major,
        semi_minor,
        i_flattening,
    )


def vincenty_distance(
    long_lat1: Union[np.ndarray, Tuple[float, float], List[float]],
    long_lat2: Union[np.ndarray, Tuple[float, float], List[float]],
    epsg: int = 4326,
) -> float:
    """Calculate distance in ellipsoid by vincenty method
    slower, more accurate

    Parameters
    ----------
    long_lat1 : tuple, list
        first point coordinate in longitude, latitude
    long_lat2 : tuple, list
        second point coordinate in longitude, latitude
    epsg : int, optional
        epsg code of the spatial reference system, by default 4326

    Returns
    -------
    float
        distance

    Notes
    -------
    https://www.johndcook.com/blog/2018/11/24/spheroid-distance/
    """

    semi_major, semi_minor, i_flattening = get_ellipsoid_par(epsg)

    return _vincenty_distance_dispatch(
        long_lat1,
        long_lat2,
        semi_major=semi_major,
        semi_minor=semi_minor,
        i_flattening=i_flattening,
    )


@singledispatch
def _vincenty_distance_dispatch(
    long_lat1: Union[np.ndarray, Tuple[float, float], List[float]],
    long_lat2: Union[np.ndarray, Tuple[float, float], List[float]],
    epsg: int,
    semi_major: float,
    semi_minor: float,
    i_flattening: float,
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
    semi_major : float
        ellipsoid's semi major axes
    semi_minor : float
        ellipsoid's semi minor axes
    i_flattening : float
        ellipsoid's inverse flattening

    Returns
    -------
    distance

    Notes
    -------
    https://www.johndcook.com/blog/2018/11/24/spheroid-distance/
    """

    print("only accept numpy array, list and tuple")
    return None


@_vincenty_distance_dispatch.register(np.ndarray)
@numba.njit()
def _vincenty_distance(
    long_lat1: np.ndarray,
    long_lat2: np.ndarray,
    semi_major: float = 6378137.0,
    semi_minor: float = 6356752.314245179,
    i_flattening=298.257223563,
) -> float:
    # WGS 1984
    flattening = 1 / i_flattening
    tolerance = 1e-11  # to stop iteration

    radians = math.pi / 180
    long1, lat1 = np.radians(long_lat1)
    long2, lat2 = np.radians(long_lat2)

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

        u2 = cos_sq_alpha * ((semi_major ** 2 - semi_minor ** 2) / semi_minor ** 2)
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
        distance = semi_minor * A * (sigma - delta_sigma)

    return np.abs(distance)


@_vincenty_distance_dispatch.register(list)
def __vincenty_distance(
    long_lat1: List[float], long_lat2: List[float], *args, **kwargs
):
    return _vincenty_distance(np.array(long_lat1), np.array(long_lat2), *args, **kwargs)


@_vincenty_distance_dispatch.register(tuple)
def __vincenty_distance(
    long_lat1: Tuple[float, float], long_lat2: Tuple[float, float], *args, **kwargs
):
    return _vincenty_distance(np.array(long_lat1), np.array(long_lat2), *args, **kwargs)


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
