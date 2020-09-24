import warnings
from collections import OrderedDict
from math import ceil, floor
from typing import Callable, Dict, Generator, Tuple, List, Union, Iterable

import numpy as np
import fiona
import rasterio
from affine import Affine
from shapely.geometry import shape, Polygon, LineString


def xy2rowcol(
    xy: Union[Tuple[float, float], List[float]],
    affine: Affine,
    interpolate: bool = False,
    round_function: Callable = int,
) -> Union[Tuple[int, int], Tuple[float, float]]:
    """
    Convert geographic coordinate to image coordinate
    Parameters
    ----------
    xy
    affine
    interpolate

    Returns
    -------

    """
    col, row = ~affine * xy
    if not interpolate:
        col, row = round_function(col), round_function(row)
    return row, col


def rowcol2xy(
    row_col: Union[Tuple[int, int], List[int]], affine: Affine
) -> Tuple[float, float]:
    """
    Convert image coordinate to geographic coordinate
    :param row_col:
    :param affine:
    :return:
    """
    row, col = row_col
    return affine * (col, row)


def _d2r_interpolate(
    row_col: Union[Tuple[float, float], List[float]],
    dsm_array: np.ndarray,
    no_data: Union[float, int],
) -> float:
    """
    Find interpolated z value on raster.
    Z is average of 4 avalue ignoring distance to each coordinate.
    :param row_col:
    :param dsm_array:
    :param no_data:
    :return: Interpolated Z
    """
    row, col = row_col
    try:
        max_col, max_row = floor(col), floor(row)
        min_col, min_row = ceil(col), ceil(row)

        z_list = list(
            filter(
                lambda x: x > no_data,
                [
                    dsm_array[min_row, min_col],
                    dsm_array[max_row, max_col],
                    dsm_array[min_row, max_col],
                    dsm_array[max_row, min_col],
                ],
            )
        )

        draped_z = sum(z_list) / len(z_list)
    except IndexError:
        warnings.warn(f"Point is out of bound, returning no data: {no_data}")
        draped_z = no_data

    return draped_z


def drape2raster(
    xy: List[float],
    dsm_array: np.ndarray,
    affine: Affine,
    interpolate: bool = False,
    no_data: Union[float, int] = -32767,
) -> Tuple[float, float, float]:
    """
    Find Z of 2D coordinate
    :param xy: 2D coordinate [x, y]
    :param dsm_array: dsm as numpy array
    :param affine: affine parameter from rasterio.transform
    :param interpolate: interpolate or exact value
    :param no_data: no data value
    :return: 3D coordinate
    """
    x, y = xy
    row, col = xy2rowcol(xy, affine, interpolate)
    if interpolate:
        draped_z = _d2r_interpolate((row, col), dsm_array, no_data)
    else:
        try:
            draped_z = dsm_array[row, col]
        except IndexError:
            warnings.warn(f"Point is out of bound, returning no data: {no_data}")
            draped_z = no_data
    return x, y, draped_z


def drape_coordinates(
    coordinates: Union[List[List[float]], Iterable],
    dsm_array: np.ndarray,
    affine: Affine,
    interpolate: bool = False,
    no_data: Union[float, int] = -32767,
) -> Generator[List[float], None, None]:
    for xy in coordinates:
        yield drape2raster(xy, dsm_array, affine, interpolate, no_data)


def drape_shapely(
    geometry: Union[Polygon, LineString],
    raster: rasterio.io.DatasetReader,
    interpolate: bool = False,
) -> Union[Polygon, LineString]:
    """
    Drape with shapely geometry as input
    :param geometry:
    :param raster: rasterio dataset reader
    :param interpolate:
    :return:
    """
    dsm_array = raster.read(1)
    affine = raster.transform
    no_data = raster.nodatavals
    if geometry.type == "Polygon":
        draped_exterior = list(
            drape_coordinates(
                geometry.exterior.coords, dsm_array, affine, interpolate, no_data
            )
        )
        draped_interiors = [
            list(
                drape_coordinates(
                    interior.coords, dsm_array, affine, interpolate, no_data
                )
            )
            for interior in geometry.interiors
        ]

        return Polygon(draped_exterior, draped_interiors)
    elif geometry.type == "LineString":
        return LineString(
            list(
                drape_coordinates(
                    geometry.coords, dsm_array, affine, interpolate, no_data
                )
            )
        )
    else:
        raise ValueError("Unsupported geometry type")


def drape_geojson(
    features: Union[Iterable[Dict], fiona.Collection],
    raster: rasterio.io.DatasetReader,
    interpolate: bool = False,
) -> Generator[Dict, None, None]:
    """
    Drape with geojson as input, fiona uses geojson as interface.
    :param features: geojson
    :param raster: rasterio dataset reader
    :param interpolate:
    :return:
    """
    dsm_array = raster.read(1)
    affine = raster.transform
    no_data = raster.nodatavals
    for i, feature in enumerate(features):
        draped_feature = feature.copy()
        geometry: Dict = feature["geometry"]
        geom_type: str = geometry["type"]

        draped_coordinates: Union[List[List[float]], List[List[List[float]]]] = []
        if geom_type == "Polygon":
            draped_coordinates = [
                list(drape_coordinates(ring, dsm_array, affine, interpolate, no_data))
                for ring in geometry
            ]

        elif geom_type == "LineString":
            draped_coordinates = list(
                drape_coordinates(geometry, dsm_array, affine, interpolate, no_data)
            )
        else:
            raise ValueError("Unsupported geometry type")

        draped_feature["geometry"]["coordinates"] = draped_coordinates
        yield draped_feature


def spatial_join(
    target: fiona.Collection, join: fiona.Collection
) -> Tuple[List[Dict], Dict]:
    """
    Join attribute from 2 vector by location.
    :param target: Vector target to be joined. [fiona.Collection]
    :param join:
    :return:
    """
    try:
        joined_schema_prop = OrderedDict(
            **target.schema["properties"], **join.schema["properties"],
        )
    except TypeError:
        raise TypeError("There are column with same name. Please change it first.")

    joined_schema = target.schema.copy()
    joined_schema["properties"] = joined_schema_prop

    joined_features = []
    join_polygon: List[Polygon] = [shape(feature["geometry"]) for feature in join]

    for feature in target:
        target_polygon = shape(feature["geom"])

        overlap_areas = (
            target_polygon.intersection(polygon).area for polygon in join_polygon
        )
        overlap_ratios = [
            overlap_area / target_polygon for overlap_area in overlap_areas
        ]

        max_ratio_index = overlap_ratios.index(max(overlap_ratios))

        joined_prop = OrderedDict(
            **feature["properties"], **join[max_ratio_index]["properties"]
        )

        feature["properties"] = joined_prop
        joined_features.append(feature)

    return joined_features, joined_schema
