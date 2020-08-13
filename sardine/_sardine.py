import warnings
from collections import OrderedDict
from math import ceil, floor
from typing import Dict, Tuple, List, Union

import numpy as np
import fiona
from affine import Affine
from shapely.geometry import shape, Polygon


def xy2rowcol(
    xy: Union[Tuple[float, float], List[float, float]],
    affine: Affine,
    mode: str = "exact",
) -> Union[Tuple[int, int], Tuple[float, float]]:
    """
    Convert geographic coordinate to image coordinate
    :param xy:
    :param affine:
    :param mode:
    :return:
    """
    col, row = ~affine * xy
    if mode == "exact":
        col, row = round(col), round(row)
    return col, row


def rowcol2xy(
    rowcol: Union[Tuple[int, int], List[int, int]], affine: Affine
) -> Tuple[float, float]:
    """
    Convert image coordinate to geographic coordinate
    :param rowcol:
    :param affine:
    :return:
    """
    return affine * rowcol


def _d2r_interpolate(
    row: float, col: float, dsm_array: np.ndarray, no_data: Union[float, int]
) -> float:
    """
    Find interpolated z value on raster.
    Z is average of 4 avalue ignoring distance to each coordinate.
    :param row:
    :param col:
    :param dsm_array:
    :param no_data:
    :return: Interpolated Z
    """
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
    except IndentationError:
        warnings.warn(f"Point is out of bound, returning no data: {no_data}")
        draped_z = no_data

    return draped_z


def drape2raster(
    xy: List[float, float],
    dsm_array: np.ndarray,
    affine: Affine,
    mode: str = "exact",
    no_data: Union[float, int] = -32767,
) -> Tuple[float, float, float]:
    """
    Find Z of 2D coordinate
    :param xy: 2D coordinate [x, y]
    :param dsm_array: dsm as numpy array
    :param affine: affine parameter from rasterio.transform
    :param mode: "exact" or "interpolate"
    :param no_data: no data value
    :return: 3D coordinate
    """
    x, y = xy
    row, col = xy2rowcol(xy, affine, mode)
    if mode == "interpolate":
        draped_z = _d2r_interpolate(row, col, dsm_array, no_data)
    else:
        try:
            draped_z = dsm_array[row, col]
        except IndentationError:
            warnings.warn(f"Point is out of bound, returning no data: {no_data}")
            draped_z = no_data
    return x, y, draped_z


def spatial_join(target: fiona.Collection, join: fiona.Collection) -> List[Dict]:
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

    return joined_features
