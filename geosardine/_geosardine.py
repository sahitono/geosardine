import warnings
from collections import OrderedDict
from math import ceil, floor
from typing import Callable, Dict, Generator, Iterable, List, Tuple, Union

import fiona
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import LineString, Polygon, shape

offset_operator: Dict[str, Tuple[float, float]] = {
    "center": (0.5, 0.5),
    "ul": (0.0, 0.0),
    "bl": (1.0, 0.0),
    "ur": (0.0, 1.0),
    "br": (1.0, 1.0),
}


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
    xy : tuple, list
        2d geographic or projected coordinate
    affine : Affine
        affine parameter from rasterio.transform
        or create it with affine.Affine https://pypi.org/project/affine/
    interpolate : bool, default True
        choose to interpolate or not
        * if True, value will be interpolated from  nearest value
        * if False, value will be obtained from exact row and column

    Returns
    -------
    tuple
        row, column

    """
    col, row = ~affine * xy
    if not interpolate:
        col, row = round_function(col), round_function(row)
    return row, col


def rowcol2xy(
    row_col: Union[Tuple[int, int], List[int]], affine: Affine, offset: str = "center"
) -> Tuple[float, float]:
    """
    Convert image coordinate to geographic coordinate
    Parameters
    ----------
    row_col : tuple, list
        image coordinate in row, column
    affine : Affine
        affine parameter from rasterio.transform
        or create it with affine.Affine https://pypi.org/project/affine/
    offset : offset
        center, upper left, upper right, bottom left, bottom right

    Returns
    -------
    tuple
        2d geographic or projected coordinate

    """
    r_op, c_op = offset_operator[offset]
    row, col = row_col
    return affine * (col + c_op, row + r_op)


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
    Parameters
    ----------
    xy : tuple, list, numpy array
        2D coordinate x,y
    dsm_array : numpy array
        height array
    affine : Affine
        affine parameter from rasterio.transform
        or create it with affine.Affine https://pypi.org/project/affine/
    interpolate : bool, default True
        choose to interpolate or not
        * if True, value will be interpolated from  nearest value
        * if False, value will be obtained from exact row and column
    no_data : float, int, default -32767
        value for pixel with no data

    Returns
    -------
    tuple
        3D coordinate

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
    Parameters
    ----------
    geometry : shapely polygon, shapely linestring
        vector data as shapely object, currently only support polygon or linestring
    raster : rasterio.io.DatasetReader
        rasterio reader of raster file
    interpolate : bool, default True
        choose to interpolate or not
        * if True, value will be interpolated from  nearest value
        * if False, value will be obtained from exact row and column

    Returns
    -------
    shapely.Polygon or shapely.LineString

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
    Parameters
    ----------
    features : Iterable[Dict], fiona.Collection
        vector as geojson
    raster : rasterio.io.DatasetReader
        rasterio reader of raster file
    interpolate : bool, default True
        choose to interpolate or not
        * if True, value will be interpolated from  nearest value
        * if False, value will be obtained from exact row and column

    Yields
    -------
    dict
        geojson

    """
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
    Parameters
    ----------
    target : fiona.Collection
        vector target which you want to be joined
    join : fiona.Collection
        vector which data wont to be obtained

    Returns
    -------
    dict
        geojson

    """
    try:
        joined_schema_prop = OrderedDict(
            **target.schema["properties"], **join.schema["properties"]
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
