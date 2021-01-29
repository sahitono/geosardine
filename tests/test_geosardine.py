import numpy as np
from affine import Affine
from geosardine import (
    drape2raster,
    harvesine_distance,
    rowcol2xy,
    vincenty_distance,
    xy2rowcol,
)


def test_convert():
    col, row = 0, 100
    affine_params = Affine.from_gdal(*(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0))

    assert (-237481.5, 195036.4) == rowcol2xy((row, col), affine_params, offset="ul")
    assert (row, col) == xy2rowcol(
        (-237481.5, 195036.4), affine_params, round_function=round
    )


def test_drape():
    width = 1859
    height = 1472
    x, y = -237481.5, 195036.4
    affine_params = Affine.from_gdal(*(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0))
    dsm = np.arange(width * height).reshape((height, width))
    assert (x, y, 185900) == drape2raster((x, y), dsm, affine_params)
    assert (x, y, 186830) == drape2raster([x, y], dsm, affine_params, interpolate=True)


def test_distance():
    assert (
        harvesine_distance(
            np.array([106 + 39 / 60 + 21 / 3600, 6 + 7 / 60 + 32 / 3600]),
            np.array([110 + 3 / 60 + 26 / 3600, 7 + 54 / 60 + 19 / 3600]),
        )
        == 424815.7225367302
    )

    assert (
        harvesine_distance(
            [106 + 39 / 60 + 21 / 3600, 6 + 7 / 60 + 32 / 3600],
            [110 + 3 / 60 + 26 / 3600, 7 + 54 / 60 + 19 / 3600],
        )
        == 424815.7225367302
    )

    assert (
        vincenty_distance(
            np.array([106 + 39 / 60 + 21 / 3600, 6 + 7 / 60 + 32 / 3600]),
            np.array([110 + 3 / 60 + 26 / 3600, 7 + 54 / 60 + 19 / 3600]),
        )
        == 424229.276852855
    )

    assert (
        vincenty_distance(
            [106 + 39 / 60 + 21 / 3600, 6 + 7 / 60 + 32 / 3600],
            [110 + 3 / 60 + 26 / 3600, 7 + 54 / 60 + 19 / 3600],
        )
        == 424229.276852855
    )
