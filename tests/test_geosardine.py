import numpy as np
from affine import Affine
from geosardine import interpolate, rowcol2xy, xy2rowcol, drape2raster
from geosardine._utility import calc_extent


def test_convert():
    col, row = 0, 100
    affine_params = Affine.from_gdal(*(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0))

    assert (-237481.5, 195036.4) == rowcol2xy((row, col), affine_params)
    assert (row, col) == xy2rowcol(
        (-237481.5, 195036.4), affine_params, round_function=round
    )


def test_drape():
    width = 1859
    height = 1472
    x, y = -237481.5, 195036.4
    affine_params = Affine.from_gdal(*(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0))
    dsm = np.arange(width * height).reshape((height, width))
    assert (x, y, 184041) == drape2raster([x, y], dsm, affine_params)
    assert (x, y, 184970.5) == drape2raster(
        [x, y], dsm, affine_params, interpolate=True
    )


def test_idw():
    xy = np.load("tests/idw/test_idw_xy.npy")

    values = np.load("tests/idw/test_idw_val.npy")

    assert (
        np.load("tests/idw/test_idw_array.npy")
        == interpolate.idw(xy, values, (0.01, 0.01), extent=calc_extent(xy)).array
    ).all()

    assert (
        np.load("tests/idw/test_idw_file_array.npy")
        == interpolate.idw(
            "tests/idw/test_idw_file.geojson",
            (0.01, 0.01),
            column_name="week1",
            extent=(106.6, -6.72, 106.97, -6.41),
        ).array
    ).all()
