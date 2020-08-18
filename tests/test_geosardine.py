import numpy as np
from affine import Affine
from geosardine import rowcol2xy, xy2rowcol, drape2raster


def test_convert():
    col, row = 0, 100
    affine_params = Affine.from_gdal(*(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0))

    assert (-237481.5, 195036.4) == rowcol2xy((row, col), affine_params)
    assert (row, col) == xy2rowcol((-237481.5, 195036.4), affine_params)


def test_drape():
    width = 1859
    height = 1472
    x, y = -237481.5, 195036.4
    affine_params = Affine.from_gdal(*(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0))
    dsm = np.arange(width * height).reshape((height, width))
    assert (x, y, 185900) == drape2raster([x, y], dsm, affine_params)
    assert (x, y, 184970.5) == drape2raster(
        [x, y], dsm, affine_params, interpolate=True
    )
