import numpy as np
from geosardine import Raster


def test_raster() -> None:
    raster = Raster(np.arange(32).reshape(4, 4, 2), 0.3, 120, 20)

    assert (raster.array == np.arange(32).reshape(4, 4, 2)).all()
    assert raster.resolution == (0.3, -0.3)
    assert raster.x_max == 120 + (0.3 * 4)
    assert raster.y_min == 20 - (0.3 * 4)
    assert isinstance(raster[1:3, 1:3], np.ndarray)
    assert raster[1:3, 1:3].shape == (2, 2, 2)


def test_raster_operator() -> None:
    raster = Raster(np.ones(32, dtype=np.float32).reshape(4, 4, 2), 0.3, 120, 0.7)
    arr = np.ones(32, dtype=np.float32).reshape(4, 4, 2)
    assert ((raster * 2).array == arr * 2).all()
    assert ((raster + 2).array == arr + 2).all()
    assert ((raster - 2).array == arr - 2).all()
    assert ((raster / 2).array == arr / 2).all()
    assert ((raster ** 2).array == arr ** 2).all()


def test_raster_manipulation() -> None:
    raster = Raster(np.ones(32, dtype=np.float32).reshape(4, 4, 2), 0.4, 120, 0.7)
    resized = raster.resize(16, 16)
    resampled = raster.resample((0.1, 0.1))

    assert (
        resized.array == np.ones(16 * 16 * 2, dtype=np.float32).reshape(16, 16, 2)
    ).all()
    assert (round(resized.resolution[0], 2), round(resized.resolution[1], 2)) == (
        0.1,
        -0.1,
    )
    assert (
        resampled.array == np.ones(16 * 16 * 2, dtype=np.float32).reshape(16, 16, 2)
    ).all()
    assert resampled.rows == 16
    assert resampled.cols == 16
    assert resampled.layers == 2
    assert raster.transform[5] == resampled.transform[5]
    assert raster.transform[2] == resampled.transform[2]
