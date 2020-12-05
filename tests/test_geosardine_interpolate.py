import numpy as np
import pytest

from geosardine import interpolate
from geosardine._utility import calc_extent


def test_idw():
    xy = np.load("tests/idw/test_idw_xy.npy")

    values = np.load("tests/idw/test_idw_val.npy")

    # assert (
    #     np.load("tests/idw/test_idw_array.npy")
    #     == interpolate.idw(xy, values, (0.01, 0.01), extent=calc_extent(xy)).array
    # ).all()

    assert (
        interpolate.idw(xy, values, (0.01, 0.01), extent=calc_extent(xy)).array
        is not None
    )

    with pytest.warns(UserWarning):
        interp = interpolate.idw("tests/idw/test_idw_file.geojson", (0.01, 0.01))
        # assert (np.load("tests/idw/test_idw_file_array.npy") == interp.array).all()
        assert type(interp.array) == np.ndarray and len(interp.array.shape) == 2
        assert interp.save() is None

    assert interpolate.idw(1, np.array([0, 1]), (0.1, 0.1)) is None
    assert interpolate.idw_single([101, -7], xy, values) == 113.8992997794633

    assert interpolate.idw_single([101, -7], xy, values) == 113.8992997794633
    assert (
        round(interpolate.idw_single(
            [860209, 9295740],
            np.array([[767984, 9261620], [838926, 9234594]]),
            np.array([[101.1, 102.2]]),
            epsg=32748,
        ),8)
        == round(101.86735169471324,8)
    )

    with pytest.raises(ValueError):
        interpolate.idw(xy, values, (0.01, 0.01), extent=calc_extent(xy)).save()
