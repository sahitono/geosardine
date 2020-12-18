from typing import Any, Callable, Dict, Tuple

import numba as nb
import numpy as np


@nb.njit("UniTuple(f8,2)(UniTuple(f8,2),UniTuple(f8,9))")
def __nb_xy2rowcol(
    xy: Tuple[float, float], transform: Tuple[float, ...]
) -> Tuple[float, float]:
    """numba version of xy2rowcol

    Parameters
    ----------
    xy : Tuple[float, float]
        coordinate
    transform : Tuple[float, ...]
        affine parameter as tuple

    Returns
    -------
    Tuple[float, float]
        row, column
    """
    x, y = xy
    a, b, c, d, e, f, _, _, _ = transform
    return x * d + y * e + f, x * a + y * b + c


@nb.njit("UniTuple(f8,2)(UniTuple(f8,2),UniTuple(f8,9))")
def __nb_rowcol2xy(
    rowcol: Tuple[float, float], transform: Tuple[float, ...]
) -> Tuple[float, float]:
    row, col = rowcol
    y, x = __nb_xy2rowcol((col, row), transform)
    return x, y


@nb.njit()
def __nb_add(a, b):
    return a + b


@nb.njit()
def __nb_sub(a, b):
    return a - b


@nb.njit()
def __nb_mul(a, b):
    return a * b


@nb.njit()
def __nb_truediv(a, b):
    return a / b


@nb.njit()
def __nb_floordiv(a, b):
    return a // b


@nb.njit()
def __nb_pow(a, b):
    return a ** b


@nb.njit()
def __nb_iadd(a, b):
    return a + b


@nb.njit()
def __nb_isub(a, b):
    return a - b


@nb.njit()
def __nb_imul(a, b):
    return a * b


@nb.njit()
def __nb_itruediv(a, b):
    return a / b


@nb.njit()
def __nb_ifloordiv(a, b):
    return a // b


@nb.njit()
def __nb_ipow(a, b):
    return a ** b


__nb_raster_ops: Dict[str, Callable[[Any, Any], Any]] = {
    "add": __nb_add,
    "sub": __nb_sub,
    "mul": __nb_mul,
    "truediv": __nb_truediv,
    "floordiv": __nb_floordiv,
    "ipow": __nb_ipow,
    "iadd": __nb_iadd,
    "isub": __nb_isub,
    "imul": __nb_imul,
    "itruediv": __nb_itruediv,
    "ifloordiv": __nb_ifloordiv,
    "ipow": __nb_ipow,
}


@nb.njit()
def __nb_raster_calc__(
    raster_a: np.ndarray,
    raster_b: np.ndarray,
    transform_a: Tuple[float, ...],
    inverse_transform_b: Tuple[float, ...],
    nodata_a: float,
    nodata_b: float,
    operator: Callable[[Any, Any], Any],
) -> np.ndarray:
    """Raster calculation per pixel using numba jit

    Parameters
    ----------
    raster_a : np.ndarray
        first raster array
    raster_b : np.ndarray
        second raster array
    transform_a : Tuple[float, ...]
        first raster affine parameter
    inverse_transform_b : Tuple[float, ...]
        second raster inversed affine parameter. it should be inversed, needed to convert xy to row column
    nodata_a : float
        the no data value of first raster
    nodata_b : float
        the no data value of second raster
    operator : Callable[[Any, Any], Any]
        operator but in numba wrapper. needed to be wrapped because numba doesn't support generic operator function

    Returns
    -------
    np.ndarray
        calculated raster
    """
    _r_out = np.zeros(raster_a.shape, dtype=raster_a.dtype)
    for row in range(_r_out.shape[0]):
        for col in range(_r_out.shape[1]):
            pixel_a = raster_a[row, col]
            _r_b_row, _r_b_col = __nb_xy2rowcol(
                __nb_rowcol2xy((row, col), transform_a), inverse_transform_b
            )

            if _r_b_row > _r_out.shape[0] or _r_b_col > _r_out.shape[1]:
                pixel_b = nodata_b
            elif _r_b_row < 0 or _r_b_col < 0:
                pixel_b = nodata_b
            else:
                pixel_b = raster_b[int(_r_b_row), int(_r_b_col)]

            if pixel_a != nodata_a and pixel_b != nodata_a:
                _r_out[row, col] = operator(pixel_a, pixel_b)
            else:
                _r_out[row, col] = nodata_a
    return _r_out
