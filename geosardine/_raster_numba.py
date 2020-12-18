from typing import Any, Callable, Dict, Tuple

import numba as nb
import numpy as np


@nb.njit("UniTuple(f8,2)(UniTuple(f8,2),UniTuple(f8,9))")
def __nb_xy2rowcol(
    xy: Tuple[float, float], inverse_transform: Tuple[float, ...]
) -> Tuple[float, float]:
    """numba version of xy2rowcol

    Parameters
    ----------
    xy : Tuple[float, float]
        coordinate
    inverse_transform : Tuple[float, ...]
        affine parameter as tuple

    Returns
    -------
    Tuple[float, float]
        row, column
    """

    x, y = xy
    a, b, c, d, e, f, _, _, _ = inverse_transform
    return x * d + y * e + f, x * a + y * b + c


@nb.njit("UniTuple(f8,2)(UniTuple(f8,2),UniTuple(f8,9), unicode_type)")
def __nb_rowcol2xy(
    rowcol: Tuple[float, float], transform: Tuple[float, ...], offset: str
) -> Tuple[float, float]:
    """numba version of rowcol2xy

    Parameters
    ----------
    rowcol : Tuple[float, float]
        row, column
    transform : Tuple[float, ...]
        affine as tuple
    offset : str
        determine the offset
        * if center, center pixel
        * if ul, upper left
        * if bl, bottom left
        * if ur, upper right
        * if br, bottom right

    Returns
    -------
    Tuple[float, float]
        x y coordinate
    """
    if offset == "center":
        r_offset = 0.5
        c_offset = 0.5
    elif offset == "ul":
        r_offset = 0.0
        c_offset = 0.0
    elif offset == "bl":
        r_offset = 1.0
        c_offset = 0.0
    elif offset == "ur":
        r_offset = 0.0
        c_offset = 1.0
    elif offset == "br":
        r_offset = 1.0
        c_offset = 1.0

    row, col = rowcol
    y, x = __nb_xy2rowcol((col + c_offset, row + r_offset), transform)
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


nb_raster_ops: Dict[str, Callable[[Any, Any], Any]] = {
    "add": __nb_add,
    "sub": __nb_sub,
    "mul": __nb_mul,
    "truediv": __nb_truediv,
    "floordiv": __nb_floordiv,
    "pow": __nb_pow,
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
    _nd_b = np.array([nodata_b for _ in range(raster_b.shape[2])], dtype=raster_b.dtype)
    for row in range(_r_out.shape[0]):
        for col in range(_r_out.shape[1]):
            pixel_a = raster_a[row, col]
            _r_b_row, _r_b_col = __nb_xy2rowcol(
                __nb_rowcol2xy((row, col), transform_a, str("center")),
                inverse_transform_b,
            )

            if _r_b_row > raster_b.shape[0] or _r_b_col > raster_b.shape[1]:
                pixel_b: np.ndarray = _nd_b
            elif _r_b_row < 0 or _r_b_col < 0:
                pixel_b = _nd_b
            else:
                pixel_b = raster_b[int(_r_b_row), int(_r_b_col)]

            if not np.any(pixel_a == nodata_a) and not np.any(pixel_b == nodata_b):
                _r_out[row, col] = operator(pixel_a, pixel_b)
            else:
                _r_out[row, col] = np.array(
                    [nodata_a for _ in range(_r_out.shape[2])],
                    dtype=raster_a.dtype,
                )
    return _r_out
