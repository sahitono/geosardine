from operator import add, iadd, imul, ipow, isub, itruediv, mul, pow, sub, truediv
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Union

import cv2
import numba
import numpy as np
from affine import Affine
from rasterio.crs import CRS

from geosardine._geosardine import rowcol2xy, xy2rowcol
from geosardine._utility import save_raster


class Raster(np.ndarray):
    """
    Construct Raster from numpy array with spatial information.
    Support calculation between different raster

    Parameters
    ----------
    array : numpy array
        array of raster
    resolution : tuple, list, default None
        spatial resolution
    x_min : float, defaults to None
        left boundary of x-axis coordinate
    y_min : float, defaults to None
        bottom boundary of y-axis coordinate
    x_max : float, defaults to None
        right boundary of x-axis coordinate
    y_max : float, defaults to None
        upper boundary of y-axis coordinate
    epsg : int, defaults to 4326
        EPSG code of reference system

    Examples
    --------
    >>> from geosardine import Raster
    >>> raster = Raster(np.ones(18, dtype=np.float32).reshape(3, 3, 2), resolution=0.4, x_min=120, y_max=0.7)
    >>> print(raster)
    [[[1. 1.]
      [1. 1.]
      [1. 1.]]
     [[1. 1.]
      [1. 1.]
      [1. 1.]]
     [[1. 1.]
      [1. 1.]
      [1. 1.]]]
    Raster can be resampled like this. (0.2,0.2) is the result's spatial resolution
    >>> resampled = raster.resample((0.2,0.2))
    >>> print(resampled.shape, resampled.resolution)
    (6, 6, 2) (0.2, 0.2)
    Raster can be resized
    >>> resized = raster.resize(height=16, width=16)
    >>> print(resized.shape, resized.resolution)
    (16, 16, 2) (0.07500000000000018, 0.07500000000000001)
    """

    __cv2_resize_method = {
        "nearest": cv2.INTER_NEAREST,
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(
        self,
        array: np.ndarray,
        resolution: Union[
            None, Tuple[float, float], List[float], Tuple[float, ...], float
        ] = None,
        x_min: Optional[float] = None,
        y_max: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        epsg: int = 4326,
        no_data: Union[None, float, int] = None,
    ):
        if (
            resolution is None
            and x_min is None
            and y_min is None
            and x_max is None
            and y_max is None
        ):
            raise ValueError(
                "Please define resolution and at least x minimum and y minimum"
            )

        if resolution is not None and x_min is None and y_min is None:
            raise ValueError("Please at least define x_min and y_min")

        if isinstance(resolution, float):
            self.resolution: Tuple[float, float] = (
                resolution,
                resolution,
            )
        elif resolution is not None and isinstance(resolution, Iterable):
            self.resolution = (resolution[0], resolution[1])

        if resolution is not None and x_min is not None and y_max is not None:
            self.x_min: float = x_min
            self.y_max: float = y_max
            self.x_max: float = x_min + (self.resolution[0] * array.shape[1])
            self.y_min: float = y_max - (self.resolution[1] * array.shape[0])
        elif (
            resolution is None
            and x_min is not None
            and y_min is not None
            and x_max is not None
            and y_max is not None
        ):
            self.resolution = (
                (x_max - x_min) / array.shape[1],
                (y_max - y_min) / array.shape[0],
            )
            self.x_min = x_min
            self.y_min = y_min
            self.x_max = x_max
            self.y_max = y_max

        self.array = array
        self.epsg = epsg
        self.transform = Affine.translation(self.x_min, self.y_min) * Affine.scale(
            self.resolution[0], -self.resolution[1]
        )
        self.crs = CRS.from_epsg(epsg)
        self.no_data = no_data
        self.__check_validity()

    def __new__(cls, array: np.ndarray, *args, **kwargs) -> "Raster":
        return array.view(cls)

    @property
    def rows(self) -> int:
        """number of row, height

        Returns
        -------
        int
            number of row
        """
        return int(self.array.shape[0])

    @property
    def cols(self) -> int:
        """number of column, width

        Returns
        -------
        int
            number of column
        """
        return int(self.array.shape[1])

    @property
    def layers(self) -> int:
        """number of layer, channel

        Returns
        -------
        int
            number of layer
        """
        _layers = 1
        if len(self.array.shape) > 2:
            _layers = self.array.shape[2]
        return _layers

    @property
    def x_extent(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_extent(self) -> float:
        return self.y_max - self.y_min

    @property
    def is_projected(self) -> bool:
        return self.crs.is_projected

    @property
    def is_geographic(self) -> bool:
        return self.crs.is_geographic

    def __check_validity(self) -> None:
        if self.x_extent < 0 and self.y_extent < 0:
            raise ValueError(
                "x min should be less than x max and y min should be less than y max"
            )
        elif self.x_extent < 0 and self.y_extent > 0:
            raise ValueError("x min should be less than x max")
        elif self.x_extent > 0 and self.y_extent < 0:
            raise ValueError("y min should be less than y max")

    def xy_value(self, x: float, y: float) -> Union[float, int, np.ndarray]:
        return self.array[xy2rowcol((x, y), self.transform)]

    def rowcol2xy(self, row: int, col: int) -> Tuple[float, float]:
        return rowcol2xy((row, col), self.transform)

    def xy2rowcol(self, x: float, y: float) -> Tuple[int, int]:
        _row, _col = xy2rowcol((x, y), self.transform)
        return int(_row), int(_col)

    def __raster_calc_by_pixel__(
        self,
        raster: "Raster",
        operator: Callable[[Any, Any], Any],
    ) -> np.ndarray:
        _raster = np.zeros(self.array.shape, dtype=self.array.dtype)
        for row in range(self.rows):
            for col in range(self.cols):
                pixel_source = self.array[row, col]
                pixel_target = raster.xy_value(*self.rowcol2xy(row, col))
                if pixel_source != self.no_data and pixel_target != self.no_data:
                    _raster[row, col] = operator(
                        pixel_source,
                        pixel_target,
                    )
                else:
                    _raster[row, col] = self.no_data
        return _raster

    def __raster_calculation__(
        self,
        raster: Union[int, float, "Raster"],
        operator: Callable[[Any, Any], Any],
    ) -> "Raster":
        if not isinstance(raster, (int, float, Raster)):
            raise ValueError(f"{type(raster)} unsupported data format")

        if isinstance(raster, Raster):
            if (
                raster.epsg == self.epsg
                and raster.resolution == self.resolution
                and raster.x_min == self.x_min
                and raster.y_min == self.y_min
                and raster.rows == self.rows
                and raster.cols == self.cols
                and raster.layers == self.layers
            ):
                _raster = operator(self.array, raster.array)
            else:
                _raster = self.__raster_calc_by_pixel__(raster, operator)
        else:
            _raster = operator(self.array, raster)

        return Raster(_raster, self.resolution, self.x_min, self.y_min, epsg=self.epsg)

    def __sub__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, sub)

    def __add__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, add)

    def __mul__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, mul)

    def __truediv__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, truediv)

    def __pow__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, pow)

    def __iadd__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, iadd)

    def __itruediv__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, itruediv)

    def __imul__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, imul)

    def __isub__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, isub)

    def __ipow__(self, raster: Union[int, float, "Raster"]) -> "Raster":
        return self.__raster_calculation__(raster, ipow)

    def __iter__(self) -> Generator[Any, None, None]:
        _iter_shape: Union[Tuple[int, int], int] = (self.rows * self.cols, self.layers)
        if self.layers == 1:
            _iter_shape = self.rows * self.cols
        _iter = self.array.reshape(_iter_shape)
        for i in range(10):
            yield _iter[i]

    def save(self, file_name: str) -> None:
        save_raster(file_name, self.array, self.crs, affine=self.transform)

    def resize(
        self, height: int, width: int, method: str = "bilinear", backend: str = "opencv"
    ) -> "Raster":
        """[summary]

        Parameters
        -------
        height: int
            height defined
        width: int
            width defined
        method: str nearest or bicubic or bilinear or area or lanczos, default bilinear
            resampling method for opencv
            * if nearest, a nearest-neighbor interpolation
            * if bicubic, a bicubic interpolation over 4×4 pixel neighborhood
            * if bilinear, a bilinear interpolation
            * if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
            * if lanczos, a Lanczos interpolation over 8×8 pixel neighborhood
        backend: str opencv or python, default opencv
            resampling backend
            * if opencv, image will be resampled using opencv
            * if python, image will be resampled using pure python. slower and nearest neighbor only


        Returns
        -------
        Raster
            Resized
        """
        if backend == "opencv":
            return self.cv_resize(height, width, method)
        elif backend == "python":
            return self.py_resize(height, width)
        else:
            raise ValueError("Please choose between python or opencv for backend")

    def resample(
        self,
        resolution: Union[Tuple[float, float], List[float], float],
        method: str = "bilinear",
        backend: str = "opencv",
    ) -> "Raster":
        """Resample image into defined resolution

        Parameters
        -------
        resolution: tuple, list, float
            spatial resolution target
        method: str nearest or bicubic or bilinear or area or lanczos, default bilinear
            resampling method for opencv
            * if nearest, a nearest-neighbor interpolation
            * if bicubic, a bicubic interpolation over 4×4 pixel neighborhood
            * if bilinear, a bilinear interpolation
            * if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
            * if lanczos, a Lanczos interpolation over 8×8 pixel neighborhood
        backend: str opencv or python, default opencv
            resampling backend
            * if opencv, image will be resampled using opencv
            * if python, image will be resampled using pure python. slower and nearest neighbor only


        Returns
        -------
        Raster
            Resampled
        """
        if backend == "opencv":
            return self.cv_resample(resolution, method)
        elif backend == "python":
            return self.py_resample(resolution)
        else:
            raise ValueError("Please choose between python or opencv for backend")

    def cv_resize(self, height: int, width: int, method: str) -> "Raster":
        resized_y_resolution = self.y_extent / height
        resized_x_resolution = self.x_extent / width
        resized = cv2.resize(
            self.array, (height, width), interpolation=self.__cv2_resize_method[method]
        )
        return Raster(
            resized,
            (resized_x_resolution, resized_y_resolution),
            self.x_min,
            self.y_min,
        )

    def cv_resample(
        self, resolution: Union[Tuple[float, float], List[float], float], method: str
    ) -> "Raster":
        if isinstance(resolution, (float, int)):
            resampled_x_resolution = float(resolution)
            resampled_y_resolution = float(resolution)
        else:
            resampled_x_resolution = resolution[0]
            resampled_y_resolution = resolution[1]

        resampled_rows = round(self.y_extent / resampled_y_resolution)
        resampled_cols = round(self.x_extent / resampled_x_resolution)

        resampled = self.cv_resize(resampled_rows, resampled_cols, method)
        return resampled

    def py_resample(
        self, resolution: Union[Tuple[float, float], List[float], float]
    ) -> "Raster":
        """
        Resample raster using nearest neighbor
        Parameters
        -------
        resolution: tuple, list
            spatial resolution target

        Returns
        -------
        Raster
            Resampled
        """

        if isinstance(resolution, (float, int)):
            resampled_x_resolution = float(resolution)
            resampled_y_resolution = float(resolution)
        else:
            resampled_x_resolution = resolution[0]
            resampled_y_resolution = resolution[1]

        resampled_rows = round(self.y_extent / resampled_y_resolution)
        resampled_cols = round(self.x_extent / resampled_x_resolution)

        resampled_shape: Tuple[int, ...] = (resampled_rows, resampled_cols, self.layers)
        if self.layers == 1:
            resampled_shape = (resampled_rows, resampled_cols)

        resampled_array = np.zeros(
            resampled_rows * resampled_cols * self.layers, dtype=self.dtype
        ).reshape(resampled_shape)

        resampled_affine = Affine.translation(self.x_min, self.y_min) * Affine.scale(
            resampled_x_resolution, -resampled_y_resolution
        )

        for row in range(resampled_rows):
            for col in range(resampled_cols):
                x, y = rowcol2xy((row, col), resampled_affine)
                resampled_array[row, col] = self.xy_value(
                    x + (resampled_x_resolution / 2), y + (resampled_y_resolution / 2)
                )

        return Raster(
            resampled_array,
            (resampled_x_resolution, resampled_y_resolution),
            self.x_min,
            self.y_min,
        )

    def py_resize(self, height: int, width: int) -> "Raster":
        """
        Resize raster using nearest neighbor
        Parameters
        -------
        height: int
            raster height
        width: int
            raster width

        Returns
        -------
        Raster
            Resampled
        """
        resized_y_resolution = self.y_extent / height
        resized_x_resolution = self.x_extent / width

        resized_affine = Affine.translation(self.x_min, self.y_min) * Affine.scale(
            resized_x_resolution, -resized_y_resolution
        )

        resized_shape: Tuple[int, ...] = (height, width, self.layers)
        if self.layers == 1:
            resized_shape = (height, width)

        resized_array = np.zeros(
            height * width * self.layers, dtype=self.dtype
        ).reshape(resized_shape)

        for row in range(height):
            for col in range(width):
                x, y = rowcol2xy((row, col), resized_affine)
                resized_array[row, col] = self.xy_value(
                    x + (resized_x_resolution / 2), y + (resized_y_resolution / 2)
                )

        return Raster(
            resized_array,
            (resized_x_resolution, resized_y_resolution),
            self.x_min,
            self.y_min,
        )
