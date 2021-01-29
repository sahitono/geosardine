from operator import (
    add,
    floordiv,
    iadd,
    ifloordiv,
    imul,
    ipow,
    isub,
    itruediv,
    mul,
    pow,
    sub,
    truediv,
)
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.plot import reshape_as_image

from geosardine._geosardine import rowcol2xy, xy2rowcol
from geosardine._raster_numba import __nb_raster_calc__, nb_raster_ops
from geosardine._utility import save_raster


def __nb_raster_calc(
    raster_a: "Raster", raster_b: "Raster", operator: str
) -> np.ndarray:
    """Wrapper for Raster calculation per pixel using numba jit.

    Parameters
    ----------
    raster_a : Raster
        first raster
    raster_b : Raster
        second raster
    operator : str
        operator name

    Returns
    -------
    np.ndarray
        calculated raster
    """
    return __nb_raster_calc__(
        raster_a.array,
        raster_b.array,
        raster_a.transform,
        ~raster_b.transform,
        raster_a.no_data,
        raster_b.no_data,
        __nb_raster_ops[operator],
    )


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
    y_max : float, defaults to None
        top boundary of y-axis coordinate
    x_max : float, defaults to None
        right boundary of x-axis coordinate
    y_min : float, defaults to None
        bottom boundary of y-axis coordinate
    epsg : int, defaults to 4326
        EPSG code of reference system
    no_data : int or float, default None
        no data value

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
        no_data: Union[float, int] = -32767.0,
        transform: Optional[Affine] = None,
    ):
        if transform is None:
            if resolution is None and x_min is None and y_min is None:
                raise ValueError(
                    "Please define resolution and at least x minimum and y minimum"
                )

            if resolution is not None and x_min is None and y_max is None:
                raise ValueError("Please define x_min and y_max")

            if isinstance(resolution, float):
                self.resolution: Tuple[float, float] = (
                    resolution,
                    resolution,
                )
            elif isinstance(resolution, Iterable):
                self.resolution = (resolution[0], resolution[1])

            if (
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

            self.transform: Affine = Affine.translation(x_min, y_max) * Affine.scale(
                self.resolution[0], -self.resolution[1]
            )
        elif isinstance(transform, Affine):
            self.transform = transform
            self.resolution = (transform[0], abs(transform[4]))
        else:
            raise ValueError(
                "Please define affine parameter or resolution and xmin ymax"
            )

        self.epsg = epsg

        self.crs = CRS.from_epsg(epsg)
        self.no_data = no_data
        self.__check_validity()

    def __new__(cls, array: np.ndarray, *args, **kwargs) -> "Raster":
        return array.view(cls)

    def __getitem__(self, key: Union[int, Tuple[Any, ...], slice]) -> np.ndarray:
        return self.array.__getitem__(key)

    @classmethod
    def from_binary(
        cls,
        binary_file: str,
        shape: Tuple[int, ...],
        resolution: Union[Tuple[float, float], List[float], float],
        x_min: float,
        y_max: float,
        epsg: int = 4326,
        no_data: Union[float, int] = -32767.0,
        dtype: np.dtype = np.float32,
        shape_order: str = "hwc",
        *args,
        **kwargs,
    ) -> "Raster":
        """Convert binary grid into Raster

        Parameters
        -------
        binary_file : str
            location of binary grid file
        shape : tuple of int
            shape of binary grid.
        resolution : tuple of float, list of float or float
            pixel / grid spatial resolution
        x_min : float, defaults to None
            left boundary of x-axis coordinate
        y_max : float, defaults to None
            top boundary of y-axis coordinate
        epsg : int, defaults to 4326
            EPSG code of reference system
        no_data : int or float, default None
            no data value
        dtype : numpy.dtype, default numpy.float32
            data type of raster
        shape_order : str, default hwc
            shape ordering,
            * if default, height x width x channel


        Returns
        -------
        Raster
            raster shape will be in format height x width x channel / layer

        """

        _bin_array = np.fromfile(binary_file, dtype=dtype, *args, **kwargs).reshape(
            shape
        )

        if shape_order not in ("hwc", "hw"):
            c_index = shape_order.index("c")
            h_index = shape_order.index("h")
            w_index = shape_order.index("w")

            _bin_array = np.transpose(_bin_array, (h_index, w_index, c_index))

        return cls(_bin_array, resolution, x_min, y_max, epsg=epsg, no_data=no_data)

    @classmethod
    def from_rasterfile(cls, raster_file: str) -> "Raster":
        """Get raster from supported gdal raster file

        Parameters
        -------
        raster_file : str
            location of raser file

        Returns
        -------
        Raster
        """
        with rasterio.open(raster_file) as file:
            _raster = reshape_as_image(file.read())

        return cls(
            _raster,
            transform=file.transform,
            epsg=file.crs.to_epsg(),
            no_data=file.nodatavals[0],
        )

    @property
    def array(self) -> np.ndarray:
        """the numpy array of raster"""
        return self.__array__()

    @property
    def __transform(self) -> Tuple[float, ...]:
        return tuple(self.transform)

    @property
    def x_min(self) -> float:
        """minimum x-axis coordinate"""
        return self.__transform[2]

    @property
    def y_max(self) -> float:
        """maximum y-axis coordinate"""
        return self.__transform[5]

    @property
    def x_max(self) -> float:
        """maximum x-axis coordinate"""
        return self.__transform[2] + (self.resolution[0] * self.cols)

    @property
    def y_min(self) -> float:
        """minimum y-axis coordinate"""
        return self.__transform[5] - (self.resolution[1] * self.rows)

    @property
    def top(self) -> float:
        """top y-axis coordinate"""
        return self.y_max

    @property
    def left(self) -> float:
        """left x-axis coordinate"""
        return self.x_min

    @property
    def right(self) -> float:
        """right x-axis coordinate"""
        return self.x_max

    @property
    def bottom(self) -> float:
        """bottom y-axis coordinate"""
        return self.y_min

    @property
    def rows(self) -> int:
        """number of row, height"""
        return int(self.array.shape[0])

    @property
    def cols(self) -> int:
        """number of column, width"""
        return int(self.array.shape[1])

    @property
    def layers(self) -> int:
        """number of layer / channel"""
        _layers: int = 1
        if len(self.array.shape) > 2:
            _layers = self.array.shape[2]
        return _layers

    @property
    def x_extent(self) -> float:
        """width of raster in the map unit (degree decimal or meters)"""
        return self.x_max - self.x_min

    @property
    def y_extent(self) -> float:
        """height of raster in the map unit (degree decimal or meters)"""
        return self.y_max - self.y_min

    @property
    def is_projected(self) -> bool:
        """check crs is projected or not"""
        return self.crs.is_projected

    @property
    def is_geographic(self) -> bool:
        """check crs is geographic or not"""
        return self.crs.is_geographic

    def __check_validity(self) -> None:
        """Check geometry validity

        Raises
        ------
        ValueError
            x min, y min is greater than x max, y max
        ValueError
            x min is greater than x max
        ValueError
            y min is greater than y max
        """
        if self.x_extent < 0 and self.y_extent < 0:
            raise ValueError(
                "x min should be less than x max and y min should be less than y max"
            )
        elif self.x_extent < 0 and self.y_extent > 0:
            raise ValueError("x min should be less than x max")
        elif self.x_extent > 0 and self.y_extent < 0:
            raise ValueError("y min should be less than y max")

    def xy_value(self, x: float, y: float) -> Union[float, int, np.ndarray]:
        """Obtain pixel value by geodetic or projected coordinate

        Parameters
        ----------
        x : float
            x-axis coordinate
        y : float
            y-axis coordinate

        Returns
        -------
        Union[float, int, np.ndarray]
            pixel value
        """
        try:
            row, col = self.xy2rowcol(x, y)
            if row < 0 or col < 0:
                raise IndexError
            return self.array[row, col]
        except IndexError:
            raise IndexError(
                f"""
                {x},{y} is out of bound. 
                x_min={self.x_min} y_min={self.y_min} x_max={self.x_max} y_max={self.y_max}
                """
            )

    def rowcol2xy(
        self, row: int, col: int, offset: str = "center"
    ) -> Tuple[float, float]:
        """Convert image coordinate (row, col) to real world coordinate

        Parameters
        ----------
        row : int
        col : int
        offset : str

        Returns
        -------
        Tuple[float, float]
            X,Y coordinate in real world
        """
        return rowcol2xy((row, col), self.transform, offset=offset)

    def xy2rowcol(self, x: float, y: float, offset="center") -> Tuple[int, int]:
        """Convert real world coordinate to image coordinate (row, col)

        Parameters
        ----------
        x : float
        y : float

        Returns
        -------
        Tuple[int, int]
            row, column
        """
        _row, _col = xy2rowcol((x, y), self.transform,offset=offset, interpolate=True)
        return _row, _col

    def __raster_calc_by_pixel__(
        self,
        raster: "Raster",
        operator: Callable[[Any, Any], Any],
    ) -> np.ndarray:
        _raster = np.zeros(self.array.shape, dtype=self.array.dtype)
        for row in range(self.rows):
            for col in range(self.cols):
                try:
                    pixel_source = self.array[row, col]
                    pixel_target = raster.xy_value(*self.rowcol2xy(row, col))
                    if pixel_source != self.no_data and pixel_target != raster.no_data:
                        _raster[row, col] = operator(
                            pixel_source,
                            pixel_target,
                        )
                    else:
                        _raster[row, col] = self.no_data
                except IndexError:
                    _raster[row, col] = self.no_data
        return _raster

    def __nb_raster_calc(
        self, raster_a: "Raster", raster_b: "Raster", operator: str
    ) -> np.ndarray:
        """Wrapper for Raster calculation per pixel using numba jit.

        Parameters
        ----------
        raster_a : Raster
            first raster
        raster_b : Raster
            second raster
        operator : str
            operator name

        Returns
        -------
        np.ndarray
            calculated raster
        """
        if raster_b.layers != raster_a.layers:
            raise ValueError(
                f"""
                    Cant calculate between different layer shape.
                    first raster layer = {raster_a.layers}
                    second raster layer = {raster_b.layers}
                    """
            )

        _a = raster_a.array
        if self.layers == 1 and len(raster_a.shape) != 3:
            _a = raster_a.array.reshape(raster_a.rows, raster_a.cols, 1)

        _b = raster_b.array
        if self.layers == 1 and len(raster_b.shape) != 3:
            _b = raster_b.array.reshape(raster_b.rows, raster_b.cols, 1)

        out = __nb_raster_calc__(
            _a,
            _b,
            tuple(raster_a.transform),
            tuple(~raster_b.transform),
            raster_a.no_data,
            raster_b.no_data,
            nb_raster_ops[operator],
        )
        if out.shape != raster_a.shape:
            out = out.reshape(raster_a.shape)
        return out

    def __raster_calculation__(
        self,
        raster: Union[int, float, "Raster", np.ndarray],
        operator: Callable[[Any, Any], Any],
    ) -> "Raster":
        if not isinstance(raster, (int, float, Raster, np.ndarray)):
            raise ValueError(f"{type(raster)} unsupported data format")

        if isinstance(raster, Raster):
            if (
                raster.epsg == self.epsg
                and raster.resolution == self.resolution
                and raster.x_min == self.x_min
                and raster.y_min == self.y_min
                and raster.shape == self.shape
            ):
                _raster = operator(self.array, raster.array)
            else:
                # _raster = self.__raster_calc_by_pixel__(raster, operator)

                _raster = self.__nb_raster_calc(
                    self,
                    raster,
                    operator.__name__,
                )
        elif isinstance(raster, np.ndarray):
            _raster = operator(self.array, raster)
        else:
            _raster = operator(self.array, raster)

        return Raster(_raster, self.resolution, self.x_min, self.y_max, epsg=self.epsg)

    def __sub__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, sub)

    def __add__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, add)

    def __mul__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, mul)

    def __truediv__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, truediv)

    def __floordiv__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, floordiv)

    def __pow__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, pow)

    def __iadd__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, iadd)

    def __itruediv__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, itruediv)

    def __ifloordiv__(
        self, raster: Union[int, float, "Raster", np.ndarray]
    ) -> "Raster":
        return self.__raster_calculation__(raster, ifloordiv)

    def __imul__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, imul)

    def __isub__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, isub)

    def __ipow__(self, raster: Union[int, float, "Raster", np.ndarray]) -> "Raster":
        return self.__raster_calculation__(raster, ipow)

    def __iter__(self) -> Generator[Any, None, None]:
        _iter_shape: Union[Tuple[int, int], int] = (self.rows * self.cols, self.layers)
        if self.layers == 1:
            _iter_shape = self.rows * self.cols
        _iter = self.array.reshape(_iter_shape)
        for i in range(10):
            yield _iter[i]

    def save(self, file_name: str, compress: bool = False) -> None:
        """Save raster as geotiff

        Parameters
        ----------
        file_name : str
            output filename
        """
        save_raster(
            file_name,
            self.array,
            self.crs,
            affine=self.transform,
            nodata=self.no_data,
            compress=compress,
        )

    def resize(
        self, height: int, width: int, method: str = "bilinear", backend: str = "opencv"
    ) -> "Raster":
        """Resize raster into defined height and width

        Parameters
        -------
        height: int
            height defined
        width: int
            width defined
        method: str nearest or bicubic or bilinear or area or lanczos, default bilinear
            resampling method for opencv  <br/>
            * if nearest, a nearest-neighbor interpolation  <br/>
            * if bicubic, a bicubic interpolation over 4×4 pixel neighborhood  <br/>
            * if bilinear, a bilinear interpolation  <br/>
            * if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.  <br/>
            * if lanczos, a Lanczos interpolation over 8×8 pixel neighborhood
        backend: str opencv or python, default opencv
            resampling backend  <br/>
            * if opencv, image will be resampled using opencv  <br/>
            * if python, image will be resampled using pure python. slower and nearest neighbor only


        Returns
        -------
        Raster
            Resized
        """
        if backend == "opencv":
            return self.__cv_resize(height, width, method)
        elif backend == "python":
            return self.__py_resize(height, width)
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
            resampling method for opencv  <br/>
            * if nearest, a nearest-neighbor interpolation  <br/>
            * if bicubic, a bicubic interpolation over 4×4 pixel neighborhood  <br/>
            * if bilinear, a bilinear interpolation  <br/>
            * if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.  <br/>
            * if lanczos, a Lanczos interpolation over 8×8 pixel neighborhood
        backend: str opencv or python, default opencv
            resampling backend  <br/>
            * if opencv, image will be resampled using opencv  <br/>
            * if python, image will be resampled using pure python. slower and nearest neighbor only


        Returns
        -------
        Raster
            Resampled
        """
        if backend == "opencv":
            return self.__cv_resample(resolution, method)
        elif backend == "python":
            return self.__py_resample(resolution)
        else:
            raise ValueError("Please choose between python or opencv for backend")

    def __cv_resize(self, height: int, width: int, method: str) -> "Raster":
        resized_y_resolution = self.y_extent / height
        resized_x_resolution = self.x_extent / width
        resized = cv2.resize(
            self.array, (width, height), interpolation=self.__cv2_resize_method[method]
        )
        return Raster(
            resized,
            (resized_x_resolution, resized_y_resolution),
            self.x_min,
            self.y_max,
            epsg=self.epsg,
        )

    def __cv_resample(
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

        resampled = self.__cv_resize(resampled_rows, resampled_cols, method)
        return resampled

    def __py_resample(
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
            self.y_max,
            epsg=self.epsg,
        )

    def __py_resize(self, height: int, width: int) -> "Raster":
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
            self.y_max,
            epsg=self.epsg,
        )
