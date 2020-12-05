from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from affine import Affine
from geosardine._utility import calc_affine, calc_extent, save_raster
from rasterio.crs import CRS


class InterpolationResult:
    """
    Class to interpolation result

    Attributes
    ----------
    array : numpy array
        array of interpolated value.
    coordinates : numpy array
        coordinate array of interpolated value  <br/>
        each pixel / grid is x and y or longitude and latitude  <br/>
    crs : `rasterio.crs.CRS`
        crs of interpolated value
    extent : tuple
        extent of interpolated  <br/>
        x min, y min, x max, y max
    source : pathlib.Path, default None
        source file location  <br/>
        * if None, there is no source file  <br/>
        * if str, location of point file
    """

    def __init__(
        self,
        array: np.ndarray,
        coordinates: np.ndarray,
        crs: CRS,
        extent: Optional[Tuple[float, float, float, float]] = None,
        source: Optional[Union[str, Path]] = None,
    ):
        """
        Constructs interpolation result
        Parameters
        ----------
        array : numpy array
            array of interpolated value
        coordinates : numpy array
            coordinate array of interpolated value <br/>
            each pixel / grid is x and y or longitude and latitude
        crs : `rasterio.crs.CRS`
            crs of interpolated value
        extent : tuple, default None
            extent of interpolated <br/>
            x min, y min, x max, y max <br/>
            * if None, extent will be calculated from coordinate <br/>
            * if tuple, extent will be same as input
        source : str, pathlib.Path, default None
            source file location <br/>
            * if None, there is no source file <br/>
            * if str or `pathlib.Path`, location of point file
        """
        self.array = array
        self.crs = crs
        self.extent = extent
        self.source = source
        self.output: Optional[Path] = None
        self.affine: Affine = calc_affine(coordinates)
        if extent is None:
            self.extent = calc_extent(coordinates)
        del coordinates

        if type(source) == str:
            self.source = Path(source)

    def save(self, location: Optional[Union[str, Path]] = None) -> None:
        """
        save interpolated array as geotif
        Parameters
        ----------
        location : str, pathlib.Path, default None
            output location  <br/>
            * if None, tiff will be saved in the same directory and same name as source
                will only work if source is not None  <br/>
            * if str or pathlib.Path, tiff will be saved in there  <br/>

        """
        if self.source is None and location is None:
            raise ValueError("Please provide output location")

        self.output = location
        if self.source is not None and location is None:
            self.output = self.source.parent / f"{self.source.stem}.tif"
            print(f"OUTPUT is not specified, will be saved as {self.output}")

        save_raster(self.output, self.array, affine=self.affine, crs=self.crs)
