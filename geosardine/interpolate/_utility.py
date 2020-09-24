from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from geosardine._utility import calc_affine, calc_extent, save_raster


class InterpolationResults:
    def __init__(
        self,
        array: np.ndarray,
        coordinates: np.ndarray,
        crs: CRS,
        extent: Optional[Tuple[float, float, float, float]] = None,
        source: Optional[Union[str, Path]] = None,
    ):
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
        if self.source is None and location is None:
            raise ValueError("Please provide output location")

        self.output = location
        if self.source is not None and location is None:
            self.output = self.source.parent / f"{self.source.stem}.tif"
            print(f"OUTPUT is not specified, will be saved as {self.output}")

        save_raster(self.output, self.array, affine=self.affine, crs=self.crs)
