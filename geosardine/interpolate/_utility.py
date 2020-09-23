from typing import Optional, Tuple

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
    ):
        self.array = array
        self.crs = crs
        self.extent = extent
        self.affine: Affine = calc_affine(coordinates)
        if extent is None:
            self.extent = calc_extent(coordinates)
        del coordinates

    def save(self, location) -> None:
        save_raster(location, self.array, affine=self.affine, crs=self.crs)
