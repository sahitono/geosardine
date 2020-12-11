# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- add no data when saving raster
- documentation for Raster attributes

### Changed

- py resample, resize and opencv resample, resize turn into private method. To prevent confusion, less abstraction.

### Removed

- remove import numba in raster.py

## [0.9.3] - 2020-12-10

### Added

- upper, left, right, bottom boundary

### Changed

- minimum argument needed is x_min, y_max, resolution or transform
- simpler conditional statement in raster init
- x_min, x_max, y_max,y_min is computed properties

### Fixed

- save raster return exact layer not +1 layer
- return no data if pixel out of bound for raster calculation
- raise error when using xy_value and the row col index is negative

## [0.9.2] - 2020-12-09
### Added
- Add numpy.ndarray type in raster calculation
- Raster.xy_value documentation
- Add pytest to check affine parameter
  
### Changed
- Show error if xy_value result is out of bound
- Change y_min into y_max in raster calculation, resize and resample
  
### Fixed
- Fix y_min attribute in raster calculation, resize and resample

## [0.9.1] - 2020-12-09
### Changed
- Change y_min into y_max in creating Affine

## [0.9.0] - 2020-12-08
### Added
- resize and resample using opencv for faster performance
- check validity, y_min should be less than y_max and x_min should be less than x_max
- add no data value
- return no data if any pixel is no data (raster calculation)

### Changed
- Use x_min and y_max as minimum input argument to fit upper left extent
- Refactor raster calculation per pixel to reduce cognitive complexity
- Check array shape as a tuple

### Fixed
- Remove pip check because different opencv name in pypi and anaconda

### Removed
- Remove shape check per element

## [0.8.0] - 2020-12-05
### Added
- Raster wrapper
- Support multiple layer in raster calculation