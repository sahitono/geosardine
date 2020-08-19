## Geo-Sardine

Collection of spatial operation which i occasionally use 

### Setup
install it with pip
```pip install --pre geosardine```

### How to use it
```python
import geosardine as dine
import rasterio
import fiona

with rasterio.open("/home/user/data.tif") as raster, fiona.open("/home/user/data.shp") as vector:
    draped = dine.drape_geojson(vector, raster)
    joined = dine.spatial_join(vector, raster) 
```

### Geosardine CLI
You can use it through terminal or command prompt by calling **dine**

```
$ dine --help
Usage: dine [OPTIONS] COMMAND [ARGS]...

  GeoSardine CLI

Options:
  --help  Show this message and exit.

Commands:
  drape         Drape vector to raster to obtain height value
  info          Get supported format
  join-spatial  Join attribute by location
```

### License
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
