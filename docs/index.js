URLS=[
"index.html",
"interpolate/index.html",
"raster.html"
];
INDEX=[
{
"ref":"geosardine",
"url":0,
"doc":"Spatial operations extend fiona and rasterio. Collection of spatial operation which i occasionally use written in python: - Interpolation with IDW (Inverse Distance Weighting) Shepard - Drape vector to raster - Spatial join between two vector - Raster wrapper, for better experience. ie: math operation between two raster, resize and resample"
},
{
"ref":"geosardine.rowcol2xy",
"url":0,
"doc":"Convert image coordinate to geographic coordinate Parameters      row_col : tuple, list image coordinate in row, column affine : Affine affine parameter from rasterio.transform or create it with affine.Affine https: pypi.org/project/affine/ Returns    - tuple 2d geographic or projected coordinate",
"func":1
},
{
"ref":"geosardine.xy2rowcol",
"url":0,
"doc":"Convert geographic coordinate to image coordinate Parameters      xy : tuple, list 2d geographic or projected coordinate affine : Affine affine parameter from rasterio.transform or create it with affine.Affine https: pypi.org/project/affine/ interpolate : bool, default True choose to interpolate or not  if True, value will be interpolated from nearest value  if False, value will be obtained from exact row and column Returns    - tuple row, column",
"func":1
},
{
"ref":"geosardine.drape2raster",
"url":0,
"doc":"Find Z of 2D coordinate Parameters      xy : tuple, list, numpy array 2D coordinate x,y dsm_array : numpy array height array affine : Affine affine parameter from rasterio.transform or create it with affine.Affine https: pypi.org/project/affine/ interpolate : bool, default True choose to interpolate or not  if True, value will be interpolated from nearest value  if False, value will be obtained from exact row and column no_data : float, int, default -32767 value for pixel with no data Returns    - tuple 3D coordinate",
"func":1
},
{
"ref":"geosardine.spatial_join",
"url":0,
"doc":"Join attribute from 2 vector by location. Parameters      target : fiona.Collection vector target which you want to be joined join : fiona.Collection vector which data wont to be obtained Returns    - dict geojson",
"func":1
},
{
"ref":"geosardine.drape_shapely",
"url":0,
"doc":"Drape with shapely geometry as input Parameters      geometry : shapely polygon, shapely linestring vector data as shapely object, currently only support polygon or linestring raster : rasterio.io.DatasetReader rasterio reader of raster file interpolate : bool, default True choose to interpolate or not  if True, value will be interpolated from nearest value  if False, value will be obtained from exact row and column Returns    - shapely.Polygon or shapely.LineString",
"func":1
},
{
"ref":"geosardine.drape_geojson",
"url":0,
"doc":"Drape with geojson as input, fiona uses geojson as interface. Parameters      features : Iterable[Dict], fiona.Collection vector as geojson raster : rasterio.io.DatasetReader rasterio reader of raster file interpolate : bool, default True choose to interpolate or not  if True, value will be interpolated from nearest value  if False, value will be obtained from exact row and column Yields    - dict geojson",
"func":1
},
{
"ref":"geosardine.harvesine_distance",
"url":0,
"doc":"Calculate distance in ellipsoid by harvesine method faster, less accurate Parameters      long_lat1 : tuple, list, numpy array first point coordinate in longitude, latitude long_lat2 : tuple, list, numpy array second point coordinate in longitude, latitude Returns    - float distance Notes    - https: rafatieppo.github.io/post/2018_07_27_idw2pyr/",
"func":1
},
{
"ref":"geosardine.vincenty_distance",
"url":0,
"doc":"Calculate distance in ellipsoid by vincenty method slower, more accurate Parameters      long_lat1 : tuple, list first point coordinate in longitude, latitude long_lat2 : tuple, list second point coordinate in longitude, latitude Returns    - distance Notes    - https: www.johndcook.com/blog/2018/11/24/spheroid-distance/",
"func":1
},
{
"ref":"geosardine.Raster",
"url":0,
"doc":"Construct Raster from numpy array with spatial information. Support calculation between different raster Parameters      array : numpy array array of raster resolution : tuple, list, default None spatial resolution x_min : float, defaults to None left boundary of x-axis coordinate y_max : float, defaults to None top boundary of y-axis coordinate x_max : float, defaults to None right boundary of x-axis coordinate y_min : float, defaults to None bottom boundary of y-axis coordinate epsg : int, defaults to 4326 EPSG code of reference system no_data : int or float, default None no data value Examples     >>> from geosardine import Raster >>> raster = Raster(np.ones(18, dtype=np.float32).reshape(3, 3, 2), resolution=0.4, x_min=120, y_max=0.7) >>> print(raster)  [1. 1.] [1. 1.] [1. 1.  1. 1.] [1. 1.] [1. 1.  1. 1.] [1. 1.] [1. 1. ] Raster can be resampled like this. (0.2,0.2) is the result's spatial resolution >>> resampled = raster.resample 0.2,0.2 >>> print(resampled.shape, resampled.resolution) (6, 6, 2) (0.2, 0.2) Raster can be resized >>> resized = raster.resize(height=16, width=16) >>> print(resized.shape, resized.resolution) (16, 16, 2) (0.07500000000000018, 0.07500000000000001)"
},
{
"ref":"geosardine.Raster.from_binary",
"url":0,
"doc":"Convert binary grid into Raster Parameters    - binary_file : str location of binary grid file shape : tuple of int shape of binary grid. resolution : tuple of float, list of float or float pixel / grid spatial resolution x_min : float, defaults to None left boundary of x-axis coordinate y_max : float, defaults to None top boundary of y-axis coordinate epsg : int, defaults to 4326 EPSG code of reference system no_data : int or float, default None no data value dtype : numpy.dtype, default numpy.float32 data type of raster shape_order : str, default hwc shape ordering,  if default, height x width x channel Returns    - Raster raster shape will be in format height x width x channel / layer",
"func":1
},
{
"ref":"geosardine.Raster.from_rasterfile",
"url":0,
"doc":"Get raster from supported gdal raster file Parameters    - raster_file : str location of raser file Returns    - Raster",
"func":1
},
{
"ref":"geosardine.Raster.array",
"url":0,
"doc":"the numpy array of raster"
},
{
"ref":"geosardine.Raster.x_min",
"url":0,
"doc":"minimum x-axis coordinate"
},
{
"ref":"geosardine.Raster.y_max",
"url":0,
"doc":"maximum y-axis coordinate"
},
{
"ref":"geosardine.Raster.x_max",
"url":0,
"doc":"maximum x-axis coordinate"
},
{
"ref":"geosardine.Raster.y_min",
"url":0,
"doc":"minimum y-axis coordinate"
},
{
"ref":"geosardine.Raster.top",
"url":0,
"doc":"top y-axis coordinate"
},
{
"ref":"geosardine.Raster.left",
"url":0,
"doc":"left x-axis coordinate"
},
{
"ref":"geosardine.Raster.right",
"url":0,
"doc":"right x-axis coordinate"
},
{
"ref":"geosardine.Raster.bottom",
"url":0,
"doc":"bottom y-axis coordinate"
},
{
"ref":"geosardine.Raster.rows",
"url":0,
"doc":"number of row, height"
},
{
"ref":"geosardine.Raster.cols",
"url":0,
"doc":"number of column, width"
},
{
"ref":"geosardine.Raster.layers",
"url":0,
"doc":"number of layer / channel"
},
{
"ref":"geosardine.Raster.x_extent",
"url":0,
"doc":"width of raster in the map unit (degree decimal or meters)"
},
{
"ref":"geosardine.Raster.y_extent",
"url":0,
"doc":"height of raster in the map unit (degree decimal or meters)"
},
{
"ref":"geosardine.Raster.is_projected",
"url":0,
"doc":"check crs is projected or not"
},
{
"ref":"geosardine.Raster.is_geographic",
"url":0,
"doc":"check crs is geographic or not"
},
{
"ref":"geosardine.Raster.xy_value",
"url":0,
"doc":"Obtain pixel value by geodetic or projected coordinate Parameters      x : float x-axis coordinate y : float y-axis coordinate Returns    - Union[float, int, np.ndarray] pixel value",
"func":1
},
{
"ref":"geosardine.Raster.rowcol2xy",
"url":0,
"doc":"Convert image coordinate (row, col) to real world coordinate Parameters      row : int col : int Returns    - Tuple[float, float] X,Y coordinate in real world",
"func":1
},
{
"ref":"geosardine.Raster.xy2rowcol",
"url":0,
"doc":"Convert real world coordinate to image coordinate (row, col) Parameters      x : float y : float Returns    - Tuple[int, int] row, column",
"func":1
},
{
"ref":"geosardine.Raster.save",
"url":0,
"doc":"Save raster as geotiff Parameters      file_name : str output filename",
"func":1
},
{
"ref":"geosardine.Raster.resize",
"url":0,
"doc":"Resize raster into defined height and width Parameters    - height: int height defined width: int width defined method: str nearest or bicubic or bilinear or area or lanczos, default bilinear resampling method for opencv   if nearest, a nearest-neighbor interpolation   if bicubic, a bicubic interpolation over 4\u00d74 pixel neighborhood   if bilinear, a bilinear interpolation   if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire\u2019-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.   if lanczos, a Lanczos interpolation over 8\u00d78 pixel neighborhood backend: str opencv or python, default opencv resampling backend   if opencv, image will be resampled using opencv   if python, image will be resampled using pure python. slower and nearest neighbor only Returns    - Raster Resized",
"func":1
},
{
"ref":"geosardine.Raster.resample",
"url":0,
"doc":"Resample image into defined resolution Parameters    - resolution: tuple, list, float spatial resolution target method: str nearest or bicubic or bilinear or area or lanczos, default bilinear resampling method for opencv   if nearest, a nearest-neighbor interpolation   if bicubic, a bicubic interpolation over 4\u00d74 pixel neighborhood   if bilinear, a bilinear interpolation   if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire\u2019-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.   if lanczos, a Lanczos interpolation over 8\u00d78 pixel neighborhood backend: str opencv or python, default opencv resampling backend   if opencv, image will be resampled using opencv   if python, image will be resampled using pure python. slower and nearest neighbor only Returns    - Raster Resampled",
"func":1
},
{
"ref":"geosardine.interpolate",
"url":1,
"doc":""
},
{
"ref":"geosardine.interpolate.idw",
"url":1,
"doc":"create interpolated raster from point by using Inverse Distance Weighting (Shepard) Parameters      points : numpy array, str list of points coordinate as numpy array or address of vector file  i.e shapefile or geojson   if numpy array, then value input needed   if str, then value is not needed instead will be created from file value : numpy array list of points value as numpy array, not needed if vector file used as input spatial_res : tuple or list of float spatial resolution in x and y axis column_name : str, default None column name needed to obtain value from attribute data of vector file   If str, value will be read from respective column name   If None, first column will be used as value epsg : int, default 4326 EPSG code of reference system   If 4326, WGS 1984 geographic system   If int, epsg will be parsed longlat_distance: str harvesine or vincenty, default harvesine method used to calculate distance in spherical / ellipsoidal   If harvesine, calculation will be faster but less accurate   If vincenty, calculation will be slower but more accurate extent: tuple of float, default None how wide the raster will be   If None, extent will be calculated from points input   If tuple of float, user input of extent will be used power: float, default 2 how smooth the interpolation will be distance_limit: float, default 0 maximum distance to be interpolated, can't be negative Returns    - InterpolationResult Examples     >>> xy = np.array( 106.8358, -6.585 ],  . [106.6039, -6.7226],  . [106.7589, -6.4053],  . [106.9674, -6.7092],  . [106.7956, -6.5988]  . ]) >>> values = np.array([132., 127., 37., 90., 182.]) >>> idw(xy, values, spatial_res=(0.01,0.01), epsg=4326) >>> print(interpolated.array)  88.63769859 86.24219616 83.60463194  . 101.98185127 103.37001289 104.54621272] [ 90.12053232 87.79279317 85.22030848  . 103.77118852 105.01425289 106.05302554] [ 91.82987695 89.60855271 87.14722258  . 105.70090081 106.76928067 107.64635337]  . [127.21214817 127.33208302 127.53878268  . 97.80436475 94.96247196 93.12113458] [127.11315081 127.18465002 127.33444124  . 95.86455668 93.19212577 91.51135399] [127.0435062 127.0827023 127.19214624  . 94.80175756 92.30685734 90.75707134 ",
"func":1
},
{
"ref":"geosardine.interpolate.idw_single",
"url":1,
"doc":"Parameters      point : list list of single point to be interpolated known_coordinates : numpy array list of points coordinate as numpy array known_value: numpy array list of points value as numpy array, not needed if vector file used as input epsg : int, default 4326 EPSG code of reference system   If 4326, WGS 1984 geographic system   If int, epsg will be parsed longlat_distance: str harvesine or vincenty, default harvesine method used to calculate distance in spherical / ellipsoidal   If harvesine, calculation will be faster but less accurate   If vincenty, calculation will be slower but more accurate power: float, default 2 how smooth the interpolation will be distance_limit: float, default 0 maximum distance to be interpolated, can't be negative Returns    - float interpolated value Examples     >>> from geosardine.interpolate import idw_single >>> result = idw_single(  . [860209, 9295740],  . np.array( 767984, 9261620], [838926, 9234594 ),  . np.array( 101.1, 102.2 ),  . epsg=32748,  . distance_limit=0  . ) >>> print(result) 101.86735169471324",
"func":1
},
{
"ref":"geosardine.interpolate.InterpolationResult",
"url":1,
"doc":"Class to interpolation result Attributes      array : numpy array array of interpolated value. coordinates : numpy array coordinate array of interpolated value  each pixel / grid is x and y or longitude and latitude  crs :  rasterio.crs.CRS crs of interpolated value extent : tuple extent of interpolated  x min, y min, x max, y max source : pathlib.Path, default None source file location   if None, there is no source file   if str, location of point file Constructs interpolation result Parameters      array : numpy array array of interpolated value coordinates : numpy array coordinate array of interpolated value  each pixel / grid is x and y or longitude and latitude crs :  rasterio.crs.CRS crs of interpolated value extent : tuple, default None extent of interpolated  x min, y min, x max, y max   if None, extent will be calculated from coordinate   if tuple, extent will be same as input source : str, pathlib.Path, default None source file location   if None, there is no source file   if str or  pathlib.Path , location of point file"
},
{
"ref":"geosardine.interpolate.InterpolationResult.save",
"url":1,
"doc":"save interpolated array as geotif Parameters      location : str, pathlib.Path, default None output location   if None, tiff will be saved in the same directory and same name as source will only work if source is not None   if str or pathlib.Path, tiff will be saved in there  ",
"func":1
},
{
"ref":"geosardine.raster",
"url":2,
"doc":""
},
{
"ref":"geosardine.raster.Raster",
"url":2,
"doc":"Construct Raster from numpy array with spatial information. Support calculation between different raster Parameters      array : numpy array array of raster resolution : tuple, list, default None spatial resolution x_min : float, defaults to None left boundary of x-axis coordinate y_max : float, defaults to None top boundary of y-axis coordinate x_max : float, defaults to None right boundary of x-axis coordinate y_min : float, defaults to None bottom boundary of y-axis coordinate epsg : int, defaults to 4326 EPSG code of reference system no_data : int or float, default None no data value Examples     >>> from geosardine import Raster >>> raster = Raster(np.ones(18, dtype=np.float32).reshape(3, 3, 2), resolution=0.4, x_min=120, y_max=0.7) >>> print(raster)  [1. 1.] [1. 1.] [1. 1.  1. 1.] [1. 1.] [1. 1.  1. 1.] [1. 1.] [1. 1. ] Raster can be resampled like this. (0.2,0.2) is the result's spatial resolution >>> resampled = raster.resample 0.2,0.2 >>> print(resampled.shape, resampled.resolution) (6, 6, 2) (0.2, 0.2) Raster can be resized >>> resized = raster.resize(height=16, width=16) >>> print(resized.shape, resized.resolution) (16, 16, 2) (0.07500000000000018, 0.07500000000000001)"
},
{
"ref":"geosardine.raster.Raster.from_binary",
"url":2,
"doc":"Convert binary grid into Raster Parameters    - binary_file : str location of binary grid file shape : tuple of int shape of binary grid. resolution : tuple of float, list of float or float pixel / grid spatial resolution x_min : float, defaults to None left boundary of x-axis coordinate y_max : float, defaults to None top boundary of y-axis coordinate epsg : int, defaults to 4326 EPSG code of reference system no_data : int or float, default None no data value dtype : numpy.dtype, default numpy.float32 data type of raster shape_order : str, default hwc shape ordering,  if default, height x width x channel Returns    - Raster raster shape will be in format height x width x channel / layer",
"func":1
},
{
"ref":"geosardine.raster.Raster.from_rasterfile",
"url":2,
"doc":"Get raster from supported gdal raster file Parameters    - raster_file : str location of raser file Returns    - Raster",
"func":1
},
{
"ref":"geosardine.raster.Raster.array",
"url":2,
"doc":"the numpy array of raster"
},
{
"ref":"geosardine.raster.Raster.x_min",
"url":2,
"doc":"minimum x-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.y_max",
"url":2,
"doc":"maximum y-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.x_max",
"url":2,
"doc":"maximum x-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.y_min",
"url":2,
"doc":"minimum y-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.top",
"url":2,
"doc":"top y-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.left",
"url":2,
"doc":"left x-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.right",
"url":2,
"doc":"right x-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.bottom",
"url":2,
"doc":"bottom y-axis coordinate"
},
{
"ref":"geosardine.raster.Raster.rows",
"url":2,
"doc":"number of row, height"
},
{
"ref":"geosardine.raster.Raster.cols",
"url":2,
"doc":"number of column, width"
},
{
"ref":"geosardine.raster.Raster.layers",
"url":2,
"doc":"number of layer / channel"
},
{
"ref":"geosardine.raster.Raster.x_extent",
"url":2,
"doc":"width of raster in the map unit (degree decimal or meters)"
},
{
"ref":"geosardine.raster.Raster.y_extent",
"url":2,
"doc":"height of raster in the map unit (degree decimal or meters)"
},
{
"ref":"geosardine.raster.Raster.is_projected",
"url":2,
"doc":"check crs is projected or not"
},
{
"ref":"geosardine.raster.Raster.is_geographic",
"url":2,
"doc":"check crs is geographic or not"
},
{
"ref":"geosardine.raster.Raster.xy_value",
"url":2,
"doc":"Obtain pixel value by geodetic or projected coordinate Parameters      x : float x-axis coordinate y : float y-axis coordinate Returns    - Union[float, int, np.ndarray] pixel value",
"func":1
},
{
"ref":"geosardine.raster.Raster.rowcol2xy",
"url":2,
"doc":"Convert image coordinate (row, col) to real world coordinate Parameters      row : int col : int Returns    - Tuple[float, float] X,Y coordinate in real world",
"func":1
},
{
"ref":"geosardine.raster.Raster.xy2rowcol",
"url":2,
"doc":"Convert real world coordinate to image coordinate (row, col) Parameters      x : float y : float Returns    - Tuple[int, int] row, column",
"func":1
},
{
"ref":"geosardine.raster.Raster.save",
"url":2,
"doc":"Save raster as geotiff Parameters      file_name : str output filename",
"func":1
},
{
"ref":"geosardine.raster.Raster.resize",
"url":2,
"doc":"Resize raster into defined height and width Parameters    - height: int height defined width: int width defined method: str nearest or bicubic or bilinear or area or lanczos, default bilinear resampling method for opencv   if nearest, a nearest-neighbor interpolation   if bicubic, a bicubic interpolation over 4\u00d74 pixel neighborhood   if bilinear, a bilinear interpolation   if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire\u2019-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.   if lanczos, a Lanczos interpolation over 8\u00d78 pixel neighborhood backend: str opencv or python, default opencv resampling backend   if opencv, image will be resampled using opencv   if python, image will be resampled using pure python. slower and nearest neighbor only Returns    - Raster Resized",
"func":1
},
{
"ref":"geosardine.raster.Raster.resample",
"url":2,
"doc":"Resample image into defined resolution Parameters    - resolution: tuple, list, float spatial resolution target method: str nearest or bicubic or bilinear or area or lanczos, default bilinear resampling method for opencv   if nearest, a nearest-neighbor interpolation   if bicubic, a bicubic interpolation over 4\u00d74 pixel neighborhood   if bilinear, a bilinear interpolation   if area, resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire\u2019-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.   if lanczos, a Lanczos interpolation over 8\u00d78 pixel neighborhood backend: str opencv or python, default opencv resampling backend   if opencv, image will be resampled using opencv   if python, image will be resampled using pure python. slower and nearest neighbor only Returns    - Raster Resampled",
"func":1
}
]