from typing import Dict

import click
import fiona
import rasterio
from tqdm.autonotebook import tqdm

from geosardine._geosardine import spatial_join, drape_geojson


@click.group()
def main():
    """GeoSardine CLI - Spatial operations extend fiona and rasterio"""
    pass


@main.command("info")
def info():
    """Get supported format"""
    supported_format = ["First band of GTiff"]
    for driver, supported in fiona.supported_drivers.items():
        if supported == "raw" or supported == "rw":
            supported_format.append(driver)

    print(f"Supported vector : {', '.join(supported_format)}")


@main.command("join-spatial")
@click.option("--target", help="vector file target")
@click.option("--overlay", help="vector file which data will be joined")
def join_spatial(target: str, join: str) -> None:
    """Join attribute by location"""
    with fiona.open(target) as target_file, fiona.open(join) as join_file:
        driver: str = target_file.driver
        crs: Dict = target_file.crs
        joined_features, joined_schema = spatial_join(target_file, join_file)

    with fiona.open(
        target, "w", driver=driver, schema=joined_schema, crs=crs
    ) as out_file:
        print("overwriting file...")
        for joined_feature in joined_features:
            out_file.write(joined_feature)
        print("Done!")


@main.command("drape")
@click.option("--target", help="any OGR supported vector data")
@click.option("--raster", help="any GDAL supported vector data")
def drape(target, raster):
    """
    Drape vector to raster to obtain height value
    """
    with fiona.open(target) as target_file, rasterio.open(raster) as raster_file:
        driver: str = target_file.driver
        crs: Dict = target_file.crs
        schema: Dict = target_file.schema
        draped_features = drape_geojson(target_file, raster_file)

    with fiona.open(target, "w", driver=driver, schema=schema, crs=crs) as out_file:
        print("overwriting file...")
        for feature in tqdm(draped_features):
            out_file.write(feature)
        print("Done!")


if __name__ == "__main__":
    main()
