from typing import Dict

import click
import fiona
import rasterio
from tqdm.autonotebook import tqdm

from geosardine._geosardine import spatial_join, drape_geojson


@click.group()
def main():
    pass


@main.command("join-spatial")
@main.argument("target")
@main.argument("overlay")
def join_spatial(target: str, join: str) -> None:
    with fiona.open(target) as target_file, fiona.open(join) as join_file:
        driver: str = target_file.driver
        crs: Dict = target_file.crs
        joined_features, joined_schema = spatial_join(target_file, join_file)

    with fiona.open(
        target, "w", driver=driver, schema=joined_schema, crs=crs
    ) as out_file:
        click.echo("overwriting file...")
        out_file.writerecords(joined_features)
        click.echo("Done!")


@main.command("drape")
@main.argument("target")
@main.argument("raster")
def drape(target, raster):
    with fiona.open(target) as target_file, rasterio.open(raster) as raster_file:
        driver: str = target_file.driver
        crs: Dict = target_file.crs
        schema: Dict = target_file.schema
        draped_features = drape_geojson(target_file, raster_file)

    with fiona.open(target, "w", driver=driver, schema=schema, crs=crs) as out_file:
        click.echo("overwriting file...")
        for feature in tqdm(draped_features):
            out_file.writerecords(feature)
        click.echo("Done!")
