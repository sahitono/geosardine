from pathlib import Path
from typing import Dict

import click
import fiona
import rasterio
from tqdm.autonotebook import tqdm

from geosardine._geosardine import drape_geojson, spatial_join
from geosardine.interpolate import idw


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
@click.argument("target", type=click.Path(exists=True))
@click.argument("overlay", type=click.Path(exists=True))
def join_spatial(target: str, join: str) -> None:
    """Join attribute to TARGET from OVERLAY's attribute by location"""
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
@click.argument("target", type=click.Path(exists=True))
@click.argument("raster", type=click.Path(exists=True))
def drape(target, raster):
    """
    Drape vector TARGET to RASTER to obtain height value
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


@main.command("idw")
@click.argument("points", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None, help="output location")
@click.argument("resolution", type=float)
@click.option("--column", type=str, default=None, help="column name of points")
@click.option("--distance_limit", type=float, default=0.0, help="column name of points")
def idw_cli(points: Path, output, resolution, column, distance_limit):
    """
    Create raster from POINTS with defined RESOLUTION
    by Inverse Distance Weighting interpolation
    """
    points = Path(points)
    print("Running...")
    interpolation = idw(
        points,
        (resolution, resolution),
        column_name=column,
        distance_limit=distance_limit,
    )
    if interpolation is not None:
        out_filename = output
        if output is None:
            out_filename = points.parent.absolute().joinpath(f"{points.stem}_idw.tif")
        interpolation.save(out_filename)
    else:
        raise ValueError("Result empty")


if __name__ == "__main__":
    main()
