import os
import pandas as pd

def add_area_to_metadata(in_file, area_file, out_file, area_column_in, area_column_out):
    # Code smell: This function is very similar to 2_process/src/make_lake_metadata_augmented.add_elevation_from_surface_metadata

    lake_metadata = pd.read_csv(in_file)
    lake_areas = pd.read_csv(area_file)
    lake_areas_only = lake_areas.loc[:, ['site_id', area_column_in]]
    # Add area to metadata
    augmented = pd.merge(lake_metadata, lake_areas_only, how="inner", on='site_id')
    # Rename column to area
    augmented = augmented.rename({area_column_in: area_column_out}, axis='columns')

    destination_dir = os.path.dirname(out_file)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    augmented.to_csv(out_file, index=False)


if __name__ == '__main__':
    # Add lake area to metadata
    add_area_to_metadata(
        snakemake.input.lake_metadata,
        snakemake.input.area_metadata,
        snakemake.output.augmented_metadata,
        snakemake.params.area_column_in,
        snakemake.params.area_column_out
    )

