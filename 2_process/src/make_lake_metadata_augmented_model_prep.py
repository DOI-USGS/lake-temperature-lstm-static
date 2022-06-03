import os
import pandas as pd

def add_area_from_file(in_file, area_file, out_file, area_column_in, area_column_out):
    """
    Add column with lake surface areas to metadata and save to new file.
    
    Code smell: This function is similar to
    2_process/src/make_lake_metadata_augmented.add_elevation_from_file

    :param in_file: Filename of csv to read existing lake metadata from
    :param area_file: Filename of csv file to read lake area data from
    :param out_file: Filename of csv to save metadata augmented with area to
    :param area_column_in: Name of column with areas in in_file
    :param area_column_out: What to name column with areas in out_file

    """

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
    add_area_from_file(
        snakemake.input.lake_metadata,
        snakemake.input.area_metadata,
        snakemake.output.augmented_metadata,
        snakemake.params.area_column_in,
        snakemake.params.area_column_out
    )

