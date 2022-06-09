import os
import pandas as pd

def add_feature_from_file(in_file, feature_file, out_file, feature_column_in, feature_column_out):
    """
    Add column with lake surface features to metadata and save to new file.
    
    Code smell: This function is similar to
    2_process/src/make_lake_metadata_augmented.add_elevation_from_file

    :param in_file: Filename of csv to read existing lake metadata from
    :param feature_file: Filename of csv file to read lake feature data from
    :param out_file: Filename of csv to save metadata augmented with feature to
    :param feature_column_in: Name of column with features in in_file
    :param feature_column_out: What to name column with features in out_file

    """

    lake_metadata = pd.read_csv(in_file)
    lake_features = pd.read_csv(feature_file)
    lake_features_only = lake_features.loc[:, ['site_id', feature_column_in]]
    # Add feature to metadata
    augmented = pd.merge(lake_metadata, lake_features_only, how="inner", on='site_id')
    # Rename column to feature
    augmented = augmented.rename({feature_column_in: feature_column_out}, axis='columns')

    destination_dir = os.path.dirname(out_file)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    augmented.to_csv(out_file, index=False)


if __name__ == '__main__':
    # Add lake feature to metadata
    add_feature_from_file(
        snakemake.input.lake_metadata,
        snakemake.input.feature_metadata,
        snakemake.output.augmented_metadata,
        snakemake.params.feature_column_in,
        snakemake.params.feature_column_out
    )

