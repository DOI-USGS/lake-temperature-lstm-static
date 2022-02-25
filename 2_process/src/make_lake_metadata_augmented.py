import os
import requests
import urllib
import pandas as pd


def elevation_query(lat, lon):
    """
    Query service using lat, lon and return the corresponding elevation values.
    Thanks to https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python

    :param lat: latitude values in decimal degrees
    :param lon: longitude values
    :returns: elevation at the specified coordinates

    """
    # USGS Elevation Point Query Service
    url = r'https://nationalmap.gov/epqs/pqs.php?'
    # define rest query params
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    # format query string and return query value
    result = requests.get((url + urllib.parse.urlencode(params)))
    elevation = result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']

    return elevation


def make_lake_metadata_augmented(in_file, out_file):
    """
    Add "elevation" column to metadata file and save to new file

    :param in_file: filename to read lake metadata from
    :param out_file: filename to save to

    """
    lake_metadata = pd.read_csv(in_file)
    # Add elevation to metadata
    lake_metadata['elevation'] = lake_metadata.apply(
        lambda row: elevation_query(
            row['centroid_lat'], row['centroid_lon']),
        axis=1)
    lake_metadata.to_csv(out_file)


if __name__ == '__main__':
    make_lake_metadata_augmented(snakemake.input[0], snakemake.output[0])
