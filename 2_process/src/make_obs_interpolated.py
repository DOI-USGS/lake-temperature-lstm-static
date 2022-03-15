import numpy as np
import pandas as pd
from scipy import interpolate


def make_obs_interpolated(in_file, out_file, depths):
    """
    Add "interpolated_depth" column to obs file and save to new file.
    For now, this uses nearest neighbor interpolation.

    :param in_file: Filename to read observations from.
    :param out_file: Filename to save observations with extra column to.
    :param depths: Depths to interpolate to.

    """

    # Assign depth to observations (nearest neighbor)
    # Read temperature observations
    obs = pd.read_csv(in_file, parse_dates=['date'])
    # lake depths to use for LSTM
    depths = np.array(depths)
    # Round depth to nearest neighbor
    depth_interpolant = interpolate.interp1d(depths, depths,
                                             kind='nearest',
                                             assume_sorted=False)
    obs['interpolated_depth'] = depth_interpolant(obs.depth)
    obs.to_csv(out_file)


if __name__ == '__main__':
    make_obs_interpolated(snakemake.input[0], snakemake.output[0], snakemake.params['depths'])
