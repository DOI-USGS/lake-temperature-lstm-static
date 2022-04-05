# lake-temperature-lstm-static

Predict lake temperatures at multiple depths using an LSTM trained to make use of static lake attributes.

## Installing

Create a new `conda` environment by running

```
conda update -n base conda
conda env create -f environment.yaml
conda activate ltls
```

Note: if [Mamba](https://github.com/mamba-org/mamba) is available, the environment may be created significantly faster by running

```
mamba update -n base mamba
mamba env create -f environment.yaml
conda activate ltls
```

## Downloading input data

For now, MNTOHA data are being used.
A more complete set of lakes will be incorporated in the future, but MNTOHA lake data are readily available to download from ScienceBase.
To download temperature observations, meteorological drivers, and metadata, run

`snakemake -c1 -p fetch_all`

Replace the `1` in `-c1` with the number of cores to execute jobs in parallel.

The data will be downloaded to the `1_fetch/out` folder.
If the download is interrupted, resume it and overwrite partially downloaded files by running

`snakemake -c1 -p --rerun-incomplete fetch_all`

To trigger a fresh download, change the date in 1_fetch/in/pull_date.txt to today's date and then call snakemake as above.

## Licenses

This work makes use of code available at
https://github.com/kratzert/ealstm_regional_modeling. That code permits such
use under the Apache-2.0 License. See
<https://opensource.org/licenses/Apache-2.0>

