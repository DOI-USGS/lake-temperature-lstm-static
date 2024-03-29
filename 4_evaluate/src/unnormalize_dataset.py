import numpy as np

def unnormalize_dataset(metadata_filepath, dataset_filepath, output_filepath):
    """
    Unnormalize and save observations or static and dynamic inputs for one dataset (training,
    validation, or testing).

    :param metadata_filepath: Path to model training metadata npz
    :param dataset_filepath: Path to dataset with features for prediction and
        observations
    :param output_filepath: Path to save unnormalized dataset to

    """

    # Load metadata
    metadata_npz = np.load(metadata_filepath)
    # Convert npz to dictionary
    # Convert arrays to scalars where possible
    metadata = {file: metadata_npz[file][()] for file in metadata_npz.files}

    # Load data
    data_npz = np.load(dataset_filepath)

    # Determine which depths and features to use
    # Input features are drivers, clarity, ice flags, and static attributes, in that order.
    # Combine full set into one list
    elements_all = (data_npz['depths_all'].tolist() +
                    data_npz['dynamic_features_all'].tolist() +
                    data_npz['static_features_all'].tolist())
    # Select set to use out of full set for dynamic features, static features,
    # and observations
    idx_dynamic = [elements_all.index(e) for e in metadata['dynamic_features_use'].tolist()]
    idx_static = [elements_all.index(e) for e in metadata['static_features_use'].tolist()]
    idx_observations = [elements_all.index(e) for e in metadata['depths_use'].tolist()]

    # Rescale by unnormalizing
    # Unnormalized data = normalized data * standard deviation + mean
    # Shape of data is (# sequences, sequence_length, # depths + # input features)
    x_d = (data_npz['data'][:, :, idx_dynamic] *
           metadata['train_data_stds'][idx_dynamic] +
           metadata['train_data_means'][idx_dynamic])
    # Only need one set of static features because they repeat at every time
    # step, hence, the 0 index in [:, 0, idx_static]
    x_s = (data_npz['data'][:, 0, idx_static] *
           metadata['train_data_stds'][idx_static] +
           metadata['train_data_means'][idx_static])
    y = (data_npz['data'][:, :, idx_observations] *
         metadata['train_data_stds'][idx_observations] +
         metadata['train_data_means'][idx_observations])

    # Save to npz file
    np.savez(output_filepath,
             static_features=x_s,
             dynamic_features=x_d,
             observations=y)


if __name__ == '__main__':
    unnormalize_dataset(snakemake.input.metadata_filepath,
                        snakemake.input.dataset_filepath,
                        snakemake.output.output_filepath)

