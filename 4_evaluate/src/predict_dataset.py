import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(1, '3_train/src')
from train import get_model, get_sequence_dataset


def get_dataset_data_loader(dataset_filepath,
                            dynamic_features_use,
                            static_features_use,
                            depths_use,
                            batch_size):
    """
    Get data loader for a dataset (training, validation, or testing).
    """

    data_set = get_sequence_dataset(
        dataset_filepath,
        dynamic_features_use,
        static_features_use,
        depths_use)
    # The dataset is not used for training with backprop, so we can
    # afford its batch size to be larger and it doesn't need to be shuffled
    return DataLoader(data_set, batch_size=batch_size)


# def unnormalize_dataset(metadata_filepath, dataset_filepath, data_type, output_filepath):
#     """
#     Unnormalize and save observations or static and dynamic inputs for one dataset (training,
#     validation, or testing).

#     :param data_type: Either "input" or "observation"
#     """

#     # Load metadata
#     metadata_npz = np.load(metadata_filepath)
#     # Convert npz to dictionary
#     # Convert arrays to scalars where possible
#     metadata = {file: metadata_npz[file][()] for file in metadata_npz.files}

#     # Get number of inputs/outputs
#     n_depths = len(metadata['depths_use'])
#     n_dynamic = len(metadata['dynamic_features_use'])
#     n_static = len(metadata['static_features_use'])

#     # Load data
#     data_npz = np.load(dataset_filepath)

#     # Determine which depths and features to use
#     # Input features are drivers, clarity, ice flags, and static attributes, in that order.
#     # Combine full set into one list
#     elements_all = (data_npz['depths_all'].tolist() +
#                     data_npz['dynamic_features_all'].tolist() +
#                     data_npz['static_features_all'].tolist())
#     # Combine set to use into one list
#     elements_use = (metadata['depths_use'].tolist() + 
#                     metadata['dynamic_features_use'].tolist() + 
#                     metadata['static_features_use'].tolist())
#     # Select set to use out of full set
#     idx_to_use = [elements_all.index(e) for e in elements_use]
#     # Shape of data is (# sequences, sequence_length, # depths + # input features)
#     data_array_to_use = data_npz['data'][:,:,idx_to_use]

#     if data_type == "input":
#         # Rescale by unnormalizing
#         x_d = data_array_to_use[:, :, n_depths:-n_static]*metadata['train_data_stds'][:n_depths] + metadata['train_data_means'][:n_depths]
#         x_s = data_array_to_use[:, 0, -n_static:]*metadata['train_data_stds'][-n_static:] + metadata['train_data_means'][-n_static:]
#         # Save to npz file
#         np.savez(output_filepath, static_features=x_s, dynamic_features=x_d)
#     elif data_type == "observation":
#         # Rescale by unnormalizing
#         y = data_array_to_use[:, :, :n_depths]*metadata['train_data_stds'][:n_depths] + metadata['train_data_means'][:n_depths]
#         # Save to npz file
#         np.savez(output_filepath, observations=y)


def compute_dataset_predictions(weights_filepath,
                                metadata_filepath,
                                dataset_filepath,
                                predictions_filepath):
    """
    Compute and save LSTM temperature predictions for one dataset (training,
    validation, or testing).
    
    Load the model weights and the model metadata, make predictions, and
    unnormalize the results along with the observations in the dataset and the
    static input features. Save the unnormalized predictions, observations, and
    static features to a npz file.

    :param weights_filepath: Path to saved model weights
    :param metadata_filepath: Path to model training metadata npz
    :param dataset_filepath: Path to dataset with features for prediction and
        observations
    :param predictions_filepath: Path to save unnormalized results to

    """

    # Load metadata
    metadata_npz = np.load(metadata_filepath)
    # Convert npz to dictionary
    # Convert arrays to scalars where possible
    metadata = {file: metadata_npz[file][()] for file in metadata_npz.files}

    # Get number of inputs/outputs
    n_depths = len(metadata['depths_use'])
    n_dynamic = len(metadata['dynamic_features_use'])
    n_static = len(metadata['static_features_use'])

    # Load data
    data_loader = get_dataset_data_loader(
        dataset_filepath,
        metadata['dynamic_features_use'].tolist(),
        metadata['static_features_use'].tolist(),
        metadata['depths_use'].tolist(),
        int(metadata['batch_size']*2)
    )

    # Set device
    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu"
    print(f"Using {device} device", flush=True)

    # Create model
    model = get_model(
        n_depths,
        n_dynamic,
        n_static,
        metadata['hidden_size'],
        metadata['initial_forget_bias'],
        metadata['dropout'],
        metadata['concat_static']).to(device)

    # Load weights
    model.load_state_dict(torch.load(weights_filepath))
    # Set dropout to eval mode
    model.eval()

    num_sequences = data_loader.dataset.len
    # Only save observations and predictions after spinup time
    spinup_time = metadata['spinup_time']
    full_sequence_length = metadata['sequence_length']
    prediction_sequence_length = full_sequence_length - spinup_time

    # Initialize results arrays
    # static_features = np.empty((num_sequences, n_static))

    output_shape = (num_sequences, prediction_sequence_length, n_depths)
    preds = np.empty(output_shape)
    # obs = np.empty(output_shape)

    def unnormalize(arr):
        return arr*metadata['train_data_stds'][:n_depths] + metadata['train_data_means'][:n_depths]

    with torch.no_grad():
        i_sequence = 0
        for x_d, x_s, y in data_loader:
            # Transfer to device
            x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
            # model only needs one set of static inputs
            # They are identical at all time steps
            pred, h, c = model(x_d, x_s[:, 0, :])
            num_pred_sequences = pred.shape[0]
            # Store obs and predictions after spinup time into results arrays
            # obs[i_sequence: i_sequence + num_pred_sequences, :, :] = y.numpy()[:, spinup_time:, :]
            preds[i_sequence: i_sequence + num_pred_sequences, :, :] = unnormalize(pred.numpy()[:, spinup_time:, :])
            # x_s is the same for all time steps, so only store the first into results array
            # static_features[i_sequence: i_sequence + num_pred_sequences, :] = x_s.numpy()[:, 0, :]
            i_sequence += num_pred_sequences

    # Sanity check
    if i_sequence != preds.shape[0]:
        raise Exception("Sequences predicted and results array are misaligned")

    # Rescale predictions, observations, and static features
    # preds = unnormalize(preds)
    # obs = unnormalize(obs)
    # static_features = (static_features*metadata['train_data_stds'][-n_static:] +
    #                    metadata['train_data_means'][-n_static:])

    np.savez(predictions_filepath,
             # static_features=static_features,
             # observations=obs,
             predictions=preds)

if __name__ == '__main__':
    compute_dataset_predictions(snakemake.input.weights_filepath,
                                snakemake.input.metadata_filepath,
                                snakemake.input.dataset_filepath,
                                snakemake.output.predictions_filepath)

