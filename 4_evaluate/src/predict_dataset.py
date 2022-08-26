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

    # Initialize results array
    output_shape = (num_sequences, prediction_sequence_length, n_depths)
    preds = np.empty(output_shape)

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
            preds[i_sequence: i_sequence + num_pred_sequences, :, :] = unnormalize(pred.numpy()[:, spinup_time:, :])
            i_sequence += num_pred_sequences

    # Sanity check
    if i_sequence != preds.shape[0]:
        raise Exception("Sequences predicted and results array are misaligned")

    np.savez(predictions_filepath,
             # static_features=static_features,
             # observations=obs,
             predictions=preds)

if __name__ == '__main__':
    compute_dataset_predictions(snakemake.input.weights_filepath,
                                snakemake.input.metadata_filepath,
                                snakemake.input.dataset_filepath,
                                snakemake.output.predictions_filepath)

