import numpy as np
import torch

import sys
sys.path.insert(1, '3_train/src')
from train import get_model

def predict_dataset(weights_filepath,
                    metadata_filepath,
                    dataset_filepath,
                    predictions_filepath):
    """
    Load or compute and save temperature predictions for one dataset (training,
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
    metadata_npz = np.load(metadata_filepath)
    # Convert npz to dictionary
    # Convert arrays to scalars where possible
    metadata = {file: metadata_npz[file][()] for file in metadata_npz.files}

    # Get number of inputs/outputs
    n_depths = len(metadata['depths_use'])
    n_dynamic = len(metadata['dynamic_features_use'])
    n_static = len(metadata['static_features_use'])

    ## Load data

    data_npz = np.load(dataset_filepath)

    # Determine which depths and features to use
    # Input features are drivers, clarity, ice flags, and static attributes, in that order.
    # Combine full set into one list
    elements_all = (data_npz['depths_all'].tolist() +
                    data_npz['dynamic_features_all'].tolist() +
                    data_npz['static_features_all'].tolist())
    # Combine set to use into one list
    elements_use = (metadata['depths_use'].tolist() + 
                    metadata['dynamic_features_use'].tolist() + 
                    metadata['static_features_use'].tolist())
    # Select set to use out of full set
    idx_to_use = [elements_all.index(e) for e in elements_use]
    # Shape of data is (# sequences, sequence_length, # depths + # input features)
    data_array_to_use = data_npz['data'][:,:,idx_to_use]

    ## Make predictions
    x_d = torch.tensor(data_array_to_use[:, :, n_depths:-n_static])
    x_s = torch.tensor(data_array_to_use[:, 0, -n_static:])
    y = torch.tensor(data_array_to_use[:, :, :n_depths])

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

    with torch.no_grad():
        pred, h, c = model(x_d, x_s)

    def unnormalize(arr):
        return arr*metadata['train_data_stds'][:n_depths] + metadata['train_data_means'][:n_depths]

    # Rescale predictions, observations, and static features
    predictions = unnormalize(pred)
    observations = unnormalize(y)
    static_features = (x_s*metadata['train_data_stds'][-n_static:] + metadata['train_data_means'][-n_static:]).numpy()

    np.savez(predictions_filepath,
             static_features=static_features,
             observations=observations,
             predictions=predictions)


if __name__ == '__main__':
    predict_dataset(snakemake.input.weights_filepath,
                    snakemake.input.metadata_filepath,
                    snakemake.input.dataset_filepath,
                    snakemake.output.predictions_filepath)

