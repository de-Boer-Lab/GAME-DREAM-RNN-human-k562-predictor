'''DREAM-RNN Model Codebase'''
import os
import sys
import tqdm
import torch
import numpy as np

from dream_rnn_preprocessing_utils import process_sequence

# Get the absolute path of the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Determine if running inside a container or not
if os.path.exists('/.singularity.d'):
    # Running inside the container
    BASE_DIR = '/dream_rnn_script_and_utils'
    UTILS_DIR = '/script_and_utils'
else:
    # Running outside the container
    # Get the current working directory
    BASE_DIR = SCRIPT_DIR
    UTILS_DIR = os.path.join(SCRIPT_DIR, "..", "script_and_utils")

# Add the directory containing `error_checking_functions.py` to the Python path
sys.path.append(UTILS_DIR)

# Import the error classes to raise 
from error_checking_functions import PredictionFailedError

# Define the model directory
MODEL_DIR = os.path.join(BASE_DIR, 'dream_rnn_k562_model_weight')

# Import prixfixe framework
from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock
from prixfixe.bhi import BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

# initialize path and variables
CUDA_DEVICE_ID = 0
SEQ_SIZE = 230
generator = torch.Generator()
generator.manual_seed(42)
device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")

# Model Definition For Dream-RNN
def build_dream_rnn():
    first = BHIFirstLayersBlock(
        in_channels = 5,
        out_channels = 320,
        seqsize = SEQ_SIZE,
        kernel_sizes = [9, 15],
        pool_size = 1,
        dropout = 0.2
    )

    core = BHICoreBlock(
        in_channels = first.out_channels,
        out_channels = 320,
        seqsize = first.infer_outseqsize(),
        lstm_hidden_channels = 320,
        kernel_sizes = [9, 15],
        pool_size = 1,
        dropout1 = 0.2,
        dropout2 = 0.5
    )

    final = AutosomeFinalLayersBlock(
        in_channels=core.out_channels,
        seqsize=core.infer_outseqsize()
    )

    model_rnn = PrixFixeNet(
        first=first,
        core=core,
        final=final,
        generator=generator
    )
    return model_rnn

# Load Pre-Trained Model Weights for DREAM-RNN
def load_dream_rnn():
    try:
        model = build_dream_rnn()
        model_path = f"{MODEL_DIR}/model_best.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found at {model_path}")
        model.load_state_dict(torch.load(model_path,
                                     map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise PredictionFailedError(f"Failed to load model. The server may not be configured correctly: {e}")

# Hardcoded Upstream and Downstream Adapter Sequences for K562 and HepG2:
TARGET_LENGTH = 200
upstream_adapter_seq = "AGGACCGGATCAACT"
downstream_adapter_seq = "CATTGCGTGAACCGA"

# Prediction Function
def predict_dream_rnn(sequences, prediction_ranges, upstream_seq, downstream_seq, include_rev):
    # ADDITION ^^^ 3 new arguments -- pred ranges, upstream, and downstream seqs
    """
    Predict expression values using the DREAM-RNN model.

    Args:
        sequences (dict): Dictionary of sequence IDs and their corresponding sequences.
        include_rev (bool): Whether to include the reverse complement flag in the input.

    Returns:
        predictions (dict): Dictionary of sequence IDs and their predicted expression values.
    """
    
    try:
        print("Loading pre-trained model weights for DREAM-RNN")
        model_rnn = load_dream_rnn()
        
        predictions = {}
        # Wrap the iteration with tqdm for a progress bar
        for seq_id, seq in tqdm.tqdm(sequences.items(),
                                    desc="Predictions in progress", unit="sequence"):
            
            seq_specific_prediction_ranges = None
            if prediction_ranges is not None:
                seq_specific_prediction_ranges = prediction_ranges.get(seq_id) or None
            
            # Process sequence for padding
            encoded_seq = process_sequence(seq, seq_specific_prediction_ranges, upstream_seq, downstream_seq,
                        TARGET_LENGTH, upstream_adapter_seq, downstream_adapter_seq)
            
            # Include reverse complement information on sequence ID
            if include_rev:
                rev_flag = 1 if "Reversed" in seq_id else 0
                rev_value = [rev_flag] * len(encoded_seq)
            else:
                rev_value = [0] * len(encoded_seq) # Default to no reverse complement info
                
            # Combine one-hot encoding with reverse flag
            encoded_seq_with_rev = [list(encoded_base) + [rev] for encoded_base,
                                    rev in zip(encoded_seq, rev_value)]
        
            # Predict expression values using the model
            # Convert to tensor for prediction
            seq_tensor = torch.tensor(
                np.array(encoded_seq_with_rev).reshape(1, SEQ_SIZE, 5).transpose(0, 2, 1),
                device=device,
                dtype=torch.float32
            )
            
            pred = model_rnn(seq_tensor)
            prediction_value = pred.detach().cpu().item()
            predictions[seq_id] = round(prediction_value, 5)
    
        return predictions
    
    except Exception as e:
        # If the error is already a known API error (like BadRequestError from preprocessing prediction_ranges bound validation),
        # re-raise it so the Flask app receives the correct Status 400 code.
        if hasattr(e, 'status_code'):
            raise e
        
        # Raise a standardized PredictionFailedError that our Flask handler can catch
        raise PredictionFailedError(f"An unexpected error occurred during model prediction: {e}")
