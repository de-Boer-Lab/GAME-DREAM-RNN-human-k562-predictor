'''Validation and Preprocessing of Payload'''
import numpy as np
from error_checking_functions import *

SUPPORTED_SCALES = ["linear", "log"]
DEFAULT_SCALE = "log"

def validate_request_payload(payload):
    """
    Performs all validation checks on the incoming request payload.
    Returns a dictionary of errors. If the dictionary is empty, validation passed.
    """
    errors = {'bad_prediction_request': []}
    
    # First confirm all mandatory keys are present
    errors = check_mandatory_keys(payload.keys(), errors)
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise BadRequestError(flagged_errors)
    
    # Check for mandatory keys inside each task object.
    errors = check_prediction_task_mandatory_keys(payload['prediction_tasks'], errors)
    if any(errors.values()):
        # Fail immediately if any task is missing keys, before we try to access them.
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise BadRequestError(flagged_errors)
    
    # Perform all other validation checks
    errors = check_key_values_readout(payload['readout'], errors)
    errors = check_prediction_task_name(payload['prediction_tasks'], errors)
    errors = check_prediction_task_type(payload['prediction_tasks'], errors)
    errors = check_prediction_task_cell_type(payload['prediction_tasks'], errors)
    errors = check_prediction_task_species(payload['prediction_tasks'], errors)
    errors = check_prediction_task_scale(payload['prediction_tasks'], errors)

    if 'prediction_ranges' in payload:
        errors = check_seq_ids(payload['prediction_ranges'], payload['sequences'], errors)
        errors = check_prediction_ranges(payload['prediction_ranges'], errors)

    if 'upstream_seq' in payload:
        errors = check_key_values_upstream_flank(payload['upstream_seq'], errors)
    if 'downstream_seq' in payload:
        errors = check_key_values_downstream_flank(payload['downstream_seq'], errors)

    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise BadRequestError(flagged_errors)

def preprocess_data(payload):
    """
    Handles data preprocessing, like checking sequence specs, readout type, species, etc.
    
    Completes model-specific error checking.

    Returns processed sequences or raises a PredictionFailedError.
    """
    sequences = payload.get('sequences', {})

    # Model-specific: Note that even if flanking sequences are present in the request,
    # they are not used by this model
    if 'upstream_seq' in payload or 'downstream_seq' in payload:
        print("INFO: Upstream/Downstream sequences found. They will be used to construct the biological context before cropping/padding.")

    # Check that the final sequences meet model specifications.
    # Since this is model-specific, it utilizes `PredictionFailedError`.
    errors = {
        'bad_prediction_request': [],
        'prediction_request_failed': []
        }
    errors = check_seqs_specifications(sequences, errors)
    
    # Model-specific: Readout type check
    readout_type = payload['readout']
    if not readout_type in ["point"]: # Flags `track` or `interaction_matrix`
        raise BadRequestError(f"DREAM-RNN cannot process '{readout_type}' readout type.")
    
    # Model-specific: Ensure this DREAM-RNN Predictor only supports homo_sapiens
    for task in payload['prediction_tasks']:
        if task.get('species').lower() != "homo_sapiens":
            errors['bad_prediction_request'].append(
                f"This predictor only supports species: homo_sapiens. Received '{task.get('species')}' for task '{task.get('name')}'."
            )
        
        req_scale = task.get('scale')
        if req_scale and req_scale.lower() not in SUPPORTED_SCALES:
            errors['bad_prediction_request'].append(
                f"Unsupported scale: '{req_scale}'. Supported scales are: {SUPPORTED_SCALES}."
            )

    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)
    
    return sequences

# --- SCALING FUNCTION ---
def apply_scaling(predictions_dict, requested_scale):
    """
    Applies scaling transformation.
    
    ASSUMPTION: Model output is natively LOGGED (log2).
    
    Logic:
      - If 'log' requested (or Default): Return as is
      - If 'linear' requested: Exponentiate -- 2^x
    
    Args:
        predictions_dict (dict): The raw linear predictions
        requested_scale (str or None): The scale requested by the user
        
    Returns:
        tuple: (transformed_dict, actual_scale_str)
    """
    
    # Determine Effective Scale
    if not requested_scale:
        # Default if None provided
        effective_scale = DEFAULT_SCALE
    else:
        effective_scale = requested_scale.lower()
        
    if effective_scale == "log":
        return predictions_dict, effective_scale
    
    transformed_preds = {}
    for seq_id, values in predictions_dict.items():
        # Convert to numpy for fast vectorized math
        arr = np.array(values)
        
        if effective_scale == "linear":
            arr = np.exp2(arr)
            
        # Convert back to list for JSON serialization
        transformed_preds[seq_id] = arr.tolist()
        
    return transformed_preds, effective_scale
