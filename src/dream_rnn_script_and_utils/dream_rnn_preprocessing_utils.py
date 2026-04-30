'''Preprocessing Functions for DREAM-RNN Model Codebase'''

import os
import sys

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

from error_checking_functions import BadRequestError

# Assay-specific offset: in Agarwal et al. 2025's lentiMPRA constructs, on which
# this DREAM-RNN model is trained on, the regulatory probe ends 145 bp upstream
# of the EGFP TSS. 
# Used in Branch 2.1 of process_sequence to anchor the model's input window
# relative to the TSS when only the raw genomic sequence + prediction_ranges
# (no flanks) are provided.
MPRA_PROBE_TO_TSS_OFFSET = 145

def _crop_centre(seq, target_length):
    """
    Crops the sequence to the centre target length bases.
    """
    seq_len = len(seq)
    if seq_len <= target_length:
        return seq
    
    center_pos = seq_len // 2
    region_half = target_length // 2
    
    cropped_seq = seq[center_pos - region_half : center_pos + region_half]
    return cropped_seq

def _crop_around_pred_range(seq, pr_start, pr_end, target_length):
    """
    Crops the sequence around the specified prediction ranges to the target length.
    
    Args:
        seq (str): The input sequence
        pr_start (int): Start index of the prediction range
        pr_end (int): End index of the prediction range
        target_length (int): The desired length of the cropped sequence
        
    Returns:
        cropped_seq (str): The cropped sequence around the prediction range
    """
    # If sequence is already less than or equal to target length, return as is
    if len(seq) <= target_length:
        return seq
    
    # Calculate the centre of the prediction range
    pr_centre = (pr_start + pr_end) // 2
    half_window = target_length // 2 # This is to first centre around the pred range
    
    # Calculate the start index
    start = pr_centre - half_window
    
    # Ensure that
    # 1. start is not negative
    # 2. end does not exceed sequence length
    max_start = len(seq) - target_length # Maximum valid start index for cropping
    start = max(0, min(start, max_start)) # Clamp start index within valid range
    
    end = start + target_length
    cropped_seq = seq[start:end]
    
    return cropped_seq

def _crop_upstream_of_tss(seq, pr_start, target_length, offset):
    
    """
    Returns a window of the sequence upstream of the TSS, anchored by 
    the prediction range start and a known offset, for an MPRA-style assay.
    Ideal window: target_length bp ending `offset` bp before pr_start (the TSS).
    
    When the sequence is long enough, this returns exactly that window:
        [window_start ...... window_end] -- offset bp (145 bp) -- [pr_start = TSS]
        |------- target_length --------|
    
    When pr_start is too close to the start of the sequence to fit the
    full target_length window upstream of TSS - offset, we clamp the start
    to position 0 and pull a full target_length window from there. This
    pulls maximum real sequence context rather than inserting unnecessary
    Ns just to preserve the "biologically correct" offset to the TSS.
    
    This may return a window that is shorter than target_length if the 
    sequence itself is too short, but in that case we will pad with Ns 
    in process_sequence and log a warning.
    
    Args:
        seq (str): The input sequence
        pr_start (int): The start index of the prediction range (interpreted as TSS)
        target_length (int): The desired length of the output window
        offset (int): The known distance from the prediction range start to the TSS,
                      145 bp in this case.
                      
    Returns:
        window (str): The sequence window upstream of the TSS, ideally 
                      of length target_length
    """
    
    # Clamp to actual sequence bounds
    start = max(0, pr_start - offset - target_length)
    end = min(len(seq), start + target_length)
    window = seq[start:end]
    
    if len(window) < target_length:
        # If we don't have enough sequence upstream of the TSS, 
        # we can pad with Ns in process_sequence
        sys.stderr.write(
            f"LOG: TSS-anchored window pulled only {len(window)} bp of real "
            f"upstream sequence (wanted {target_length}); shortfall will be "
            f"left-padded with Ns. Predictions may be unreliable.\n"
        )
    
    return window

def _pad_sequence_with_Ns(seq, target_length):
    """
    Left-pad a sequence with 'N' until it reaches the target length

    Args:
        seq (str): The sequence that needs to be padded
        target_length (int): The desired length of the sequence after padding
        
    Returns:
        padded_seq (str): The left-padded sequence
    """
    
    if len(seq) >= target_length:
        return seq
    
    total_padding = target_length - len(seq)
    pad = 'N' * total_padding
    
    padded_seq = pad + seq
    return padded_seq

def _pad_with_flanks(seq, upstream_seq, downstream_seq, target_length):
    
    """
    Pads a probe (variable region) shorter than target_length with flank
    context. Takes the tail of the upstream flank and the head of the downstream flank
    so the probe sits in the same context it would in the actual construct.
    
    If the available flanks are not long enough to fill target_length, the
    remaining is filled with N-padding.
    
    Args:
        seq (str): The variable region sequence (probe)
        upstream_seq (str): The upstream flank sequence provided by the evaluator
        downstream_seq (str): The downstream flank sequence provided by the evaluator
        target_length (int): The desired length of the output sequence (200 bp)
        
    Returns:
        padded_seq (str): The sequence with flanks applied and padded to target_length if needed
    """
    
    if len(seq) >= target_length:
        return seq
    
    # Calculate how much padding is needed
    total_padding = target_length - len(seq)
    upstream_pad_len = total_padding // 2
    downstream_pad_len = total_padding - upstream_pad_len
    
    # Take the TAIL of the upstream_seq and the HEAD of the downstream_seq
    upstream_pad = upstream_seq[-upstream_pad_len:] if upstream_pad_len > 0 else ""
    downstream_pad = downstream_seq[:downstream_pad_len] if downstream_pad_len > 0 else ""
    
    padded_seq = upstream_pad + seq + downstream_pad
    
    # If flanked sequence is still shorter than target_length, pad with Ns
    if len(padded_seq) < target_length:
        padded_seq = _pad_sequence_with_Ns(padded_seq, target_length)
    
    return padded_seq

def _add_hardcoded_adapters(seq, upstream_adapter, downstream_adapter):
    """
    Add upstream and downstream adapter sequences to a given sequence.
    
    Args:
        seq (str): The input sequence
        upstream_adapter (str): Upstream adapter sequence to prepend
        downstream_adapter (str): Downstream adapter sequence to append
        
    Returns:
        seq_with_adapters (str): The sequence with adapters added.
    """
    seq_with_adapters = upstream_adapter + seq + downstream_adapter
    return seq_with_adapters
    
# One-hot encode sequences
def _one_hot_encode(seq):
    """
    Convert a sequence to a one-hot encoded representation.

    Args:
        seq (str): Sequence consisting of 'A', 'T', 'G', 'C', and 'N'.

    Returns:
        list: A one-hot encoded list for the sequence.
    """
    mapping = {'A': [1, 0, 0, 0],
               'G': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}
    return [mapping.get(base, [0, 0, 0, 0]) for base in seq]

# Validate if prediction ranges are within flanked sequence bounds (for DREAM-RNN)
# This could be different for different Predictors, so kept separate
def _validate_prediction_range_bounds(prediction_ranges, seq_length):
    """
    Validate if prediction ranges are within the bounds of the flanked sequence length.
    Called only when prediction_ranges are actually used --
    CASE 2.1 (in process_sequence) no flanks provided + prediction_ranges provided.

    Args:
        prediction_ranges (list): List containing start and end indices of prediction ranges
        seq_length (int): Length of the flanked sequence

    Raises:
        BadRequestError: If prediction ranges are out of bounds.
    """
    if prediction_ranges is None:
        return

    pr_start, pr_end = prediction_ranges
    
    if pr_start < 0 or pr_end > seq_length:
        raise BadRequestError(
            f"prediction_ranges: [{pr_start}, {pr_end}] are out of bounds for flanked sequence length: {seq_length}."
        )

# Full preprocessing pipeline for a sequence
def process_sequence(seq,  prediction_ranges, upstream_seq, downstream_seq,
            target_length, upstream_adapter, downstream_adapter):
    """
    Process a sequence through padding/cropping, adding adapters, handling prediction_ranges, and one-hot encoding.

    Args:
        seq (str): The input sequence -- typically the variable region (probe) without flanks, provided by the evaluator
        prediction_ranges (list or None): List of prediction ranges to consider
        upstream_seq (str): Upstream flanking sequence provided by evaluator
        downstream_seq (str): Downstream flanking sequence provided by evaluator
        target_length (int): Desired length of the sequence before adding adapters
        upstream_adapter (str): Upstream adapter sequence to prepend (HARDCODED for DREAM-RNN)
        downstream_adapter (str): Downstream adapter sequence to append (HARDCODED for DREAM-RNN)

    Returns:
        list: One-hot encoded representation of the processed sequence
        
    Logic:
        1. If constant flanks (upstream_seq / downstream_seq) provided:
            1.1. Check probe (variable region) length against target_length (200bp)
                    1.1.1. probe length <= target_length:
                        Pad with flanks to reach target_length (NOT Ns)
                        - Take tail of upstream_seq + probe + head of downstream_seq
                        - If target length not reached, pad with Ns
                        - Ignore prediction_ranges
                    1.1.2. probe length > target_length:
                        Crop centre to target_length, ignore prediction_ranges

        2. If no constant flanks:
            2.1. If probe length <= target length:
                    - Simply use the entire probe
                    - Pad with Ns, as necessary
            2.2. If probe length > target length and prediction_ranges provided (marks TSS/reporter location):
                    - MPRA-specific offset: probe ends 145bp upstream of TSS (MPRA_PROBE_TO_TSS_OFFSET)
                    - Region of interest:
                        start = max(0, PR_start - target_length - 145)
                        end = min(start + target_length, len(seq))
                        Pull as much real sequence context
                    - Pad with Ns if sequence still < target_length
            2.2. Else: crop to centre (probe length > target length and no pred_ranges)

        3. Add hardcoded DREAM-RNN adapters (15bp upstream + 15bp downstream)
        4. One-hot encode the final sequence
    """
    
    # First check if constant flanks are provided
    flanks_provided = bool(upstream_seq) or bool(downstream_seq)
    
    if flanks_provided:
        # Branch 1: Flanks provided, ignore prediction_ranges
        sys.stderr.write("LOG: Flanks provided. Applying flanks and ignoring prediction_ranges... \n")
        upstream_seq = upstream_seq if upstream_seq is not None else ""
        downstream_seq = downstream_seq if downstream_seq is not None else ""
        
        if len(seq) <= target_length:
            # Branch 1.1: Pad probe with flank context
            target_length_seq = _pad_with_flanks(
                seq, upstream_seq, downstream_seq, target_length
            )
        else:
            # Branch 1.2: probe is longer than target length, crop to centre and ignore prediction_ranges
            target_length_seq = _crop_centre(seq, target_length)
    
    else:        
        # Branch 2: No flanks provided
        sys.stderr.write("LOG: No flanks provided. Handling sequence based on length and prediction_ranges... \n")
        if len(seq) <= target_length:
            # Branch 2.1: probe is shorter than target length, use as is and pad with Ns
            target_length_seq = _pad_sequence_with_Ns(seq, target_length)
        elif prediction_ranges is not None:
            # Branch 2.2: probe is longer than target length, prediction_ranges provided
            # marks the TSS -- pull upstream window
            sys.stderr.write("LOG: No flanks provided but prediction_ranges provided. Applying MPRA-specific TSS-anchored cropping... \n")
            _validate_prediction_range_bounds(prediction_ranges, len(seq))
            pr_start, _ = prediction_ranges
            target_length_seq = _crop_upstream_of_tss(
                seq, pr_start, target_length, MPRA_PROBE_TO_TSS_OFFSET
            )
            if len(target_length_seq) < target_length:
                target_length_seq = _pad_sequence_with_Ns(target_length_seq, target_length)
        else:
            # Branch 2.3: probe is longer than target length, no prediction_ranges provided -- crop to centre
            sys.stderr.write("LOG: No flanks or prediction_ranges provided. Cropping to centre... \n")
            target_length_seq = _crop_centre(seq, target_length)
            
    # Step 3: Add hardcoded DREAM-RNN adapters
    seq_with_adapters = _add_hardcoded_adapters(
        target_length_seq, upstream_adapter, downstream_adapter
    )
    
    # Step 4: One-hot encode the sequence
    encoded_seq = _one_hot_encode(seq_with_adapters)
    
    return encoded_seq
    
    
# # Step 1: Apply evaluator-provided flanks first
# flanked_seq = _pad_with_flanks(seq, upstream_seq, downstream_seq)

# # Validate prediction ranges if provided
# _validate_prediction_range_bounds(prediction_ranges, len(flanked_seq))

# # Step 2: Check length of flanked sequence against target_length
# if len(flanked_seq) < target_length:
#     # Pad with Ns
#     # sys.stderr.write("LOG:Ignoring prediction_ranges since flanked sequence is shorter than target length. \n")
#     # sys.stderr.write("LOG:padding with Ns... \n")
#     target_length_seq = _pad_sequence_with_Ns(flanked_seq, target_length)
# elif len(flanked_seq) == target_length:
#     # sys.stderr.write("LOG:Ignoring prediction_ranges since flanked sequence matches target length. \n")
#     target_length_seq = flanked_seq
# else: # len(flanked_seq) > target_length
#     if prediction_ranges is not None:
#         # sys.stderr.write("LOG:Cropping around prediction_ranges... \n")
#         pr_start, pr_end = prediction_ranges
#         target_length_seq = _crop_around_pred_range(flanked_seq, pr_start, pr_end, target_length)
#     else:
#         # sys.stderr.write("LOG:Cropping centre since no prediction_ranges provided... \n")
#         target_length_seq = _crop_centre(flanked_seq, target_length)

# # Step 3: Add hardcoded adapters
# seq_with_adapters = _add_hardcoded_adapters(target_length_seq, upstream_adapter, downstream_adapter)

# # Step 4: One-hot encode the sequence
# encoded_seq = _one_hot_encode(seq_with_adapters)

# return encoded_seq
