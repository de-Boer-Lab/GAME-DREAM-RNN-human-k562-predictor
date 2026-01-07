'''Preprocessing Functions for DREAM-RNN Model Codebase'''

def pad_sequence(seq, target_length):
    """
    Pad a sequence, without adapters, with 'N' until it reached the target length.
    
    Args:
        seq (str): The input sequence.
        target_length (int): Model-dependent desired length of the sequence before
                             adding adapters.
    
    Returns:
        padded_seq: The padded sequence.
    """
    seq_len = len(seq)
    
    # If sequence length is less than target_length, excluding adapters, simply
    # pad with Ns until it is target_length and then add the adapters.
    if seq_len < target_length:
        total_padding = target_length - seq_len
        # right_padding = 'N' * (total_padding // 2)
        # left_padding = 'N' * (total_padding - len(right_padding))
        # Only pad to the left of the sequence
        pad = 'N' * total_padding
        padded_seq = pad + seq
        return padded_seq
    
    # When the sequence length, excluding adapters, is the target_length,
    # simply add adapters to each side.
    elif seq_len == target_length:
        return seq
    
    # TODO: The condition for when sequence length is greater than the target length  
    # (not including adapter sequences on each side), is a work in progress.
    # For now Ishika just takes the center sequnce
    else:
        center_pos = seq_len//2
        region_half = target_length//2
        cropped_seq = seq[center_pos-region_half:center_pos+region_half]
        # seqs_updated[key] = cropped_seq
        # truncated_seq =  seq[:target_length] # For now, just truncate [WIP: Work in progress]
        return cropped_seq
    
def add_adapters(seq, upstream_adapter, downstream_adapter):
    """
    Add upstream and downstream adapter sequences to a given sequence.
    
    Args:
        seq (str): The input sequence after padding, if needed. 
        upstream_adapter (str): Upstream adapter sequence to prepend.
        downstream_adapter (str): Downstream adapter sequence to append.
        
    Returns:
        seq_with_adapters (str): The sequence with adapters added.
    """
    seq_with_adapters = upstream_adapter + seq + downstream_adapter
    return seq_with_adapters

# One-hot encode sequences
def one_hot_encode(seq):
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

# Full preprocessing pipeline for a sequence
def process_sequence(seq, target_length, seq_size,
                     upstream_adapter, downstream_adapter):
    """
    Process a sequence by padding, adding adapters, and encoding.

    Args:
        seq (str): The input sequence.
        target_length (int): Length of the sequence before adding adapters (200 for Dream-RNN).
        seq_size (int): Model-specific final sequence size (230 for Dream-RNN)
        upstream_adapter (str): Upstream adapter sequence to prepend.
        downstream_adapter (str): Downstream adapter sequence to append.

    Returns:
        list: A one-hot encoded list with padding and adapters applied.
    """
    # Step 1: Pad the sequence
    padded_seq = pad_sequence(seq, target_length)
    
    # Step 2: Add adapter sequence/ primers
    seq_with_adapters = add_adapters(padded_seq, upstream_adapter, downstream_adapter)

    # Step 3: One-hot encode the sequence
    encoded_seq = one_hot_encode(seq_with_adapters)
    
    return encoded_seq