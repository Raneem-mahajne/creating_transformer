"""Data generation, encoding, splitting, and batching."""
import random
import torch
from IntegerStringGenerator import IntegerStringGenerator


def generate_integer_string_data(generator: IntegerStringGenerator, num_sequences: int = 1000, 
                                  min_length: int = 50, max_length: int = 200) -> list[list[int]]:
    """
    Generate integer sequences using the generator and return as a list of sequences.
    
    Args:
        generator: IntegerStringGenerator instance
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        List of sequences (each sequence is a list of integers)
    """
    sequences = generator.generate_dataset(num_sequences, min_length, max_length)
    return sequences


def build_encoder_for_integers(min_value: int = 0, max_value: int = 20):
    """
    Builds encoder/decoder for integer tokens.
    Each integer value (min_value to max_value) is a unique token.
    Works directly with integers - no text conversion.
    
    Args:
        min_value: Minimum integer value (inclusive)
        max_value: Maximum integer value (inclusive)
        
    Returns:
        encode, decode, vocab_size, index_to_string, string_to_index
    """
    # Vocabulary: all integers from min_value to max_value
    vocab_size = max_value - min_value + 1
    
    # index_to_string: maps token index -> integer value string (only for plotting/display)
    # string_to_index: maps integer value string -> token index (for compatibility)
    index_to_string = {i: str(min_value + i) for i in range(vocab_size)}
    string_to_index = {str(min_value + i): i for i in range(vocab_size)}
    
    def encode(integers: list[int]) -> list[int]:
        """
        Encode integer values directly to token indices.
        Args:
            integers: List of integer values (e.g., [0, 1, 2, 15])
        Returns:
            List of token indices (integers)
        """
        return [val - min_value for val in integers]
    
    def decode(token_indices) -> list[int]:
        """
        Decode token indices directly back to integer values.
        Args:
            token_indices: List of token indices (integers)
        Returns:
            List of integer values
        """
        return [idx + min_value for idx in token_indices]
    
    return encode, decode, vocab_size, index_to_string, string_to_index


def build_encoder_with_operators(min_value: int, max_value: int, operators: list[str]):
    """
    Build encoder/decoder for mixed vocabulary (integers + operators).
    
    Args:
        min_value: Minimum integer value (inclusive)
        max_value: Maximum integer value (inclusive)
        operators: List of operator strings (e.g., ["+", "-"])
        
    Returns:
        encode, decode, vocab_size, index_to_string, string_to_index
    """
    # Vocabulary: integers first, then operators
    # Integers: indices 0 to (max_value - min_value)
    # Operators: indices after integers
    num_integers = max_value - min_value + 1
    vocab_size = num_integers + len(operators)
    
    # Build mappings
    # token_to_index: maps actual token (int or str) -> index
    # index_to_token: maps index -> actual token (int or str)
    token_to_index = {}
    index_to_token = {}
    
    # Add integers
    for i in range(num_integers):
        val = min_value + i
        token_to_index[val] = i
        index_to_token[i] = val
    
    # Add operators
    for i, op in enumerate(operators):
        idx = num_integers + i
        token_to_index[op] = idx
        index_to_token[idx] = op
    
    # index_to_string for display purposes
    index_to_string = {i: str(index_to_token[i]) for i in range(vocab_size)}
    string_to_index = {str(index_to_token[i]): i for i in range(vocab_size)}
    
    def encode(tokens: list) -> list[int]:
        """
        Encode mixed tokens (integers and operators) to token indices.
        Args:
            tokens: List of tokens (can be int or str)
        Returns:
            List of token indices
        """
        return [token_to_index[t] for t in tokens]
    
    def decode(token_indices) -> list:
        """
        Decode token indices back to original tokens (int or str).
        Args:
            token_indices: List of token indices
        Returns:
            List of tokens (can be int or str)
        """
        return [index_to_token[idx] for idx in token_indices]
    
    return encode, decode, vocab_size, index_to_string, string_to_index


def split_train_val_sequences(sequences: list[list[int]], train_ratio: float = 0.9):
    """Split sequences into training and validation sets."""
    n_train = int(train_ratio * len(sequences))
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:]
    return train_sequences, val_sequences


def get_batch_from_sequences(sequences: list[list[int]], block_size: int, batch_size: int):
    """
    Sample batches that respect sequence boundaries.
    Only samples from sequences long enough to contain a block.
    
    Args:
        sequences: List of sequences (each is a list of integers)
        block_size: Size of the context block
        batch_size: Number of samples in the batch
        
    Returns:
        X, Y tensors of shape (batch_size, block_size)
    """
    # Filter sequences that are long enough
    valid_sequences = [seq for seq in sequences if len(seq) >= block_size + 1]
    if not valid_sequences:
        raise ValueError(f"No sequences long enough for block_size {block_size}")
    
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        seq = random.choice(valid_sequences)
        if len(seq) <= block_size + 1:
            start_idx = 0
        else:
            start_idx = random.randint(0, len(seq) - block_size - 1)
        
        x = seq[start_idx:start_idx + block_size]
        y = seq[start_idx + 1:start_idx + block_size + 1]
        batch_x.append(x)
        batch_y.append(y)
    
    return torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)
