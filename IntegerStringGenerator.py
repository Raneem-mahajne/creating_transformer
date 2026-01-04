import random
import torch
from abc import ABC, abstractmethod


class IntegerStringGenerator(ABC):
    """
    Superclass for generating integer strings with specific rules.
    Subclasses should implement the generate_sequence method to define
    their specific rule.
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        """
        Args:
            min_value: Minimum integer value (inclusive)
            max_value: Maximum integer value (inclusive)
            sequence_length: Length of sequences to generate (None for variable length)
        """
        self.min_value = min_value
        self.max_value = max_value
        self.sequence_length = sequence_length
        
    @abstractmethod
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a single sequence following the rule.
        
        Args:
            length: Length of the sequence to generate
            
        Returns:
            List of integers following the rule
        """
        pass
    
    def generate_dataset(self, num_sequences: int, min_length: int = 10, max_length: int = 100) -> list[list[int]]:
        """
        Generate multiple sequences for training.
        
        Args:
            num_sequences: Number of sequences to generate
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            
        Returns:
            List of sequences (each sequence is a list of integers)
        """
        sequences = []
        for _ in range(num_sequences):
            if self.sequence_length is not None:
                length = self.sequence_length
            else:
                length = random.randint(min_length, max_length)
            sequence = self.generate_sequence(length)
            sequences.append(sequence)
        return sequences
    


class OddEvenIndexRule(IntegerStringGenerator):
    """
    Rule: Odd indices (1-indexed) have odd numbers, even indices (1-indexed) have even numbers.
    Note: Python uses 0-indexing, so:
    - Index 0 (even in 0-index) gets even numbers
    - Index 1 (odd in 0-index) gets odd numbers
    - Index 2 (even in 0-index) gets even numbers
    - etc.
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        super().__init__(min_value, max_value, sequence_length)
        # Precompute even and odd number lists once
        self.even_nums = [n for n in range(min_value, max_value + 1) if n % 2 == 0]
        self.odd_nums = [n for n in range(min_value, max_value + 1) if n % 2 == 1]
        
        # Fallback values if lists are empty (shouldn't happen with reasonable ranges)
        self.even_fallback = min_value
        self.odd_fallback = min(min_value + 1, max_value) if min_value % 2 == 0 else min_value
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a sequence where:
        - Even indices (0, 2, 4, ...) get even numbers
        - Odd indices (1, 3, 5, ...) get odd numbers
        """
        sequence = []
        for i in range(length):
            if i % 2 == 0:  # Even index
                sequence.append(random.choice(self.even_nums) if self.even_nums else self.even_fallback)
            else:  # Odd index
                sequence.append(random.choice(self.odd_nums) if self.odd_nums else self.odd_fallback)
        return sequence


class EvenToOddTransitionRule(IntegerStringGenerator):
    """
    Rule: If the current number is even, the next number must be odd.
    If the current number is odd, the next number can be anything.
    Starts with a random number.
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        super().__init__(min_value, max_value, sequence_length)
        # Precompute all numbers, even numbers, and odd numbers
        self.all_nums = list(range(min_value, max_value + 1))
        self.odd_nums = [n for n in range(min_value, max_value + 1) if n % 2 == 1]
        
        # Fallback values if lists are empty
        self.fallback = min_value
        self.odd_fallback = min(min_value + 1, max_value) if min_value % 2 == 0 else min_value
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a sequence where:
        - If current number is even, next must be odd
        - If current number is odd, next can be anything
        - Starts with a random number
        """
        if length == 0:
            return []
        
        sequence = []
        # Start with a random number
        current = random.choice(self.all_nums) if self.all_nums else self.fallback
        sequence.append(current)
        
        # Generate the rest based on the rule
        for _ in range(length - 1):
            if current % 2 == 0:  # Current is even, next must be odd
                current = random.choice(self.odd_nums) if self.odd_nums else self.odd_fallback
            else:  # Current is odd, next can be anything
                current = random.choice(self.all_nums) if self.all_nums else self.fallback
            sequence.append(current)
        
        return sequence


class EvenRepeatLastOddRule(IntegerStringGenerator):
    """
    Rule: If the current number is even, the next number should be the same as the last odd number.
    If there is no last odd number, then random number.
    If the current number is odd, the next number can be anything.
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        super().__init__(min_value, max_value, sequence_length)
        self.all_nums = list(range(min_value, max_value + 1))
        self.fallback = min_value
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a sequence where:
        - If current number is even, next is the last odd number seen (or random if none)
        - If current number is odd, next can be anything
        """
        if length == 0:
            return []
        
        sequence = []
        last_odd = None
        
        # Start with a random number
        current = random.choice(self.all_nums) if self.all_nums else self.fallback
        sequence.append(current)
        if current % 2 == 1:
            last_odd = current
        
        # Generate the rest based on the rule
        for _ in range(length - 1):
            if current % 2 == 0:  # Current is even
                if last_odd is not None:
                    current = last_odd
                else:
                    current = random.choice(self.all_nums) if self.all_nums else self.fallback
            else:  # Current is odd, next can be anything
                current = random.choice(self.all_nums) if self.all_nums else self.fallback
            
            sequence.append(current)
            if current % 2 == 1:
                last_odd = current
        
        return sequence


class EvenAbsDiffRule(IntegerStringGenerator):
    """
    Rule: If the current number is even, the next number is the absolute difference
    of the two numbers that came before it.
    If the current number is odd (or not enough history), the next number can be anything.
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        super().__init__(min_value, max_value, sequence_length)
        self.all_nums = list(range(min_value, max_value + 1))
        self.fallback = min_value
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a sequence where:
        - If current number is even, next is |seq[-2] - seq[-3]| (abs diff of two before it)
        - If current number is odd or not enough history, next can be anything
        """
        if length == 0:
            return []
        
        sequence = []
        
        # Start with a random number
        current = random.choice(self.all_nums) if self.all_nums else self.fallback
        sequence.append(current)
        
        # Generate the rest based on the rule
        for i in range(length - 1):
            if current % 2 == 0 and len(sequence) >= 2:
                # Current is even and we have at least 2 numbers before
                # Next number is |sequence[-1] - sequence[-2]| = |current - previous|
                prev = sequence[-2]
                current = abs(current - prev)
                # Clamp to valid range
                current = max(self.min_value, min(self.max_value, current))
            else:
                # Current is odd or not enough history, next can be anything
                current = random.choice(self.all_nums) if self.all_nums else self.fallback
            
            sequence.append(current)
        
        return sequence


class CopyModuloRule(IntegerStringGenerator):
    """
    Rule: Token at position i copies the token at position (i mod k).
    First k positions are random, then the pattern repeats.
    
    This is pedagogically ideal because:
    - Token embeddings must encode token identity (to copy correctly)
    - Positional embeddings must encode position mod k (to know which slot)
    - Attention should show clear bands (each position attends to same-slot positions)
    - Q/K should match positions in the same slot
    - V carries the token value to be copied
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None, period: int = 3):
        super().__init__(min_value, max_value, sequence_length)
        self.all_nums = list(range(min_value, max_value + 1))
        self.fallback = min_value
        self.period = period  # k - the modulo period
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a sequence where:
        - First k positions get random values
        - Position i >= k copies from position (i mod k)
        
        Example with k=3:
        [5, 12, 8, 5, 12, 8, 5, 12, 8, ...]
        """
        if length == 0:
            return []
        
        sequence = []
        k = self.period
        
        # First k positions: random values (these are the "template")
        for i in range(min(k, length)):
            val = random.choice(self.all_nums) if self.all_nums else self.fallback
            sequence.append(val)
        
        # Remaining positions: copy from position (i mod k)
        for i in range(k, length):
            source_pos = i % k
            sequence.append(sequence[source_pos])
        
        return sequence


class SuccessorRule(IntegerStringGenerator):
    """
    Rule: Next token = (current token + 1) mod vocab_size
    
    This is ideal for understanding the Value (V) matrix because:
    - V must encode "what comes after this token" (the successor)
    - Q/K work together to attend to the previous position
    - V[token=5] should encode information to output "6"
    - The token embedding encodes identity, V transforms it to successor
    
    Example: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12...
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        super().__init__(min_value, max_value, sequence_length)
        self.vocab_size = max_value - min_value + 1
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate a sequence where each token is the successor of the previous.
        Wraps around: max_value -> min_value
        """
        if length == 0:
            return []
        
        sequence = []
        
        # Start with a random number
        current = random.randint(self.min_value, self.max_value)
        sequence.append(current)
        
        # Each subsequent token is (current + 1) mod vocab_size, mapped to range
        for _ in range(length - 1):
            # Increment with wraparound
            current = self.min_value + ((current - self.min_value + 1) % self.vocab_size)
            sequence.append(current)
        
        return sequence


class ConditionalTransformRule(IntegerStringGenerator):
    """
    Rule: Transform depends on token value
    - If current token is EVEN: next = token // 2
    - If current token is ODD: next = (token + 1) mod vocab_size
    
    This clearly shows V's role because:
    - V must encode DIFFERENT transformations for even vs odd tokens
    - Looking at V vectors, even tokens should cluster differently than odd tokens
    - The model can't just use a uniform transformation
    
    Example: 6 -> 3 (6//2), 3 -> 4 (3+1), 4 -> 2 (4//2), 2 -> 1 (2//2), 1 -> 2 (1+1)...
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None):
        super().__init__(min_value, max_value, sequence_length)
        self.vocab_size = max_value - min_value + 1
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate sequence with conditional transformation.
        """
        if length == 0:
            return []
        
        sequence = []
        
        # Start with a random number
        current = random.randint(self.min_value, self.max_value)
        sequence.append(current)
        
        for _ in range(length - 1):
            if current % 2 == 0:  # Even
                next_val = current // 2
            else:  # Odd
                next_val = (current + 1) % self.vocab_size
            # Clamp to valid range
            current = max(self.min_value, min(self.max_value, next_val))
            sequence.append(current)
        
        return sequence


class LookupPermutationRule(IntegerStringGenerator):
    """
    Rule: Each token maps to exactly one other token via a fixed permutation.
    next_token = permutation[current_token]
    
    This PERFECTLY illustrates V's role because:
    - The model must attend to the previous token (Q/K)
    - V MUST encode the lookup table mapping each token to its permuted value
    - There's no position dependency - pure content-based transformation
    
    Example with permutation [7,3,9,0,5,2,8,1,4,6]:
    0→7, 1→3, 2→9, 3→0, 4→5, 5→2, 6→8, 7→1, 8→4, 9→6
    Sequence: 4, 5, 2, 9, 6, 8, 4, 5, 2, 9, 6, 8, ...
    """
    
    def __init__(self, min_value: int = 0, max_value: int = 20, sequence_length: int = None, seed: int = 42):
        super().__init__(min_value, max_value, sequence_length)
        self.vocab_size = max_value - min_value + 1
        # Create a fixed permutation (shuffle)
        rng = random.Random(seed)
        self.permutation = list(range(min_value, max_value + 1))
        rng.shuffle(self.permutation)
    
    def generate_sequence(self, length: int) -> list[int]:
        """
        Generate sequence following the permutation lookup.
        """
        if length == 0:
            return []
        
        sequence = []
        
        # Start with a random token
        current = random.randint(self.min_value, self.max_value)
        sequence.append(current)
        
        for _ in range(length - 1):
            # Next token is the permuted value
            current = self.permutation[current - self.min_value]
            sequence.append(current)
        
        return sequence


def main():
    generator = OddEvenIndexRule(min_value=0, max_value=20)
    sequences = generator.generate_dataset(5, min_length=10, max_length=20)
    for seq in sequences:
        print(seq)


if __name__ == "__main__":
    main()

