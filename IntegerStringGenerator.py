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


def main():
    generator = OddEvenIndexRule(min_value=0, max_value=20)
    sequences = generator.generate_dataset(5, min_length=10, max_length=20)
    for seq in sequences:
        print(seq)


if __name__ == "__main__":
    main()

