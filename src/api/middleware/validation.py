"""sgRNA sequence validation middleware for ChromaGuide API.

Validates sgRNA sequences for:
- Length: 20-23 base pairs
- Nucleotides: Only valid ACGT bases
"""

import re
from typing import Tuple


class InvalidSgrnaError(Exception):
    """Raised when sgRNA sequence validation fails."""
    pass


def validate_sgrna_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate sgRNA sequence.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if sequence is valid
        - error_message: Empty string if valid, error description otherwise
        
    Raises:
        InvalidSgrnaError: If sequence fails validation
    """
    if not sequence:
        raise InvalidSgrnaError("Sequence cannot be empty")
    
    # Clean up whitespace and convert to uppercase
    sequence = sequence.strip().upper()
    
    # Check for valid nucleotides only (ACGT)
    if not re.match(r'^[ACGT]+$', sequence):
        invalid_bases = set(sequence) - set('ACGT')
        raise InvalidSgrnaError(
            f"Invalid nucleotides found: {', '.join(sorted(invalid_bases))}. "
            "Only ACGT are allowed."
        )
    
    # Check length (20-23 bp)
    length = len(sequence)
    if length < 20 or length > 23:
        raise InvalidSgrnaError(
            f"Sequence length {length}bp is invalid. "
            "Must be between 20-23 base pairs."
        )
    
    return True, ""


def validate_sgrna_batch(sequences: list) -> Tuple[list, list]:
    """
    Validate a batch of sgRNA sequences.
    
    Args:
        sequences: List of sequence strings
        
    Returns:
        Tuple of (valid_sequences, error_messages)
        - valid_sequences: List of valid sequences
        - error_messages: List of validation error messages
    """
    valid_sequences = []
    error_messages = []
    
    for i, seq in enumerate(sequences):
        try:
            validate_sgrna_sequence(seq)
            valid_sequences.append(seq.strip().upper())
            error_messages.append(None)
        except InvalidSgrnaError as e:
            valid_sequences.append(None)
            error_messages.append(str(e))
    
    return valid_sequences, error_messages
