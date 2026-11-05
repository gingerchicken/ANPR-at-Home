"""Shared plate labels for the positional recognizer, not a CTC alphabet."""

import re

# Class zero denotes absent trailing characters; all other classes follow this order.
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Checkpoints record this vocabulary so a changed ordering cannot silently corrupt labels.
NUM_CLASSES = len(CHARACTERS) + 1


def normalize_plate(plate: str) -> str:
    """Remove display separators and reject unsupported or empty registrations."""
    # Spaces and ASCII hyphens are presentation separators, not recognition targets.
    compact_plate = re.sub(r"[\s-]", "", plate)
    if not compact_plate or any(
        character not in CHARACTERS + CHARACTERS[:26].lower() for character in compact_plate
    ):
        # Reject unknown glyphs rather than silently changing a plate's identity.
        raise ValueError(f"Unsupported or empty plate: {plate!r}")
    # Validate before uppercasing: Unicode expansions such as sharp-s would invent two letters.
    return compact_plate.upper()


def encode_plate(plate: str, max_length: int = 7) -> list[int]:
    """Encode a registration with trailing zero padding; never truncate a label."""
    # A configured length is a model capacity, not an inferred jurisdiction rule.
    normalized_plate = normalize_plate(plate)
    if max_length < 1 or len(normalized_plate) > max_length:
        # Truncation would manufacture a different registration during training.
        raise ValueError(f"Plate {normalized_plate!r} exceeds capacity {max_length}")
    return [CHARACTERS.index(character) + 1 for character in normalized_plate] + [0] * (
        max_length - len(normalized_plate)
    )
