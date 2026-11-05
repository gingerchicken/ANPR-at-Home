"""Decode fixed-position OCR heads without discarding character uncertainty.

This is the notebook model's independent-head score, not CTC decoding and not a
calibrated probability that an entire registration is correct. Padding is a real
class and is permitted only after the last character.
"""

import heapq
import math
from collections.abc import Iterable, Sequence

from anpr.alphabet import CHARACTERS, normalize_plate


# These limits bound search allocation even when the caller supplies hostile sizes.
MAX_POSITIONS = 64
MAX_CANDIDATES = 1000
# The notebook reserves zero for padding and assigns the alphabet from index one.
CLASS_COUNT = len(CHARACTERS) + 1
CHARACTER_INDICES = {character: index + 1 for index, character in enumerate(CHARACTERS)}


def _bounded_integer(number: int, name: str, minimum: int, maximum: int) -> int:
    """Reject non-integral or out-of-range sizes before allocating search state."""
    # bool is an int subclass, but accepting True as a candidate budget is a typo.
    if isinstance(number, bool) or not isinstance(number, int) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return number


def _log_probabilities(logits: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    """Validate heads and compute stable temperature-scaled log probabilities."""
    # Shift before division to avoid overflowing an otherwise harmless large offset.
    log_probabilities = []
    for position, row in enumerate(logits):
        # Every head must use the saved notebook alphabet, including padding.
        if len(row) != CLASS_COUNT:
            raise ValueError(f"position {position} must contain {CLASS_COUNT} logits")
        # Keep a detached Python copy so no caller-owned logits are modified.
        scores = [float(score) for score in row]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("all logits must be finite")
        # Centered scores are nonpositive, making every exponential safe.
        maximum_score = max(scores)
        shifted_scores = [(score - maximum_score) / temperature for score in scores]
        if not all(math.isfinite(score) for score in shifted_scores):
            # An unrepresentable score cannot be exported as valid JSON later.
            raise ValueError("temperature-scaled logit differences exceed floating-point range")
        # fsum reduces rounding error while preserving all classes, even tiny ones.
        log_denominator = math.log(math.fsum(math.exp(score) for score in shifted_scores))
        log_probabilities.append([score - log_denominator for score in shifted_scores])
    return log_probabilities


def _ranking_key(candidate: tuple[str, float]) -> tuple[float, str]:
    """Order visual scores descending, with deterministic lexical tie breaking."""
    # A stable tie rule makes CPU runs and repeated lookup requests reproducible.
    plate, log_score = candidate
    return -log_score, plate


def _add_scores(first_score: float, second_score: float) -> float:
    """Add log scores while rejecting a non-JSON floating-point overflow."""
    # Ordinary OCR scores are far from this limit; reject pathological inputs clearly.
    combined_score = first_score + second_score
    if not math.isfinite(combined_score):
        raise ValueError("combined plate log score exceeds floating-point range")
    return combined_score


def _matches_pattern(plate: str, pattern: str | None) -> bool:
    """Apply only an explicitly requested letter/digit/alphanumeric format."""
    # Without a declared format, visual evidence may rank any supported character.
    if pattern is None:
        return True
    if len(plate) != len(pattern):
        return False
    # Character classes are deliberately ASCII, matching the trained alphabet.
    return all(
        mask == "A"
        or (mask == "L" and character in CHARACTERS[:26])
        or (mask == "D" and character in CHARACTERS[26:])
        for character, mask in zip(plate, pattern)
    )


def _known_candidates(
    known_plates: Iterable[str],
    log_probabilities: list[list[float]],
    suffix_padding_scores: list[float],
    top_k: int,
    min_length: int,
    max_length: int,
    pattern: str | None,
) -> list[tuple[str, float]]:
    """Score every eligible lookup entry directly, without a local character cutoff."""
    # The set prevents spelling variants such as 'AB 12' from duplicating candidates.
    seen_plates = set()

    def scored_plates():
        """Stream lookup scores to a bounded heap while validating every entry."""
        # A lookup entry absent from the image is still scored, rather than filtered by beam.
        for supplied_plate in known_plates:
            # Normalization follows the same rules as training labels.
            plate = normalize_plate(supplied_plate)
            if not plate or any(character not in CHARACTER_INDICES for character in plate):
                raise ValueError("known plates must contain supported letters or digits")
            if plate in seen_plates:
                continue
            seen_plates.add(plate)
            if not min_length <= len(plate) <= max_length or not _matches_pattern(plate, pattern):
                # Ineligible lengths and explicit format mismatches are constraints, not priors.
                continue
            # Preserve multiplication of all heads in log space, including trailing padding.
            character_score = 0.0
            for position, character in enumerate(plate):
                # Each label maps to exactly one fixed output head.
                character_score = _add_scores(
                    character_score, log_probabilities[position][CHARACTER_INDICES[character]]
                )
            yield plate, _add_scores(character_score, suffix_padding_scores[len(plate)])

    # Only K scores are retained; the seen set grows with unique supplied lookup entries.
    return heapq.nsmallest(top_k, scored_plates(), key=_ranking_key)


def _generated_candidates(
    log_probabilities: list[list[float]],
    suffix_padding_scores: list[float],
    top_k: int,
    min_length: int,
    max_length: int,
    pattern: str | None,
) -> list[tuple[str, float]]:
    """Find exact top-K valid sequences using bounded independent-prefix search.

    At a fixed length every prefix shares the same possible suffixes, so a prefix
    below rank K cannot enter the final top K. For P positions, C nonpadding
    classes and budget K, numeric heap work is O(P K C log K). Copying and
    comparing strings adds a worst-case O(P) factor per extension/comparison.
    Live prefix/candidate strings occupy O(K P), alongside O(P C) probabilities;
    extensions are streamed rather than retaining a Cartesian product.
    """
    # Prefix state excludes padding; complete candidates add the remaining padding suffix.
    prefixes = [("", 0.0)]
    candidates = []
    for position in range(max_length):
        # A format selects an ASCII class at each position, never a learned prior.
        mask = pattern[position] if pattern else "A"
        allowed_characters = (
            CHARACTERS[:26] if mask == "L" else CHARACTERS[26:] if mask == "D" else CHARACTERS
        )
        # Enumerate at most K*C extensions, retaining the best K for the next position.
        extensions = (
            (
                prefix + character,
                _add_scores(prefix_score, log_probabilities[position][CHARACTER_INDICES[character]]),
            )
            for prefix, prefix_score in prefixes
            for character in allowed_characters
        )
        prefixes = heapq.nsmallest(top_k, extensions, key=_ranking_key)
        # A pattern fixes length; otherwise every allowed prefix length may terminate.
        length = position + 1
        if length >= min_length and (pattern is None or length == len(pattern)):
            # Include every remaining padding head instead of rewarding shortened strings.
            completed = [
                (prefix, _add_scores(prefix_score, suffix_padding_scores[length]))
                for prefix, prefix_score in prefixes
            ]
            candidates = heapq.nsmallest(top_k, candidates + completed, key=_ranking_key)
    return candidates


def decode_logits(
    logits: Sequence[Sequence[float]],
    *,
    temperature: float = 1.0,
    top_k: int = 10,
    min_length: int = 1,
    max_length: int | None = None,
    pattern: str | None = None,
    known_plates: Iterable[str] | None = None,
    min_position_confidence: float = 0.0,
) -> dict:
    """Return full character distributions and bounded ranked plate candidates.

    Input heads use index zero for trailing padding and indices 1..36 for
    ``CHARACTERS``. Lengths are 1..number of heads, optionally restricted by an
    explicit L/D/A format or known-plate iterable. Lookup scores remain visual
    scores; this function applies no database or registration-frequency prior.

    ``uncertain`` means candidates exist and pass a minimal evidence heuristic;
    it never means calibrated correctness. ``unreadable`` retains alternatives
    but flags uniform, weak forced, or empty evidence. Character softmax scores
    also require held-out calibration before interpreting them as accuracy.

    Raises ValueError for malformed heads, unsupported labels or constraints,
    nonfinite numbers, unrepresentable scores, or resource limits. Input is not
    mutated; no model execution, network access, or file IO occurs here.
    """
    # Limit both dimensions before starting search or retaining caller probabilities.
    position_count = _bounded_integer(len(logits), "number of positions", 1, MAX_POSITIONS)
    top_k = _bounded_integer(top_k, "top_k", 1, MAX_CANDIDATES)
    max_length = position_count if max_length is None else max_length
    max_length = _bounded_integer(max_length, "max_length", 1, position_count)
    min_length = _bounded_integer(min_length, "min_length", 1, max_length)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(min_position_confidence) or not 0 <= min_position_confidence <= 1:
        raise ValueError("min_position_confidence must be in [0, 1]")
    if pattern is not None:
        # Explicit formats must fit the configured length interval exactly.
        if (
            not isinstance(pattern, str)
            or not min_length <= len(pattern) <= max_length
            or any(mask not in "LDA" for mask in pattern)
        ):
            raise ValueError("pattern must use L/D/A and have a length within the configured range")
        max_length = len(pattern)
    if known_plates is not None:
        # Materialize iterables once to distinguish an empty configuration from no lookup.
        if isinstance(known_plates, (str, bytes)):
            raise ValueError("known_plates must be an iterable of plate strings")
        known_plates = list(known_plates)
        if not known_plates:
            raise ValueError("known_plates must not be empty")

    # Store full distributions independently of candidate restrictions and rank truncation.
    log_probabilities = _log_probabilities(logits, temperature)
    positions = []
    for position, row in enumerate(log_probabilities):
        # Padding remains visible as null rather than a character that could enter a plate.
        alternatives = [
            {
                "index": index,
                "character": None if index == 0 else CHARACTERS[index - 1],
                "probability": math.exp(log_probability),
            }
            for index, log_probability in enumerate(row)
        ]
        alternatives.sort(key=lambda alternative: (-alternative["probability"], alternative["index"]))
        positions.append({"position": position, "alternatives": alternatives})
    # Suffix scores include all omitted character heads when a shorter plate terminates.
    suffix_padding_scores = [0.0] * (position_count + 1)
    for position in range(position_count - 1, -1, -1):
        # A suffix consists entirely of class-zero padding, never internal blank removal.
        suffix_padding_scores[position] = _add_scores(
            log_probabilities[position][0], suffix_padding_scores[position + 1]
        )
    if known_plates is None:
        # Exact prefix search needs no Cartesian enumeration of all complete registrations.
        ranked_candidates = _generated_candidates(
            log_probabilities, suffix_padding_scores, top_k, min_length, max_length, pattern
        )
    else:
        # Known registrations are evaluated directly so weak but eligible characters survive.
        ranked_candidates = _known_candidates(
            known_plates, log_probabilities, suffix_padding_scores, top_k, min_length, max_length, pattern
        )

    # The minimum is an abstention heuristic, not a learned or calibrated decision boundary.
    evidence_floor = max(min_position_confidence, 1.0 / CLASS_COUNT + 1e-12)
    status = "unreadable"
    status_reason = "no_candidates_match_constraints"
    if ranked_candidates:
        # Include trailing padding evidence when evaluating the best complete sequence.
        best_plate = ranked_candidates[0][0]
        selected_indices = [CHARACTER_INDICES[character] for character in best_plate]
        selected_indices += [0] * (position_count - len(best_plate))
        selected_probabilities = [
            math.exp(log_probabilities[position][index]) for position, index in enumerate(selected_indices)
        ]
        if min(selected_probabilities) > evidence_floor:
            # Passing this floor permits ranked suggestions but never claims a correct read.
            status = "uncertain"
            status_reason = "ranked_candidates_are_not_calibrated_whole_plate_confidence"
        else:
            # Forced weak characters must remain visibly weak even in a one-entry lookup.
            status_reason = "uniform_or_weak_selected_character_evidence"
    return {
        "positions": positions,
        "candidates": [{"plate": plate, "log_score": log_score} for plate, log_score in ranked_candidates],
        "status": status,
        "status_reason": status_reason,
        "metadata": {
            "search_mode": "known_plates" if known_plates is not None else "generated",
            "score_kind": "independent_head_visual_log_score_including_trailing_padding",
            "calibrated_whole_plate_probability": False,
            "temperature": temperature,
            "top_k": top_k,
            "min_length": min_length,
            "max_length": max_length,
            "pattern": pattern,
            "padding_rule": "trailing_only",
            "heuristic_min_selected_probability": evidence_floor,
        },
    }
