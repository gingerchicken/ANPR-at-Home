"""Score and decode CTC plate frames while retaining uncertainty and blank semantics.

Class zero is CTC blank, not the positional model's trailing padding. Sequence
scores sum every alignment collapsing to a registration. Generated candidate
search is bounded and approximate; returned candidate scores themselves are
recomputed exactly. Ordered character alternatives marginalize the full CTC
distribution, including an explicit absent-after-end class. Separate Viterbi
frame snapshots remain available for alignment diagnostics and status heuristics.
Neither type of model probability establishes calibrated correctness.
"""

import heapq
import math
from collections.abc import Iterable, Sequence

from anpr.alphabet import CHARACTERS, normalize_plate


# Limits bound frame storage and prefix search independently of input image size.
MAX_FRAMES = 256
MAX_LENGTH = 32
MAX_CANDIDATES = 1000
MAX_BEAM_WIDTH = 1000
# Blank occupies index zero; the shared plate alphabet remains unchanged at indices 1..36.
CLASS_COUNT = len(CHARACTERS) + 1
CHARACTER_INDICES = {character: index + 1 for index, character in enumerate(CHARACTERS)}
NEGATIVE_INFINITY = -math.inf


def _bounded_integer(number, name, minimum, maximum):
    """Reject mistaken sizes before building a beam or retaining frame distributions."""
    # Booleans are integer subclasses but do not describe intentional resource budgets.
    if isinstance(number, bool) or not isinstance(number, int) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return number


def _logadd(first_score, second_score):
    """Add two log-domain path weights without exponentiating a positive number."""
    # Impossible states contribute zero mass and must not create an undefined inf-minus-inf.
    if first_score == NEGATIVE_INFINITY:
        return second_score
    if second_score == NEGATIVE_INFINITY:
        return first_score
    # Ordering makes the exponent nonpositive and safely handles very unequal path weights.
    larger_score, smaller_score = max(first_score, second_score), min(first_score, second_score)
    return larger_score + math.log1p(math.exp(smaller_score - larger_score))


def _log_softmax(logits, temperature):
    """Validate raw frames and return stable normalized log probabilities."""
    # A scale has to be a real positive scalar, not a JSON boolean or missing calibration value.
    try:
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
    except (TypeError, OverflowError) as error:
        raise ValueError("temperature must be finite and positive") from error
    _bounded_integer(len(logits), "number of frames", 1, MAX_FRAMES)
    # Detached numeric rows avoid mutating model outputs supplied by another caller.
    log_probabilities = []
    for frame, supplied_row in enumerate(logits):
        if len(supplied_row) != CLASS_COUNT:
            raise ValueError(f"frame {frame} must contain {CLASS_COUNT} logits")
        # Converting each numeric score once keeps inner dynamic-programming loops lightweight.
        scores = [float(score) for score in supplied_row]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("all CTC logits must be finite")
        # Center before scaling so a large common offset cannot overflow softmax.
        peak = max(scores)
        centered = [(score - peak) / temperature for score in scores]
        if not all(math.isfinite(score) for score in centered):
            raise ValueError("temperature-scaled CTC differences exceed floating-point range")
        denominator = math.log(math.fsum(math.exp(score) for score in centered))
        log_probabilities.append([score - denominator for score in centered])
    return log_probabilities


def _validated_log_rows(log_probabilities):
    """Accept normalized log rows, including exact zero mass represented by -inf."""
    _bounded_integer(len(log_probabilities), "number of frames", 1, MAX_FRAMES)
    # Re-normalizing only tiny rounding error supports float32 torch log_softmax output.
    rows = []
    for supplied_row in log_probabilities:
        if len(supplied_row) != CLASS_COUNT:
            raise ValueError(f"each CTC frame must contain {CLASS_COUNT} log probabilities")
        # Negative infinity represents a legitimate zero-probability emission.
        row = [float(score) for score in supplied_row]
        if any(math.isnan(score) or score == math.inf or score > 1e-6 for score in row):
            raise ValueError("CTC log probabilities must be nonpositive and contain no NaN or +inf")
        peak = max(row)
        if peak == NEGATIVE_INFINITY:
            raise ValueError("a CTC frame must contain nonzero probability mass")
        # The helper accepts probabilities, not arbitrary logits or unnormalized path weights.
        denominator = peak + math.log(math.fsum(math.exp(score - peak) for score in row))
        if abs(denominator) > 1e-5:
            raise ValueError("CTC log probability rows must be normalized")
        rows.append([score - denominator for score in row])
    return rows


def _target_indices(plate):
    """Normalize nonempty display labels while retaining the empty CTC sequence."""
    if not isinstance(plate, str):
        raise ValueError("plate must be a supported registration string")
    # Empty output is a valid CTC event used to detect blank-dominated evidence.
    normalized_plate = normalize_plate(plate) if plate else ""
    return normalized_plate, [CHARACTER_INDICES[character] for character in normalized_plate]


def _extended_target(target_indices):
    """Insert optional blank states before, between and after target characters."""
    # This state layout permits repeats only when they visit an intervening blank.
    extended = [0]
    for index in target_indices:
        extended.extend((index, 0))
    return extended


def _sequence_score(log_probabilities, target_indices):
    """Sum every valid CTC alignment in O(frames * target length) time and O(length) space."""
    # Empty registrations have exactly one alignment: blank at every frame.
    if not target_indices:
        return sum(row[0] for row in log_probabilities)
    # Adjacent equal characters need their own separating blank frame.
    required_frames = len(target_indices) + sum(
        first == second for first, second in zip(target_indices, target_indices[1:])
    )
    if required_frames > len(log_probabilities):
        return NEGATIVE_INFINITY
    # Initial alignment can start at the leading blank or the first visible character.
    extended = _extended_target(target_indices)
    previous = [NEGATIVE_INFINITY] * len(extended)
    previous[0] = log_probabilities[0][0]
    previous[1] = log_probabilities[0][extended[1]]
    for row in log_probabilities[1:]:
        # Each state receives stay/advance paths; unequal visible symbols can also skip a blank.
        current = [NEGATIVE_INFINITY] * len(extended)
        for state, class_index in enumerate(extended):
            incoming = previous[state]
            if state > 0:
                incoming = _logadd(incoming, previous[state - 1])
            if state > 1 and class_index != 0 and class_index != extended[state - 2]:
                incoming = _logadd(incoming, previous[state - 2])
            current[state] = incoming + row[class_index]
        previous = current
    # A valid path ends on the final character or its optional trailing blank.
    return min(0.0, _logadd(previous[-1], previous[-2]))


def ctc_log_score(log_probabilities: Sequence[Sequence[float]], plate: str) -> float:
    """Return exact log P(plate | frames) from already normalized CTC log rows.

    Class zero is blank; no padding tokens appear in targets. The empty string is
    supported. Infeasible alignments or zero model probability return -inf, which
    callers must not export as a finite JSON candidate. Rows are validated and
    copied, with tiny float32 normalization error corrected. Cost is O(T*(C+L))
    for T frames, C=37 classes and L target characters.
    """
    # Validation separates the low-level probability contract from raw-logit decoding.
    rows = _validated_log_rows(log_probabilities)
    _, target_indices = _target_indices(plate)
    return _sequence_score(rows, target_indices)


def ctc_sequence_log_score(logits: Sequence[Sequence[float]], plate: str, temperature: float = 1.0) -> float:
    """Return exact CTC sequence log probability after temperature-scaled softmax.

    This sums all collapsed alignments, not just the greedy frame path. It does
    not imply calibrated correctness. The empty sequence is allowed; infeasible
    strings return -inf. Input frames are copied and never changed.
    """
    # All target scoring uses the same normalization as the public candidate decoder.
    rows = _log_softmax(logits, temperature)
    _, target_indices = _target_indices(plate)
    return _sequence_score(rows, target_indices)


def _allows_character(character, mask):
    """Apply only an explicitly supplied ASCII position-class restriction."""
    return (
        mask == "A"
        or (mask == "L" and character in CHARACTERS[:26])
        or (mask == "D" and character in CHARACTERS[26:])
    )


def _matches_pattern(plate, pattern):
    """Check a completed registration against an optional explicit format."""
    return pattern is None or (
        len(plate) == len(pattern)
        and all(_allows_character(character, mask) for character, mask in zip(plate, pattern))
    )


def _update_prefix(beam, prefix, ending_state, score):
    """Accumulate paths for one collapsed prefix and final blank/nonblank state."""
    if score == NEGATIVE_INFINITY:
        return
    # Reusing one entry merges alignments without treating them as distinct registrations.
    weights = beam.setdefault(prefix, [NEGATIVE_INFINITY, NEGATIVE_INFINITY])
    weights[ending_state] = _logadd(weights[ending_state], score)


def _prefix_candidates(rows, beam_width, min_length, max_length, pattern):
    """Run a full-alphabet CTC prefix beam with explicitly bounded approximate search.

    Numeric work is O(T * B * C * log(B*C)) for T frames, beam B and C=37;
    Python prefix copying/comparison adds a length factor. Live beam extensions
    use O(B*C*L) string/state space. No per-frame class cutoff drops alternatives.
    """
    # Before consuming frames, the empty prefix has a unit-probability blank ending.
    beam = {"": [0.0, NEGATIVE_INFINITY]}
    for row in rows:
        # A separate next-frame map prevents paths from consuming the same frame twice.
        next_beam = {}
        for prefix, (blank_score, nonblank_score) in beam.items():
            # Blank can follow either ending without changing the collapsed registration.
            total_score = _logadd(blank_score, nonblank_score)
            _update_prefix(next_beam, prefix, 0, total_score + row[0])
            for class_index, character in enumerate(CHARACTERS, start=1):
                # Repeating a current nonblank run stays on the same collapsed prefix.
                is_repeat = bool(prefix) and prefix[-1] == character
                if is_repeat:
                    _update_prefix(next_beam, prefix, 1, nonblank_score + row[class_index])
                if len(prefix) >= max_length:
                    # A completed registration can consume blanks or repeat its last run, but cannot grow.
                    continue
                if pattern is not None and not _allows_character(character, pattern[len(prefix)]):
                    continue
                # Appending an equal character requires a preceding blank; unequal symbols need none.
                source_score = blank_score if is_repeat else total_score
                _update_prefix(next_beam, prefix + character, 1, source_score + row[class_index])
        # Ranking partial prefix mass is approximate after pruning; ties remain reproducible.
        retained = heapq.nsmallest(
            beam_width, next_beam.items(), key=lambda entry: (-_logadd(*entry[1]), entry[0])
        )
        beam = dict(retained)
    # Exact re-scoring restores all alignments for retained strings, including paths lost during search.
    candidates = []
    for prefix in beam:
        if min_length <= len(prefix) <= max_length and _matches_pattern(prefix, pattern):
            score = _sequence_score(rows, [CHARACTER_INDICES[character] for character in prefix])
            if math.isfinite(score):
                candidates.append((prefix, score))
    return candidates


def _known_candidates(rows, known_plates, min_length, max_length, pattern, top_k):
    """Score every eligible unique lookup entry exactly, independently of beam retention."""
    # A bounded score heap avoids retaining N candidate scores; duplicate detection still costs O(N).
    seen = set()
    supplied_count = 0
    skipped_alignments = 0

    def scored_plates():
        """Validate and stream exact lookup scores to the bounded top-K selection."""
        nonlocal supplied_count, skipped_alignments
        for supplied_plate in known_plates:
            # Display separators cannot create duplicate registration candidates.
            supplied_count += 1
            plate, target_indices = _target_indices(supplied_plate)
            if not plate:
                raise ValueError("known plates must not contain empty registrations")
            if plate in seen:
                continue
            seen.add(plate)
            if not min_length <= len(plate) <= max_length or not _matches_pattern(plate, pattern):
                continue
            # Insufficient frames or exactly zero probability must not become JSON Infinity scores.
            score = _sequence_score(rows, target_indices)
            if not math.isfinite(score):
                skipped_alignments += 1
                continue
            yield plate, score

    # Every supplied entry is visited, even if a weak character is absent from a narrow beam.
    candidates = heapq.nsmallest(top_k, scored_plates(), key=lambda candidate: (-candidate[1], candidate[0]))
    if supplied_count == 0:
        raise ValueError("known_plates must not be empty")
    return candidates, skipped_alignments


def _alternatives(row):
    """Retain all probabilities, representing the caller's index-zero event as null."""
    # Zero means frame blank or ordered-output absence; metadata distinguishes those different events.
    alternatives = [
        {
            "index": index,
            "character": None if index == 0 else CHARACTERS[index - 1],
            "probability": math.exp(score),
        }
        for index, score in enumerate(row)
    ]
    return sorted(alternatives, key=lambda alternative: (-alternative["probability"], alternative["index"]))


def _ordered_output_marginals(rows, max_length):
    """Compute unconditional ordered-character events in O(T*K*C) time.

    B[j] holds paths ending blank with j collapsed characters; N[j,c] holds
    paths ending c at that count. A new c comes from B[j-1] or N[j-1,d] for
    d != c. Its emission is the unique event that creates output position j.
    Summing that event over frames gives its full probability: every possible
    suffix has total mass one, so no backward pass or best-string condition is
    needed. Counts above K can be discarded because they never return to a
    lower count. Prefix/suffix log sums exclude one class without subtraction.

    Storage is O(K*C), in addition to the input rows. Returned absence means
    collapsed length below the position, not a blank at any particular frame.
    """
    # Initially only the empty prefix has mass; there is no visible run ending at a character yet.
    blank_weights = [NEGATIVE_INFINITY] * (max_length + 1)
    blank_weights[0] = 0.0
    nonblank_weights = [[NEGATIVE_INFINITY] * len(CHARACTERS) for _ in range(max_length + 1)]
    # Each entry accumulates the event that emits this ordered character exactly once per path.
    emission_events = [[NEGATIVE_INFINITY] * len(CHARACTERS) for _ in range(max_length)]
    for row in rows:
        # A separate frame prevents an emission from being reused as its own predecessor.
        next_blank_weights = [NEGATIVE_INFINITY] * (max_length + 1)
        next_nonblank_weights = [[NEGATIVE_INFINITY] * len(CHARACTERS) for _ in range(max_length + 1)]
        for count in range(max_length + 1):
            # Prefix and suffix sums represent all previous run classes except the prospective repeat.
            prefix_sums = [NEGATIVE_INFINITY]
            for score in nonblank_weights[count]:
                prefix_sums.append(_logadd(prefix_sums[-1], score))
            suffix_sums = [NEGATIVE_INFINITY] * (len(CHARACTERS) + 1)
            for character_index in range(len(CHARACTERS) - 1, -1, -1):
                suffix_sums[character_index] = _logadd(
                    nonblank_weights[count][character_index], suffix_sums[character_index + 1]
                )
            # Blank consumes a frame without increasing collapsed length, whatever the previous ending.
            next_blank_weights[count] = _logadd(blank_weights[count], prefix_sums[-1]) + row[0]
            for character_index in range(len(CHARACTERS)):
                # Continuing one nonblank run does not emit another output character.
                continuation = nonblank_weights[count][character_index] + row[character_index + 1]
                next_nonblank_weights[count][character_index] = _logadd(
                    next_nonblank_weights[count][character_index], continuation
                )
                if count < max_length:
                    # A blank or a different preceding run emits the next ordered character.
                    other_runs = _logadd(prefix_sums[character_index], suffix_sums[character_index + 1])
                    emission = _logadd(blank_weights[count], other_runs) + row[character_index + 1]
                    next_nonblank_weights[count + 1][character_index] = _logadd(
                        next_nonblank_weights[count + 1][character_index], emission
                    )
                    # Suffix probabilities sum to one even when a future path exceeds the output cap.
                    emission_events[count][character_index] = _logadd(
                        emission_events[count][character_index], emission
                    )
        blank_weights, nonblank_weights = next_blank_weights, next_nonblank_weights
    # Final count masses provide exact absence events; discarded longer paths are reported as overflow.
    length_weights = []
    retained_weight = NEGATIVE_INFINITY
    for count in range(max_length + 1):
        # Either ending state represents the same collapsed output length.
        length_weight = blank_weights[count]
        for score in nonblank_weights[count]:
            length_weight = _logadd(length_weight, score)
        length_weights.append(length_weight)
        retained_weight = _logadd(retained_weight, length_weight)
    # expm1 avoids catastrophic cancellation when only a tiny amount of mass exceeds the cap.
    overflow_probability = max(0.0, -math.expm1(min(0.0, retained_weight)))
    positions = []
    absence_weight = NEGATIVE_INFINITY
    for position in range(max_length):
        # Position j is absent exactly when final collapsed length is less than j+1.
        absence_weight = _logadd(absence_weight, length_weights[position])
        marginal_row = [min(0.0, absence_weight)] + [min(0.0, score) for score in emission_events[position]]
        positions.append({"position": position, "alternatives": _alternatives(marginal_row)})
    return positions, overflow_probability


def ctc_position_marginals(
    logits: Sequence[Sequence[float]], max_length: int = 7, temperature: float = 1.0
) -> dict:
    """Return exact unconditional output-position probabilities and length overflow.

    Each of max_length positions retains A-Z, 0-9 and index-zero/null absence.
    Absence means that the collapsed string ended before this position; it is
    different from frame blank. Events include strings longer than max_length
    and are never conditioned on candidate filters, a beam or a selected length.
    Rows sum to one up to floating-point error. Overflow reports model mass for
    longer strings without renormalizing it away. These are model marginals,
    not empirically calibrated character correctness estimates.
    """
    # Capacity bounds output storage only; it does not condition the underlying sequence distribution.
    max_length = _bounded_integer(max_length, "max_length", 1, MAX_LENGTH)
    rows = _log_softmax(logits, temperature)
    positions, overflow_probability = _ordered_output_marginals(rows, max_length)
    return {
        "positions": positions,
        "output_length_overflow_probability": overflow_probability,
        "metadata": {
            "position_probability_kind": "ctc_ordered_output_marginal_including_absence",
            "score_kind": "ctc_ordered_output_marginal_including_absence",
            "positions_conditioned_on": "full_ctc_distribution_not_beam_or_best_alignment",
            "position_class_zero_meaning": "absent_after_output_end",
            "max_length": max_length,
            "frame_count": len(rows),
            "temperature": temperature,
            "calibrated_character_confidence": False,
        },
    }


def _aligned_positions(rows, plate):
    """Return best-path frame snapshots for each visible character, not character marginals."""
    # Viterbi uses the same extended-state transitions as the exact sum, replacing sums by maxima.
    target_indices = [CHARACTER_INDICES[character] for character in plate]
    extended = _extended_target(target_indices)
    previous = [NEGATIVE_INFINITY] * len(extended)
    previous[0], previous[1] = rows[0][0], rows[0][extended[1]]
    # Backpointers cost O(T*L), while numeric state scores use one frame of O(L) storage.
    backpointers = [[-1] * len(extended)]
    for row in rows[1:]:
        # Each new state retains its best predecessor to reconstruct one valid monotone alignment.
        current = [NEGATIVE_INFINITY] * len(extended)
        predecessors = [-1] * len(extended)
        for state, class_index in enumerate(extended):
            # Staying wins ties before advancing, making repeated-character alignment deterministic.
            source_states = [state]
            if state > 0:
                source_states.append(state - 1)
            if state > 1 and class_index != 0 and class_index != extended[state - 2]:
                source_states.append(state - 2)
            predecessor = max(source_states, key=previous.__getitem__)
            current[state] = previous[predecessor] + row[class_index]
            predecessors[state] = predecessor
        previous = current
        backpointers.append(predecessors)
    # The last visible state wins a final tie instead of manufacturing an extra blank requirement.
    state = max((len(extended) - 2, len(extended) - 1), key=previous.__getitem__)
    if not math.isfinite(previous[state]):
        raise ValueError("Cannot align a zero-probability CTC candidate")
    alignment = [state]
    for frame in range(len(rows) - 1, 0, -1):
        # Moving backwards follows exactly one path through the valid CTC state graph.
        state = backpointers[frame][state]
        alignment.append(state)
    alignment.reverse()
    # Each target state must have appeared at least once in any valid full-sequence path.
    positions = []
    for position, class_index in enumerate(target_indices):
        state_frames = [
            frame for frame, visited_state in enumerate(alignment) if visited_state == 2 * position + 1
        ]
        best_frame = max(state_frames, key=lambda frame: (rows[frame][class_index], -frame))
        positions.append(
            {
                "position": position,
                "character": plate[position],
                "alignment_frame": best_frame,
                "alignment_span": [state_frames[0], state_frames[-1] + 1],
                "alternatives": _alternatives(rows[best_frame]),
            }
        )
    return positions


def decode_ctc_logits(
    logits: Sequence[Sequence[float]],
    *,
    temperature: float = 1.0,
    top_k: int = 10,
    min_length: int = 1,
    max_length: int = 7,
    pattern: str | None = None,
    known_plates: Iterable[str] | None = None,
    min_position_confidence: float = 0.0,
    beam_width: int | None = None,
) -> dict:
    """Return ranked CTC registrations and full frame/aligned-character alternatives.

    The generated prefix beam is approximate and bounded. Eligible known plates
    are scored directly and exactly, without beam pruning. Returned sequence
    scores sum all CTC alignments and include blank behavior, not trailing padding.
    ``positions`` are exact ordered marginals under the full model distribution,
    with null meaning absent after the output ends. ``alignment_positions`` retain
    best-candidate Viterbi frame snapshots separately. Neither kind of row can be
    multiplied into a plate probability. ``uncertain`` never means calibrated correctness. Uniform,
    blank-dominated, weak selected-frame or absent candidates are unreadable.
    """
    # Bound output and search allocation before consuming model logits or lookup iterators.
    top_k = _bounded_integer(top_k, "top_k", 1, MAX_CANDIDATES)
    max_length = _bounded_integer(max_length, "max_length", 1, MAX_LENGTH)
    min_length = _bounded_integer(min_length, "min_length", 1, max_length)
    # Candidate constraints may shorten search, but must not condition or truncate the visual marginals.
    candidate_max_length = max_length
    beam_width = min(MAX_BEAM_WIDTH, max(100, 2 * top_k)) if beam_width is None else beam_width
    beam_width = _bounded_integer(beam_width, "beam_width", top_k, MAX_BEAM_WIDTH)
    if not math.isfinite(min_position_confidence) or not 0 <= min_position_confidence <= 1:
        raise ValueError("min_position_confidence must be in [0, 1]")
    if pattern is not None:
        # An explicit format fixes final length without introducing a learned text prior.
        if (
            not isinstance(pattern, str)
            or not min_length <= len(pattern) <= max_length
            or any(mask not in "LDA" for mask in pattern)
        ):
            raise ValueError("pattern must use L/D/A within the configured length range")
        candidate_max_length = len(pattern)
    if isinstance(known_plates, (str, bytes)):
        raise ValueError("known_plates must be an iterable of registration strings")
    # Raw frame distributions are preserved independently of candidate restrictions.
    rows = _log_softmax(logits, temperature)
    frames = [{"frame": frame, "alternatives": _alternatives(row)} for frame, row in enumerate(rows)]
    # Exact positions include every alignment and string length, independently of the bounded beam.
    positions, overflow_probability = _ordered_output_marginals(rows, max_length)
    skipped_alignments = 0
    if known_plates is None:
        # Every nonblank class participates; only prefix count is pruned for bounded inference.
        scored_candidates = _prefix_candidates(rows, beam_width, min_length, candidate_max_length, pattern)
        ranked_candidates = heapq.nsmallest(
            top_k, scored_candidates, key=lambda candidate: (-candidate[1], candidate[0])
        )
    else:
        # Closed-list lookup uses exact CTC scoring even when a character is visually unlikely.
        ranked_candidates, skipped_alignments = _known_candidates(
            rows, known_plates, min_length, candidate_max_length, pattern, top_k
        )
    # Empty emission is independently exact, even when a narrow beam would prune its prefix.
    empty_score = _sequence_score(rows, [])
    alignment_positions = _aligned_positions(rows, ranked_candidates[0][0]) if ranked_candidates else []
    # Record the exact evidence heuristic even when no eligible registration survives constraints.
    evidence_floor = max(min_position_confidence, 1.0 / CLASS_COUNT + 1e-12)
    status = "unreadable"
    status_reason = "no_candidates_match_constraints_or_feasible_alignment"
    if ranked_candidates:
        # The status threshold is a frame-evidence heuristic, never a calibrated decision rule.
        selected_probabilities = [
            math.exp(rows[position["alignment_frame"]][CHARACTER_INDICES[position["character"]]])
            for position in alignment_positions
        ]
        if all(max(row) - min(row) <= 1e-12 for row in rows):
            status_reason = "uniform_frame_evidence"
        elif empty_score >= ranked_candidates[0][1]:
            status_reason = "empty_sequence_dominates_nonempty_suggestions"
        elif min(selected_probabilities) <= evidence_floor:
            status_reason = "weak_selected_viterbi_frame_evidence"
        else:
            # Passing the heuristic permits ranked suggestions without asserting that any plate is correct.
            status = "uncertain"
            status_reason = "ranked_candidates_are_not_calibrated_whole_plate_confidence"
    return {
        "positions": positions,
        "alignment_positions": alignment_positions,
        "frames": frames,
        "output_length_overflow_probability": overflow_probability,
        "candidates": [{"plate": plate, "log_score": score} for plate, score in ranked_candidates],
        "status": status,
        "status_reason": status_reason,
        "metadata": {
            "model_kind": "ctc",
            "score_kind": "ctc_sequence_log_probability_sum_over_alignments",
            "search_mode": "ctc_prefix_beam" if known_plates is None else "known_plates",
            "search_exact": known_plates is not None,
            "candidate_scores_exact": True,
            "beam_width": beam_width if known_plates is None else None,
            "top_k": top_k,
            "min_length": min_length,
            "max_length": max_length,
            "candidate_max_length": candidate_max_length,
            "pattern": pattern,
            "temperature": temperature,
            "frame_count": len(rows),
            "blank_index": 0,
            "frame_class_zero_meaning": "ctc_blank",
            "alignment_class_zero_meaning": "ctc_blank",
            "position_absence_index": 0,
            "position_class_zero_meaning": "absent_after_output_end",
            "padding_rule": "ctc_collapse_not_padding",
            "empty_sequence_log_score": empty_score if math.isfinite(empty_score) else None,
            "skipped_zero_probability_or_infeasible_lookup_entries": skipped_alignments,
            "positions_conditioned_on": "full_ctc_distribution_not_beam_or_best_alignment",
            "position_probability_kind": "ctc_ordered_output_marginal_including_absence",
            "alignment_probability_kind": "viterbi_frame_softmax_not_character_marginal",
            "calibrated_character_confidence": False,
            "heuristic_min_selected_frame_probability": evidence_floor,
            "calibrated_whole_plate_probability": False,
        },
    }
