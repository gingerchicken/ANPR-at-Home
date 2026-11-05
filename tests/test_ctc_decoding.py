"""Check CTC path sums, repeated labels and honest bounded decoding output."""

import itertools
import json
import math

import pytest

from anpr.alphabet import CHARACTERS
from anpr.ctc_decoding import ctc_log_score, ctc_position_marginals, ctc_sequence_log_score, decode_ctc_logits


def frame_logits(*preferred):
    """Make a peaked frame fixture without removing any of the 37 vocabulary classes."""
    # Every frame retains finite scores for blank and the complete plate alphabet.
    rows = []
    for character in preferred:
        # None denotes CTC blank, which merges repeats before disappearing from the text.
        row = [-12.0] * 37
        row[0 if character is None else CHARACTERS.index(character) + 1] = 8.0
        rows.append(row)
    return rows


def collapse_path(path):
    """Implement the definition of CTC collapse independently of the dynamic program."""
    # Repeats collapse before blanks disappear, so A/blank/A remains two visible A characters.
    symbols = []
    previous = None
    for index in path:
        if index != previous and index != 0:
            symbols.append(CHARACTERS[index - 1])
        previous = index
    return "".join(symbols)


@pytest.mark.parametrize("plate", ["", "A", "AA", "AB", "ABA"])
def test_exact_forward_score_matches_all_enumerated_paths(plate):
    """The CTC score is a sum over alignments, including repeats and optional blanks."""
    # Three frames make 37^3 paths affordable and exercise all possible transition rules.
    logits = [[0.3, 1.2, -0.2] + [-8.0] * 34, [1.1, 0.7, 0.2] + [-8.0] * 34, [-0.5, 0.9, 1.4] + [-8.0] * 34]
    probabilities = [
        [math.exp(score) / sum(math.exp(entry) for entry in row) for score in row] for row in logits
    ]
    expected = math.fsum(
        math.prod(probabilities[frame][index] for frame, index in enumerate(path))
        for path in itertools.product(range(37), repeat=3)
        if collapse_path(path) == plate
    )
    actual = math.exp(ctc_sequence_log_score(logits, plate))
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_blank_separates_repeated_characters_and_adjacent_frames_do_not():
    """A duplicated registration character needs a blank between its emitted runs."""
    assert ctc_sequence_log_score(frame_logits("A", "A"), "AA") == -math.inf
    assert ctc_sequence_log_score(frame_logits("A", None, "A"), "AA") > -1e-5
    decoded = decode_ctc_logits(frame_logits("A", None, "A"), top_k=3, max_length=2)
    assert decoded["candidates"][0]["plate"] == "AA"
    assert len(decoded["positions"]) == 2
    assert (
        decoded["alignment_positions"][0]["alignment_frame"]
        < decoded["alignment_positions"][1]["alignment_frame"]
    )


def test_prefix_merging_ranks_strings_by_all_paths_instead_of_greedy_frames():
    """Several paths producing one string can outweigh one locally stronger frame path."""
    # Blank is the per-frame argmax, but A collects three nonempty alignments and beats empty.
    logits = [[math.log(0.6), math.log(0.4)] + [-100.0] * 35] * 2
    decoded = decode_ctc_logits(logits, top_k=2, max_length=2)
    assert decoded["candidates"][0]["plate"] == "A"
    assert math.exp(decoded["candidates"][0]["log_score"]) == pytest.approx(0.64)
    assert decoded["metadata"]["candidate_scores_exact"] is True
    assert decoded["metadata"]["search_exact"] is False


def test_beam_results_match_exhaustive_small_sequence_ranking():
    """A sufficiently wide tiny-frame beam must retain the correct distinct strings."""
    # Two frames admit at most 36 + 36^2 strings; a strong A/B/blank fixture has known leaders.
    logits = [[0.8, 1.4, 1.0] + [-25.0] * 34, [1.2, 0.4, 1.5] + [-25.0] * 34]
    possible = list(CHARACTERS) + [
        first + second for first in CHARACTERS for second in CHARACTERS if first != second
    ]
    expected = sorted(
        ((plate, ctc_sequence_log_score(logits, plate)) for plate in possible),
        key=lambda candidate: (-candidate[1], candidate[0]),
    )[:5]
    decoded = decode_ctc_logits(logits, top_k=5, max_length=2, beam_width=1000)
    assert [candidate["plate"] for candidate in decoded["candidates"]] == [plate for plate, _ in expected]
    assert [candidate["log_score"] for candidate in decoded["candidates"]] == pytest.approx(
        [score for _, score in expected]
    )


def test_explicit_formats_and_lookup_keep_visual_scores_separate():
    """A constrained lookup scores even weak eligible plates directly rather than through the beam."""
    # Digit alternatives are weaker than letters, but an explicit format can request their evaluation.
    logits = frame_logits("A", "B")
    logits[1][CHARACTERS.index("1") + 1] = 7.0
    decoded = decode_ctc_logits(logits, pattern="LD", max_length=2, top_k=3)
    assert decoded["candidates"][0]["plate"] == "A1"
    lookup = decode_ctc_logits(
        logits, known_plates=["Z9", "A1", "A 1", "AA"], pattern="LD", max_length=2, top_k=3
    )
    assert [candidate["plate"] for candidate in lookup["candidates"]] == ["A1", "Z9"]
    assert lookup["metadata"]["search_exact"] is True
    for candidate in lookup["candidates"]:
        assert candidate["log_score"] == pytest.approx(ctc_sequence_log_score(logits, candidate["plate"]))


def test_full_frame_and_aligned_character_alternatives_are_preserved():
    """Full-distribution marginals and best-path snapshots must remain distinct."""
    decoded = decode_ctc_logits(frame_logits("A", None, "B"), top_k=2, max_length=2)
    assert len(decoded["frames"]) == 3
    assert len(decoded["positions"]) == 2
    assert decoded["metadata"]["position_probability_kind"] == "ctc_ordered_output_marginal_including_absence"
    assert decoded["metadata"]["alignment_probability_kind"] == "viterbi_frame_softmax_not_character_marginal"
    assert len(decoded["alignment_positions"]) == 2
    for observation in decoded["frames"] + decoded["positions"] + decoded["alignment_positions"]:
        assert len(observation["alternatives"]) == 37
        assert math.fsum(
            alternative["probability"] for alternative in observation["alternatives"]
        ) == pytest.approx(1)
        assert any(
            alternative["index"] == 0 and alternative["character"] is None
            for alternative in observation["alternatives"]
        )
    json.dumps(decoded, allow_nan=False)


@pytest.mark.parametrize("max_length", [1, 2, 4])
def test_ordered_character_marginals_match_all_collapsed_paths(max_length):
    """Ordered characters, absence and overflow must aggregate the entire CTC distribution."""
    # Three full-vocabulary frames permit exhaustive verification independent of decoder recurrence.
    logits = [[0.3, 1.2, -0.2] + [-8.0] * 34, [1.1, 0.7, 0.2] + [-8.0] * 34, [-0.5, 0.9, 1.4] + [-8.0] * 34]
    probabilities = [
        [math.exp(score) / sum(math.exp(entry) for entry in row) for score in row] for row in logits
    ]
    # Ground-truth accumulators contain absence at index zero, not CTC frame blank.
    expected_terms = [[[] for _ in range(37)] for _ in range(max_length)]
    overflow_terms = []
    for path in itertools.product(range(37), repeat=3):
        # Collapse first, then inspect each ordered output event exactly once per frame path.
        plate = collapse_path(path)
        mass = math.prod(probabilities[frame][index] for frame, index in enumerate(path))
        if len(plate) > max_length:
            overflow_terms.append(mass)
        for position in range(max_length):
            class_index = CHARACTERS.index(plate[position]) + 1 if position < len(plate) else 0
            expected_terms[position][class_index].append(mass)
    # Accurate independent sums avoid thousands of tiny additions obscuring a correct log-domain result.
    expected_positions = [[math.fsum(terms) for terms in position] for position in expected_terms]
    expected_overflow = math.fsum(overflow_terms)
    actual = ctc_position_marginals(logits, max_length=max_length)
    assert len(actual["positions"]) == max_length
    for position, observation in enumerate(actual["positions"]):
        # Probability presentation order may differ, but class-index events are fixed.
        observed = {
            alternative["index"]: alternative["probability"] for alternative in observation["alternatives"]
        }
        assert [observed[index] for index in range(37)] == pytest.approx(
            expected_positions[position], abs=1e-12
        )
        assert math.fsum(observed.values()) == pytest.approx(1, abs=1e-12)
    assert actual["output_length_overflow_probability"] == pytest.approx(expected_overflow, abs=1e-12)


def test_length_overflow_does_not_condition_or_renormalize_character_marginals():
    """A first character from an over-capacity string still belongs in the unconditional marginal."""
    # AB has mass0.7 and B has mass0.3; a one-character candidate constraint must not hide AB's first A.
    logits = [[math.log(0.3), math.log(0.7)] + [-100.0] * 35, [-100.0, -100.0, 0.0] + [-100.0] * 34]
    decoded = decode_ctc_logits(logits, top_k=1, max_length=1)
    probabilities = {
        alternative["character"]: alternative["probability"]
        for alternative in decoded["positions"][0]["alternatives"]
    }
    assert decoded["candidates"][0]["plate"] == "B"
    assert probabilities["A"] == pytest.approx(0.7)
    assert probabilities["B"] == pytest.approx(0.3)
    assert decoded["output_length_overflow_probability"] == pytest.approx(0.7)
    assert (
        decoded["metadata"]["positions_conditioned_on"] == "full_ctc_distribution_not_beam_or_best_alignment"
    )


def test_ordered_marginals_match_hand_checked_blank_repeat_reference():
    """A compact independent reference checks absence, repeat transitions and overflow together."""
    # Only blank/A/B have material mass: the remaining vocabulary scores stay finite and negligible.
    logits = [[math.log(0.5), math.log(0.3), math.log(0.2)] + [-100.0] * 34] * 3
    result = ctc_position_marginals(logits, max_length=2)
    observed = [
        {alternative["index"]: alternative["probability"] for alternative in position["alternatives"]}
        for position in result["positions"]
    ]
    assert [observed[0][index] for index in range(3)] == pytest.approx([0.125, 0.525, 0.35])
    assert [observed[1][index] for index in range(3)] == pytest.approx([0.665, 0.177, 0.158])
    assert result["output_length_overflow_probability"] == pytest.approx(0.03)


def test_character_marginals_ignore_beam_and_lookup_but_alignment_snapshots_change():
    """Search priors may select another candidate without changing visual model evidence."""
    # The exact distribution is shared by a narrow AB beam, a forced B lookup and a one-letter format.
    logits = [[math.log(0.3), math.log(0.7)] + [-100.0] * 35, [-100.0, -100.0, 0.0] + [-100.0] * 34]
    narrow = decode_ctc_logits(logits, top_k=1, max_length=2, beam_width=1)
    lookup = decode_ctc_logits(logits, top_k=1, max_length=2, known_plates=["B"])
    constrained = decode_ctc_logits(logits, top_k=1, max_length=2, pattern="L")
    assert narrow["candidates"][0]["plate"] == "AB"
    assert lookup["candidates"][0]["plate"] == constrained["candidates"][0]["plate"] == "B"
    assert narrow["positions"] == lookup["positions"] == constrained["positions"]
    assert len(narrow["alignment_positions"]) == 2
    assert len(lookup["alignment_positions"]) == len(constrained["alignment_positions"]) == 1
    assert len(constrained["positions"]) == 2


def test_frame_blank_and_output_absence_have_distinct_meanings():
    """A blank separating repeated characters does not mean that the second character is absent."""
    decoded = decode_ctc_logits(frame_logits("A", None, "A"), top_k=1, max_length=4)
    second_output = {
        alternative["index"]: alternative["probability"]
        for alternative in decoded["positions"][1]["alternatives"]
    }
    second_frame = {
        alternative["index"]: alternative["probability"]
        for alternative in decoded["frames"][1]["alternatives"]
    }
    assert second_output[1] > 0.99999
    assert second_output[0] < 1e-5
    assert second_frame[0] > 0.99999
    last_absence = decoded["positions"][3]["alternatives"][0]
    assert last_absence["index"] == 0 and last_absence["character"] is None
    assert last_absence["probability"] == pytest.approx(1, abs=1e-12)


def test_uniform_or_blank_dominated_evidence_is_unreadable_without_discarding_scores():
    """A bounded list of guesses is not evidence that a plate is readable."""
    uniform = decode_ctc_logits([[0.0] * 37] * 3, top_k=3, max_length=2)
    blank = decode_ctc_logits(frame_logits(None, None, None), top_k=3, max_length=2)
    assert uniform["status"] == "unreadable"
    assert blank["status"] == "unreadable"
    assert uniform["candidates"]
    assert blank["candidates"]


def test_impossible_lookup_alignment_is_skipped_without_nonfinite_json():
    """Zero-probability registrations must not export an Infinity score."""
    decoded = decode_ctc_logits(frame_logits("A", "A"), known_plates=["AA"], max_length=2)
    assert decoded["candidates"] == []
    assert decoded["status"] == "unreadable"
    json.dumps(decoded, allow_nan=False)


def test_normalized_log_score_and_extreme_logits_remain_stable():
    """Log-space computation must avoid softmax overflow and preserve very unlikely candidates."""
    logits = [[10_000.0, 10_001.0] + [-10_000.0] * 35] * 3
    decoded = decode_ctc_logits(logits, known_plates=["A"], max_length=2)
    assert math.isfinite(decoded["candidates"][0]["log_score"])
    # Exact zero class probabilities are valid normalized inputs for the low-level score helper.
    rows = [[-math.inf] * 37 for _ in range(3)]
    rows[0][1] = rows[1][0] = rows[2][1] = 0.0
    assert ctc_log_score(rows, "AA") == 0.0
    assert ctc_log_score(rows, "A") == -math.inf


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"beam_width": 1, "top_k": 2},
        {"temperature": 0},
        {"temperature": math.nan},
        {"max_length": 0},
        {"known_plates": []},
        {"pattern": "LX"},
        {"min_position_confidence": 1.1},
        {"top_k": 1001},
        {"beam_width": 1001},
        {"max_length": 33},
    ],
)
def test_invalid_search_configuration_fails_clearly(options):
    """Malformed settings must fail before allocating or ranking beam states."""
    with pytest.raises(ValueError):
        decode_ctc_logits(frame_logits("A", "B"), **options)


def test_malformed_logits_fail_without_mutating_caller_rows():
    """Frame shape, finiteness and normalized low-level inputs are part of the score contract."""
    for logits in [[], [[0.0] * 36], [[math.nan] * 37], [[0.0] * 37] * 257]:
        with pytest.raises(ValueError):
            decode_ctc_logits(logits)
    with pytest.raises(ValueError):
        ctc_log_score([[0.0] * 37], "A")
