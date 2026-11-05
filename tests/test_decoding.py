"""Behavioral checks for ranked, uncertainty-preserving plate decoding."""

import itertools
import math
import unittest

from anpr.alphabet import CHARACTERS
from anpr.decoding import decode_logits


def position_logits(**character_scores):
    """Build a mostly suppressed head, using PAD for the no-character class."""
    # Unspecified classes remain possible, so tests detect premature truncation.
    logits = [-30.0] * (len(CHARACTERS) + 1)
    for character, score in character_scores.items():
        # PAD uses the notebook's zero class; letters and digits retain its order.
        index = 0 if character == "PAD" else CHARACTERS.index(character) + 1
        logits[index] = score
    return logits


class RankedDecodingTests(unittest.TestCase):
    """Check valid whole-plate ranking rather than a particular search algorithm."""

    def test_keeps_repeated_characters_and_all_position_alternatives(self):
        """Independent character heads must never collapse repeated letters."""
        # Two confident A heads represent AA, unlike CTC repeat handling.
        decoded = decode_logits([position_logits(A=8), position_logits(A=8)], min_length=2, top_k=3)
        self.assertEqual(decoded["candidates"][0]["plate"], "AA")
        self.assertEqual(len(decoded["positions"][0]["alternatives"]), 37)
        self.assertAlmostEqual(
            sum(alternative["probability"] for alternative in decoded["positions"][0]["alternatives"]), 1.0
        )
        self.assertEqual(decoded["status"], "uncertain")

    def test_matches_exhaustive_valid_sequences(self):
        """Top-K must match full enumeration, including the padding suffix score."""
        # All 1,332 valid one/two-character plates are small enough to enumerate.
        logits = [position_logits(A=3, B=2, PAD=0), position_logits(A=1, B=4, PAD=2)]
        decoded = decode_logits(logits, top_k=20)
        # Build independent reference log probabilities without decoder helpers.
        log_probabilities = []
        for row in logits:
            # Moderate scores keep direct exponentiation safe in this reference.
            denominator = sum(math.exp(score) for score in row)
            log_probabilities.append([score - math.log(denominator) for score in row])
        # Exhaustion considers valid trailing padding, never an internal blank.
        expected = []
        for length in (1, 2):
            for characters in itertools.product(CHARACTERS, repeat=length):
                # Fixed-position scores include every head, including padding.
                indices = [CHARACTERS.index(character) + 1 for character in characters]
                indices += [0] * (2 - length)
                expected.append(
                    (
                        "".join(characters),
                        sum(log_probabilities[position][index] for position, index in enumerate(indices)),
                    )
                )
        expected.sort(key=lambda candidate: (-candidate[1], candidate[0]))
        self.assertEqual(
            [candidate["plate"] for candidate in decoded["candidates"]], [plate for plate, _ in expected[:20]]
        )
        for candidate, (_, expected_score) in zip(decoded["candidates"], expected):
            # Scores remain full visual log scores, without shortlist renormalization.
            self.assertAlmostEqual(candidate["log_score"], expected_score)

    def test_short_plate_requires_only_trailing_padding(self):
        """A blank in the middle cannot silently move a later character left."""
        # The final confident A cannot form A-A by dropping the middle blank.
        decoded = decode_logits(
            [position_logits(A=5), position_logits(PAD=8, B=4), position_logits(A=5, PAD=1)], top_k=10
        )
        self.assertTrue(all("-" not in candidate["plate"] for candidate in decoded["candidates"]))
        self.assertEqual(decoded["candidates"][0]["plate"], "A")
        self.assertNotIn("AA", [candidate["plate"] for candidate in decoded["candidates"][:1]])

    def test_known_plates_are_scored_without_beam_filtering(self):
        """A lookup restriction must still score characters outside a local top-K."""
        # AZ is deliberately weaker than many unrestricted candidates.
        decoded = decode_logits(
            [position_logits(A=9), position_logits(B=8, Z=1)], known_plates=["a z", "AZ", "BB"], top_k=1
        )
        self.assertEqual(decoded["candidates"][0]["plate"], "AZ")
        self.assertEqual(decoded["metadata"]["search_mode"], "known_plates")

    def test_pattern_is_explicit_and_does_not_alter_visual_score(self):
        """Format rules restrict candidates without pretending to be image evidence."""
        # L,D disallow the visually stronger digit-first sequence.
        logits = [position_logits(A=2, **{"1": 5}), position_logits(B=6, **{"2": 3})]
        constrained = decode_logits(logits, pattern="LD", top_k=1)
        known = decode_logits(logits, known_plates=["A2"], top_k=1)
        self.assertEqual(constrained["candidates"][0]["plate"], "A2")
        self.assertEqual(constrained["candidates"][0]["log_score"], known["candidates"][0]["log_score"])

    def test_extreme_logits_and_temperature_stay_numerically_stable(self):
        """Softmax must subtract its maximum before exponentiating."""
        # A large positive offset must not overflow exponentiation.
        decoded = decode_logits([[10000.0] + [-10000.0] * 36], temperature=0.5, known_plates=["A"])
        self.assertTrue(math.isfinite(decoded["candidates"][0]["log_score"]))
        self.assertEqual(decoded["positions"][0]["alternatives"][0]["probability"], 1.0)

    def test_uniform_or_forced_weak_evidence_is_unreadable(self):
        """A shortlist must not turn an uninformative image into a confident read."""
        self.assertEqual(decode_logits([[0.0] * 37])["status"], "unreadable")
        self.assertEqual(
            decode_logits([position_logits(A=8, B=0)], known_plates=["B"], min_position_confidence=0.1)[
                "status"
            ],
            "unreadable",
        )

    def test_no_matching_known_plate_is_unreadable(self):
        """An empty constrained search reports failure rather than inventing a plate."""
        decoded = decode_logits([position_logits(A=4)], known_plates=["12"])
        self.assertEqual(decoded["candidates"], [])
        self.assertEqual(decoded["status"], "unreadable")

    def test_invalid_inputs_raise_clear_errors(self):
        """Reject malformed tensors and resource limits before searching."""
        # Each case violates a public precondition rather than implementation detail.
        invalid_calls = [
            ([], {}),
            ([[0.0] * 36], {}),
            ([[math.nan] * 37], {}),
            ([[math.inf] * 37], {}),
            ([[0.0] * 37], {"temperature": 0}),
            ([[0.0] * 37], {"top_k": 0}),
            ([[0.0] * 37], {"top_k": 1001}),
            ([[0.0] * 37], {"min_length": 2}),
            ([[0.0] * 37], {"pattern": "X"}),
            ([[0.0] * 37], {"pattern": "LL"}),
            ([[0.0] * 37], {"known_plates": []}),
            ([[0.0] * 37], {"min_position_confidence": 2}),
        ]
        for logits, options in invalid_calls:
            # Subtests retain the failed input if validation accidentally regresses.
            with self.subTest(options=options, rows=len(logits)):
                with self.assertRaises(ValueError):
                    decode_logits(logits, **options)


if __name__ == "__main__":
    # unittest keeps this module runnable before optional ML dependencies exist.
    unittest.main()
