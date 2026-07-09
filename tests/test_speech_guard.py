"""
Unit tests for the deterministic speech guard (C4.6b).
The guard screens model-authored spoken text; failure falls back to the
approved template content, never rejecting the proposal itself.
"""
import unittest

from voice_intake.speech_guard import MAX_SPOKEN_CHARS, guard_spoken_text


class SpeechGuardTest(unittest.TestCase):
    def test_plain_prose_accepted(self):
        text, reason = guard_spoken_text(
            "Thanks, I have that noted. What is the reason for the visit?"
        )
        self.assertEqual(reason, "accepted")
        self.assertEqual(
            text, "Thanks, I have that noted. What is the reason for the visit?"
        )

    def test_none_is_absent(self):
        text, reason = guard_spoken_text(None)
        self.assertIsNone(text)
        self.assertEqual(reason, "absent")

    def test_whitespace_only_is_absent(self):
        text, reason = guard_spoken_text("   \n\t ")
        self.assertIsNone(text)
        self.assertEqual(reason, "absent")

    def test_internal_whitespace_normalized(self):
        text, reason = guard_spoken_text("Sure,\n  one   moment please.")
        self.assertEqual(reason, "accepted")
        self.assertEqual(text, "Sure, one moment please.")

    def test_too_long_rejected(self):
        text, reason = guard_spoken_text("word " * (MAX_SPOKEN_CHARS // 4))
        self.assertIsNone(text)
        self.assertEqual(reason, "too_long")

    def test_markup_rejected(self):
        for bad in ("hello <b>there</b>", "say {{field_label}} now", "a > b"):
            text, reason = guard_spoken_text(bad)
            self.assertIsNone(text, bad)
            self.assertEqual(reason, "markup", bad)

    def test_long_digit_run_rejected(self):
        text, reason = guard_spoken_text("Your number is 5551234567, correct?")
        self.assertIsNone(text)
        self.assertEqual(reason, "unmasked_digits")

    def test_ssn_pattern_rejected(self):
        text, reason = guard_spoken_text("I have 123-45-6789 on file.")
        self.assertIsNone(text)
        self.assertEqual(reason, "unmasked_digits")

    def test_short_digits_allowed(self):
        # Dates and short numbers are fine - masked readback uses them.
        text, reason = guard_spoken_text("I have the 14th of March, is that right?")
        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(text)

    def test_clinical_language_rejected(self):
        for bad in (
            "That sounds like a diagnosis of arthritis.",
            "I can prescribe something for that.",
            "The usual dosage is fine.",
        ):
            text, reason = guard_spoken_text(bad)
            self.assertIsNone(text, bad)
            self.assertEqual(reason, "clinical_language", bad)


if __name__ == "__main__":
    unittest.main()
