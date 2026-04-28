"""
ML/config.py — Single source of truth for label constants shared across
data generation, training pipeline, and CSV upload/analysis paths.

Importing from here prevents label-name drift between the three paths.
"""

LABEL_COLUMN = "label"
FAKE_VALUE = 1
LEGIT_VALUE = 0

# All string variants that should be treated as FAKE (1)
FAKE_STRINGS = frozenset({"1", "fake", "true", "yes", "y", "bot", "spam"})
# All string variants that should be treated as LEGIT (0)
LEGIT_STRINGS = frozenset({"0", "legit", "false", "no", "n", "real", "human", "genuine"})


def normalize_label(value) -> int:
    """
    Convert any label format to standard int: 1=fake, 0=legit.

    Handles int, float, bool, and string variants (case-insensitive).
    Raises ValueError on unrecognized input.
    """
    import math
    if isinstance(value, bool):
        return FAKE_VALUE if value else LEGIT_VALUE
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            raise ValueError(f"Label value is NaN")
        if int(value) == FAKE_VALUE:
            return FAKE_VALUE
        if int(value) == LEGIT_VALUE:
            return LEGIT_VALUE
    s = str(value).strip().lower()
    if s in FAKE_STRINGS:
        return FAKE_VALUE
    if s in LEGIT_STRINGS:
        return LEGIT_VALUE
    raise ValueError(
        f"Unrecognized label value: '{value}'. "
        f"Expected one of: {sorted(FAKE_STRINGS | LEGIT_STRINGS)}"
    )
