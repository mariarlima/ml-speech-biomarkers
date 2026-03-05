"""
Utilities to reproduce the 100-dimensional LIWC + lexical feature vector
used in the research notebooks (see `notebooks/checks.ipynb`).

This module is intentionally small and explicit so that another engineer
can see exactly how the final feature vector is constructed for a single
transcript + LIWC row and reuse it in experiments or deployment code.
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mlcog.ling_features import text_analysis_features


# Columns removed in the notebook before building the feature vector.
# We keep this list here (instead of hard-coding in-place) so that changes
# to the LIWC selection are tracked in version control and easy to audit.
_DROP_LIWC_COLS: Sequence[str] = [
    "Segment",
    "Emoji",
    "netspeak",
    "sexual",
    "swear",
    "conflict",
    "politic",
    "ethnicity",
    "tech",
    "relig",
    "illness",
    "wellness",
    "mental",
    "substances",
    "death",
    "fatigue",
    "reward",
]


def build_liwc_lexical_features(liwc_csv: Path | str, transcript_txt: Path | str) -> np.ndarray:
    """
    Build the 100-D feature vector for a single transcript.

    This mirrors the steps in `notebooks/checks.ipynb`:

      1. Load a LIWC CSV row for the transcript.
      2. Drop a fixed set of columns (see `_DROP_LIWC_COLS`).
      3. Drop the *last* 7 remaining columns.
      4. Compute 5 lexical features from the raw text file.
      5. Concatenate LIWC and lexical features, excluding the first column
         (filename / pid), to obtain a 100-dimensional numeric vector.

    Parameters
    ----------
    liwc_csv:
        Path to a LIWC CSV file (exactly one row for the transcript,
        including a `Filename`-like identifier in the first column).
    transcript_txt:
        Path to the corresponding raw transcript `.txt` file.

    Returns
    -------
    np.ndarray
        1D array of length 100 with LIWC + 5 lexical features, matching the
        ordering used in the training features (e.g. `ling_test.pkl`).

    Raises
    ------
    ValueError
        If the LIWC CSV has zero or more than one row, or if the resulting
        feature vector does not have length 100.
    """
    liwc_csv = Path(liwc_csv)
    transcript_txt = Path(transcript_txt)

    liwc = pd.read_csv(liwc_csv)

    if liwc.empty:
        raise ValueError(f"LIWC CSV {liwc_csv} has no rows; expected exactly one.")
    if len(liwc) != 1:
        raise ValueError(
            f"LIWC CSV {liwc_csv} has {len(liwc)} rows; "
            "this helper expects exactly one row corresponding to the transcript."
        )

    # Drop the same LIWC columns as in the notebook (ignore if missing).
    cols_to_drop = [c for c in _DROP_LIWC_COLS if c in liwc.columns]
    if cols_to_drop:
        liwc = liwc.drop(columns=cols_to_drop)

    # Drop the last 7 columns, as in:
    #   liwc = liwc.iloc[:, :-7]
    # This relies on the LIWC column ordering being the same as in training.
    if liwc.shape[1] > 7:
        liwc = liwc.iloc[:, :-7]

    # Compute the 5 lexical features from the raw text file.
    lexical = text_analysis_features(str(transcript_txt))
    if lexical.empty:
        raise ValueError(f"No lexical features extracted from {transcript_txt}.")

    # Make sure indices are aligned, then concatenate column-wise.
    merged = pd.concat(
        [liwc.reset_index(drop=True), lexical.reset_index(drop=True)],
        axis=1,
    )

    # In the notebook, the first column is the filename/pid; the vector is row[1:].
    # We mirror that logic here so that the feature ordering matches training.
    row = merged.iloc[0]
    features = np.asarray(row[1:], dtype=float)

    if features.size != 100:
        raise ValueError(
            f"Expected 100 features but got {features.size}. "
            f"Check LIWC columns and preprocessing for {liwc_csv}."
        )

    return features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 100-dimensional LIWC + lexical feature vector "
            "for a single transcript (.txt + LIWC .csv)."
        )
    )
    parser.add_argument(
        "--liwc-csv",
        required=True,
        type=str,
        help="Path to LIWC CSV file for the transcript (one-row file).",
    )
    parser.add_argument(
        "--txt",
        required=True,
        type=str,
        help="Path to the raw transcript .txt file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.

    Example
    -------
    python -m test_single_pred.liwc \\
        --liwc-csv test_single_pred/liwc-test-pid71.csv \\
        --txt test_single_pred/stt_t71.txt
    """
    args = _parse_args()
    features = build_liwc_lexical_features(args.liwc_csv, args.txt)
    # Print as a simple comma-separated list for easy inspection/redirect.
    # Use general-format floats so we don't clutter the output with trailing zeros.
    print(",".join("{:g}".format(float(x)) for x in features))


if __name__ == "__main__":
    main()

