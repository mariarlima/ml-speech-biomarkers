"""
Single-sample prediction script. Given a transcript (.txt):
  1. Runs LIWC-22 on a .txt file and builds the 100 feature vector (via test_single_pred.liwc).
  2. Predicts ADRD probability with a pre-trained RandomForestClassifier.
  3. Predicts MMSE score with a pre-trained RandomForestRegressor.
  4. Computes and optionally saves a SHAP feature importance plot.

Requires an academic LIWC-22 license.

Both models were trained in notebooks/classif.ipynb and notebooks/reg.ipynb
respectively, and are loaded read-only (no retraining).

For reproducibility and different environments, LIWC-22 paths are configurable via
variables and env vars—see the "LIWC-22 paths (reproducible / per-environment)"
section below.

Quick setup for new machines:
  1) Install LIWC-22.
  2) Set LIWC22_CLI and LIWC22_LICENSE_SERVER (or pass CLI args).
  3) Run:
       python -m test_single_pred.predict_transcript_liwc --file path/to/file.txt
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import webbrowser

from mlcog.shap import ShapDisplay, risk_feature_plot
from mlcog.utils import plotting

from test_single_pred.liwc import build_liwc_lexical_features



def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Guarantee a (1, n_features) array so sklearn transformers never see a 1-D input."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


# Ordered feature names matching the 100d vector produced by build_liwc_lexical_features().
# First 95 are LIWC-22 categories; last 5 are lexical richness measures (Brunet, Honore, CTTR,
# PIDensity, Duplic). Order must match the columns used during model training.
LING_FEATURE_NAMES = [
    "WordCount",
    "Analytic",
    "Clout",
    "Authentic",
    "Tone",
    "WordsPerSentence",
    "BigWords",
    "Dic",
    "Linguistic",
    "function",
    "pronoun",
    "ppron",
    "i",
    "we",
    "you",
    "shehe",
    "they",
    "ipron",
    "det",
    "article",
    "number",
    "prep",
    "auxverb",
    "adverb",
    "conj",
    "negate",
    "verb",
    "adj",
    "quantity",
    "Drives",
    "affiliation",
    "achieve",
    "power",
    "Cognition",
    "allnone",
    "cogproc",
    "insight",
    "cause",
    "discrep",
    "tentat",
    "certitude",
    "differ",
    "memory",
    "Affect",
    "tone_pos",
    "tone_neg",
    "emotion",
    "emo_pos",
    "emo_neg",
    "emo_anx",
    "emo_anger",
    "emo_sad",
    "Social",
    "socbehav",
    "prosocial",
    "polite",
    "moral",
    "comm",
    "socrefs",
    "family",
    "friend",
    "female",
    "male",
    "Culture",
    "Lifestyle",
    "leisure",
    "home",
    "work",
    "money",
    "Physical",
    "health",
    "food",
    "need",
    "want",
    "acquire",
    "lack",
    "fulfill",
    "risk",
    "curiosity",
    "allure",
    "Perception",
    "attention",
    "motion",
    "space",
    "visual",
    "auditory",
    "feeling",
    "time",
    "focuspast",
    "focuspresent",
    "focusfuture",
    "Linguistic Disfluency",
    "assent",
    "nonflu",
    "filler",
    "Brunet",
    "Honore",
    "CTTR",
    "PIDensity",
    "Duplic",
]

# Normalize to title-case so mixed-case LIWC names (e.g. "ppron", "shehe") are consistent.
LING_FEATURE_NAMES = [word.capitalize() for word in LING_FEATURE_NAMES]

# Human-readable labels used on SHAP plots; only entries that differ from the capitalized name.
FEATURE_RENAME_MAP = {
    "Wordcount": "Word Count",
    "Linguistic": "Linguistic Variables",
    "Analytic": "Analytical Thinking",
    "Wordspersentence": "Words/Sentence",
    "Pronoun": "Total Pronouns",
    "Ipron": "Impersonal Pronouns",
    "Family": "Family Referents",
    "Fulfill": "Fulfill Words",
    "Assent": "Assent",
    "Affiliation": "Social Affiliation",
    "Article": "Articles",
    "Cttr": "Type-Token Ratio",
    "Honore": "Honore Index",
    "Auxverb": "Auxiliary Verbs",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = PROJECT_ROOT / "data"  # mirrors notebook paths

# -----------------------------------------------------------------------------
# LIWC-22 paths (reproducible / per-environment)
# -----------------------------------------------------------------------------
# Set these for your machine so others can run the script via env vars or CLI
# arguments, without code edits. Requires an academic LIWC-22 license.
#
# Override without editing the file:
#   export LIWC22_LICENSE_SERVER="/path/to/LIWC-22-license-server"
#   export LIWC22_CLI="/path/to/LIWC-22-cli"
# Or pass:
#   --liwc-cli /path/to/LIWC-22-cli
#   --liwc-license-server /path/to/LIWC-22-license-server
#
# If LIWC22_LICENSE_SERVER is not provided, the script tries common macOS locations.
# Windows/Linux users should set LIWC22_LICENSE_SERVER explicitly.
# -----------------------------------------------------------------------------

# CLI executable name or path. Override via env LIWC22_CLI if not on PATH.
LIWC22_CLI = os.environ.get("LIWC22_CLI", "LIWC-22-cli")

# License server (must be running for the CLI to work). Prefer explicit env/arg.
LIWC22_LICENSE_SERVER = os.environ.get("LIWC22_LICENSE_SERVER")

# Seconds to wait after starting the license server before calling the CLI.
LIWC22_SERVER_STARTUP_WAIT = float(os.environ.get("LIWC22_SERVER_STARTUP_WAIT", "3"))


def _resolve_liwc22_license_server(override: Optional[str] = None) -> Path:
    """Resolve license server path from arg/env/common macOS install locations."""
    if override:
        return Path(override).resolve()
    if LIWC22_LICENSE_SERVER:
        return Path(LIWC22_LICENSE_SERVER).resolve()
    for candidate in [
        Path("/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-license-server"),
        Path.home() / "Applications" / "LIWC-22.app" / "Contents" / "MacOS" / "LIWC-22-license-server",
    ]:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate LIWC-22 license server. Set LIWC22_LICENSE_SERVER or pass "
        "--liwc-license-server /path/to/LIWC-22-license-server."
    )


def _start_liwc22_license_server(server_path: Optional[str] = None, startup_wait_s: float = LIWC22_SERVER_STARTUP_WAIT) -> None:
    """Start the LIWC-22 license server in the background so the CLI can connect."""
    resolved_server = _resolve_liwc22_license_server(server_path)
    if not resolved_server.exists():
        raise FileNotFoundError(
            f"LIWC-22 license server not found at {resolved_server}. "
            "Set LIWC22_LICENSE_SERVER or pass --liwc-license-server to the script."
        )
    subprocess.Popen(
        [str(resolved_server)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(max(0.0, startup_wait_s))


def _find_liwc22_cli(cli: str) -> str:
    """Resolve LIWC-22 CLI to an absolute path. Raises FileNotFoundError if not found."""
    # Explicit path (from --liwc-cli or env)
    if os.path.sep in cli or (os.path.altsep and os.path.altsep in cli):
        path = Path(cli).resolve()
        if path.exists():
            return str(path)
        raise FileNotFoundError(
            f"LIWC-22 CLI not found at {path}. "
            "Set --liwc-cli to the full path of the LIWC-22-cli executable."
        )
    # Env override (full path or name)
    env_cli = os.environ.get("LIWC22_CLI")
    if env_cli:
        return _find_liwc22_cli(env_cli)
    # PATH
    found = shutil.which(cli)
    if found:
        return found
    # Common macOS install locations (.app bundle and standalone)
    for candidate in [
        Path("/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-cli"),
        Path("/Applications/LIWC-22") / cli,
        Path.home() / "Applications" / "LIWC-22" / cli,
    ]:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"LIWC-22 CLI '{cli}' not found on PATH or in /Applications/LIWC-22. "
        "Either add the CLI to your PATH, set environment variable LIWC22_CLI to its full path, "
        "or pass --liwc-cli /path/to/LIWC-22-cli"
    )


def run_liwc22_cli(
    txt_path: Path,
    output_csv: Optional[Path] = None,
    cli: str = LIWC22_CLI,
    start_license_server: bool = True,
    license_server_path: Optional[str] = None,
    server_startup_wait_s: float = LIWC22_SERVER_STARTUP_WAIT,
) -> Path:
    """
    Run LIWC-22 CLI on a single .txt file and return the path to the one-row CSV.

    By default starts the LIWC-22 license server in the background so the CLI
    can connect. Use start_license_server=False if it is already running.

    Parameters
    ----------
    txt_path
        Path to the transcript .txt file.
    output_csv
        Where to save the LIWC output CSV. Default: same directory as txt,
        name ``<stem>_liwc22.csv``.
    cli
        Name or path of the LIWC-22 CLI executable (default ``LIWC-22-cli``).
    start_license_server
        If True, start LIWC-22-license-server in the background first (default True).
    license_server_path
        Optional explicit path to LIWC-22-license-server.
    server_startup_wait_s
        Wait time after starting server before calling CLI.

    Returns
    -------
    Path
        Path to the saved one-row LIWC CSV.

    Raises
    ------
    FileNotFoundError
        If the .txt file does not exist.
    RuntimeError
        If the CLI is not found or returns a non-zero exit code.
    """
    txt_path = Path(txt_path).resolve()
    if not txt_path.exists():
        raise FileNotFoundError(f"Transcript not found: {txt_path}")

    if start_license_server:
        _start_liwc22_license_server(
            server_path=license_server_path,
            startup_wait_s=server_startup_wait_s,
        )

    cli_path = _find_liwc22_cli(cli)

    if output_csv is None:
        output_csv = txt_path.with_name(f"{txt_path.stem}_liwc22.csv")
    output_csv = Path(output_csv).resolve()

    tmpdir = Path(tempfile.mkdtemp(prefix="liwc22_"))
    try:
        # Copy the single .txt into temp dir so CLI sees only this file.
        dest_txt = tmpdir / txt_path.name
        shutil.copy2(txt_path, dest_txt)

        out_in_tmp = tmpdir / "liwc_out.csv"
        result = subprocess.run(
            [cli_path, "--mode", "wc", "--input", str(tmpdir), "--output", str(out_in_tmp)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LIWC-22 CLI failed (exit {result.returncode}). "
                "Ensure the LIWC-22 desktop app or license-server is running.\n"
                f"stderr: {result.stderr or result.stdout}"
            )

        if not out_in_tmp.exists():
            raise RuntimeError(
                f"LIWC-22 CLI did not produce output at {out_in_tmp}. stderr: {result.stderr}"
            )

        df = pd.read_csv(out_in_tmp)
        # If multiple rows (e.g. multiple .txt in folder), keep only our file.
        if "Filename" in df.columns and len(df) > 1:
            df = df.loc[df["Filename"] == txt_path.name]
        if len(df) != 1:
            raise RuntimeError(
                f"Expected exactly one LIWC row for {txt_path.name}, got {len(df)}."
            )
        df.to_csv(output_csv, index=False)
        return output_csv
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@lru_cache(maxsize=1)
def _load_classification_artifacts() -> Tuple[Any, StandardScaler, np.ndarray]:
    """
    Load the tuned RandomForestClassifier for ADRD probability on the
    100-D LIWC+lexical features (no retraining), and fit only the scaler.
    """
    # Pre-trained RF classifier saved from `notebooks/classif.ipynb`
    model_path = DATA_DIR / "cv_eval/cv_ling/10fcv_rf.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained RF classifier not found at {model_path}")
    rf = joblib.load(model_path)

    # Rebuild scaler on original training features (no refitting of RF)
    ling_train = pd.read_pickle(DATA_DIR / "features/ling_train.pkl")
    X_train = np.stack(ling_train["data"].values)

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)

    return rf, scaler, X_scaled_train


@lru_cache(maxsize=1)
def _load_regression_artifacts() -> Tuple[Any, StandardScaler]:
    """
    Load the tuned RandomForestRegressor for MMSE prediction on the
    100-D LIWC+lexical features (no retraining), and fit only the scaler.
    """
    # Pre-trained RF regressor saved from `notebooks/reg.ipynb`
    model_path = DATA_DIR / "cv_eval/cv_ling/10fcv_reg_rfr.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained RF regressor not found at {model_path}")
    rfr = joblib.load(model_path)

    # Rebuild scaler on same features used for the regression training
    ling_train = pd.read_pickle(DATA_DIR / "features/ling_train.pkl")
    df_all = pd.read_csv(DATA_DIR / "dx-mmse.csv")

    # pid is the last 3 digits of the filename (e.g. "adrso024" → "024").
    df_all["pid"] = df_all["adressfname"].str.extract(r"(\d{3})$")
    selected_columns = ["age", "gender", "mmse", "dx", "adressfname", "test", "pid"]
    df_all = df_all[selected_columns]
    # Keep only training-set participants (test == False) to avoid data leakage in the scaler.
    df_train = df_all[df_all["test"] == False]

    df_reg = ling_train.merge(df_train, on="pid", how="left")
    df_reg["mmse"] = df_reg["mmse"].round().astype(int)

    X = np.stack(df_reg["data"].values)

    # Fit scaler on training features only; regressor weights are loaded as-is.
    scaler = StandardScaler()
    scaler.fit(X)

    return rfr, scaler


class SingleSamplePredictor:
    """
    Utility for running ADRD probability, MMSE regression, and SHAP on a single
    100-D LIWC+lexical feature vector (as constructed in `test_single_pred.liwc`).
    """

    def __init__(self) -> None:
        self.classifier, self._clf_scaler, self._clf_background = _load_classification_artifacts()
        self.regressor, self._reg_scaler = _load_regression_artifacts()

    def _predict_proba(self, x: np.ndarray) -> float:
        """
        Return probability for positive class for a single row.
        """
        x = _ensure_2d(x)
        x = self._clf_scaler.transform(x)
        # Prefer predict_proba (RF); fall back to sigmoid of decision_function (SVM-like),
        # then to a hard prediction for any other estimator type.
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(x)
            if proba.shape[1] == 1:  # single-class edge case
                return float(proba[0, 0])
            return float(proba[0, 1])  # positive-class probability
        if hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(x)
            return float(1.0 / (1.0 + np.exp(-scores[0])))  # sigmoid
        pred = self.classifier.predict(x)
        return float(pred[0])

    def _predict_mmse(self, x: np.ndarray) -> float:
        """
        Predict MMSE score for a single row.
        """
        x = _ensure_2d(x)
        x = self._reg_scaler.transform(x)
        if hasattr(self.regressor, "predict"):
            mmse = float(self.regressor.predict(x)[0])
            return float(np.clip(mmse, 0.0, 30.0))  # MMSE is defined on [0, 30]
        raise AttributeError("Regressor does not implement predict().")

    def _compute_shap_for_sample(
        self,
        x_scaled: np.ndarray,
        background: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values for the classifier on a single (already scaled) sample.
        """
        x_scaled = _ensure_2d(x_scaled)
        # Without an external background, duplicate the sample — a minimal but valid reference.
        if background is None:
            background = np.repeat(x_scaled, repeats=10, axis=0)
        else:
            background = _ensure_2d(background)

        def predict_fn(z: np.ndarray) -> np.ndarray:
            z = _ensure_2d(z)
            # z is already in the scaled space used to train the classifier.
            if hasattr(self.classifier, "predict_proba"):
                return self.classifier.predict_proba(z)[:, 1]
            if hasattr(self.classifier, "decision_function"):
                scores = self.classifier.decision_function(z)
                return 1.0 / (1.0 + np.exp(-scores))
            preds = self.classifier.predict(z)
            return preds.astype(float)

        # Cap background at 100 rows for speed; KernelSHAP cost scales with |background|.
        explainer = shap.KernelExplainer(
            predict_fn,
            shap.sample(background, min(100, background.shape[0])),
        )
        shap_values = explainer.shap_values(x_scaled)

        # Older shap versions return a list (one array per class); take the positive-class slice.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values_sample = np.asarray(shap_values, dtype=float)[0]
        expected_value = float(np.asarray(explainer.expected_value).ravel()[-1])
        return shap_values_sample, expected_value

    def predict_with_explanation(
        self,
        features: np.ndarray,
        feature_names: Sequence[str],
        show_plot: bool = True,
        save_plot_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run ADRD probability, MMSE prediction, and SHAP explanation on a
        pre-computed 100-D feature vector.

        Returns a dictionary with:
            - 'proba': probability of ADRD (positive class)
            - 'mmse': predicted MMSE score
            - 'features': original (unscaled) 1D numpy array of features
            - 'shap_values': 1D numpy array of SHAP values
            - 'expected_value': SHAP expected value (baseline)
        """
        features = np.asarray(features, dtype=float).ravel()

        proba = self._predict_proba(features)
        mmse = self._predict_mmse(features)

        x_scaled_clf = self._clf_scaler.transform(_ensure_2d(features))
        shap_vals, expected_value = self._compute_shap_for_sample(
            x_scaled_clf,
            background=self._clf_background,
        )

        # Scale SHAP values to percentage points (model output is in [0, 1]).
        shap_df = pd.DataFrame(
            {
                "Feature": list(feature_names),
                "SHAP Value": shap_vals * 100.0,
                "Value": features,
            }
        )
        shap_df["Feature"] = shap_df["Feature"].replace(FEATURE_RENAME_MAP)

        ax = None
        if show_plot or save_plot_path is not None:
            shap_values_point_df = pd.DataFrame(
                {
                    "Feature": list(feature_names),
                    "Risk": shap_vals * 100.0,
                    "Value": features,
                    "Value Norm": x_scaled_clf.ravel(),
                }
            )

            shap_values_point_df["Feature"] = shap_values_point_df["Feature"].replace(
                FEATURE_RENAME_MAP
            )

            base = expected_value * 100.0

            plt.close()
            with plotting.paper_theme():
                fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
                ax = risk_feature_plot(shap_values_point_df, base, ax=ax)
                fig.subplots_adjust(left=0.375, right=0.96, top=0.85, bottom=0.2)
                ax.set_title("Feature importance SHAP analysis", fontsize=15)
                plt.tight_layout()

                if save_plot_path is not None:
                    save_path = Path(save_plot_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")

                if show_plot:
                    plt.show(block=False)
                else:
                    plt.close(fig)

        return {
            "proba": proba,
            "mmse": mmse,
            "features": features,
            "shap_values": shap_vals,
            "expected_value": expected_value,
            "axes": ax,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LIWC-22 on a transcript .txt, then ADRD probability, MMSE regression, "
            "and SHAP. Requires LIWC-22 academic license (desktop or license-server running)."
        )
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the raw transcript .txt file.",
    )
    parser.add_argument(
        "--liwc-output",
        type=str,
        default=None,
        help="Path to save the LIWC-22 CSV output (default: <stem>_liwc22.csv next to .txt).",
    )
    parser.add_argument(
        "--liwc-cli",
        type=str,
        default=LIWC22_CLI,
        help="LIWC-22 CLI executable name or full path. Default: LIWC22_CLI env or %(default)s.",
    )
    parser.add_argument(
        "--liwc-license-server",
        type=str,
        default=LIWC22_LICENSE_SERVER,
        help=(
            "Path to LIWC-22 license server executable. "
            "Default: LIWC22_LICENSE_SERVER env, else auto-detect on macOS."
        ),
    )
    parser.add_argument(
        "--liwc-startup-wait",
        type=float,
        default=LIWC22_SERVER_STARTUP_WAIT,
        help="Seconds to wait after starting LIWC-22 license server (default: %(default)s).",
    )
    parser.add_argument(
        "--save-shap",
        type=str,
        default=None,
        help="Optional path to save the SHAP plot (e.g. 'outputs/shap_single.png').",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Do not start LIWC-22 license server (use if it is already running).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="If set, do not display the SHAP plot interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    txt_path = Path(args.file)
    if not txt_path.is_absolute():
        txt_path = PROJECT_ROOT / txt_path

    if not txt_path.exists():
        raise FileNotFoundError(f"Transcript file not found at {txt_path}")

    liwc_output = None
    if args.liwc_output:
        liwc_output = Path(args.liwc_output)
        if not liwc_output.is_absolute():
            liwc_output = PROJECT_ROOT / liwc_output

    print("Running LIWC-22 CLI on transcript...")
    liwc_csv_path = run_liwc22_cli(
        txt_path,
        output_csv=liwc_output,
        cli=args.liwc_cli,
        start_license_server=not args.no_start_server,
        license_server_path=args.liwc_license_server,
        server_startup_wait_s=args.liwc_startup_wait,
    )
    print(f"LIWC output saved to {liwc_csv_path}")

    features = build_liwc_lexical_features(liwc_csv_path, txt_path)
    feature_names = LING_FEATURE_NAMES

    if len(feature_names) != len(features):
        raise ValueError(
            f"Expected {len(feature_names)} features but got {len(features)}. "
            "Check that the single-sample features align with ling_feature_names."
        )

    if args.save_shap:
        save_shap_path = Path(args.save_shap)
    else:
        # Default: save alongside the transcript file as shap_<stem>.png
        save_shap_path = txt_path.with_name(f"shap_{txt_path.stem}.png")

    predictor = SingleSamplePredictor()

    result = predictor.predict_with_explanation(
        features=features,
        feature_names=feature_names,
        show_plot=not args.no_show,
        save_plot_path=save_shap_path,
    )

    print(f"Predicted ADRD probability: {result['proba']:.3f}")
    print(f"Predicted MMSE: {result['mmse']:.2f}")

    if save_shap_path.exists():
        try:
            webbrowser.open(save_shap_path.as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()

# Example usage:
#
# Basic — LIWC-22 license server auto-started:
#
#   python -m test_single_pred.pred_transcript_liwc \
#       --file test_single_pred/stt_t71.txt
#
# If the LIWC-22 license server is already running:
#
#   python -m test_single_pred.pred_transcript_liwc \
#       --file test_single_pred/stt_t71.txt \
#       --no-start-server
#
# Suppress interactive plot and save SHAP figure explicitly:
#
#   python -m test_single_pred.pred_transcript_liwc \
#       --file test_single_pred/stt_t71.txt \
#       --no-show \
#       --save-shap test_single_pred/shap_stt_t71.png
#
# Override LIWC-22 paths without editing the file:
#
#   export LIWC22_CLI="/path/to/LIWC-22-cli"
#   export LIWC22_LICENSE_SERVER="/path/to/LIWC-22-license-server"
#   python -m test_single_pred.pred_transcript_liwc --file test_single_pred/stt_t71.txt