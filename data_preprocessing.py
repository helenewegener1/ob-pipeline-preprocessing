#!/usr/bin/env python

"""
CLI utility to convert gzipped FCS data into gzipped CSV outputs with optional column relabeling.

Args:
    --data.raw      Path to a gz-compressed FCS file OR a directory of FCS files.
    --data.labels   Path to a gz-compressed labels file. Text replaces FCS headers; XML is not supported.
    --output_dir    Directory where the matrix/label CSV files will be written.
    --name          Dataset name used for the output filenames.
    --seed          Random seed used for deterministic train/test splits.
    --method        Train/test split method (only 'default' is supported today).
"""

import argparse
import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import fcsparser
import numpy as np
import pandas as pd


def read_bytes_handling_gzip(path: str) -> bytes:
    """
    Return file contents, transparently handling gzip-compressed files.

    Some inputs may have a .gz suffix even when they are plain text; fall back to
    normal reads if gzip decompression fails.
    """
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read()
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as fh:
            return fh.read()


def parse_fcs_to_dataframe(raw_gz_path: str):
    data_bytes = read_bytes_handling_gzip(raw_gz_path)

    # fcsparser.parse expects a file path; use a temporary file to avoid keeping data on disk.
    with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
        tmp.write(data_bytes)
        tmp_path = tmp.name

    try:
        _, data = fcsparser.parse(tmp_path, reformat_meta=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # If cleanup fails, we still want to return the parsed data/error.

    return data


def parse_label_lines(label_text: str, expected_count: int, source: str) -> List[str]:
    labels = [line.strip() for line in label_text.splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {source}.")

    if len(labels) != expected_count:
        raise ValueError(
            f"Label count ({len(labels)}) does not match number of columns ({expected_count})."
        )
    return labels


def detect_label_format(label_path: str, label_text: str) -> str:
    """Return 'txt' or 'xml' based on path suffix or content."""
    suffixes = [s.lower() for s in Path(label_path).suffixes if s.lower() != ".gz"]
    if ".xml" in suffixes:
        return "xml"
    if ".txt" in suffixes:
        return "txt"

    stripped = label_text.lstrip()
    if stripped.startswith("<"):
        return "xml"

    return "txt"


def is_flowjo_workspace(path: str) -> bool:
    suffixes = {s.lower() for s in Path(path).suffixes}
    return ".wps" in suffixes or ".wsp" in suffixes


def apply_labels(label_gz_path: str, df):
    """Apply labels to DataFrame columns according to the provided rules."""
    try:
        label_text = read_bytes_handling_gzip(label_gz_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Unexpected label file format: unable to decode as UTF-8 text."
        ) from exc

    if not label_text.strip():
        raise ValueError(
            "Unexpected label file format: file is empty after decompression."
        )

    label_format = detect_label_format(label_gz_path, label_text)

    if label_format == "xml":
        raise NotImplementedError("XML label handling not implemented.")
    if label_format != "txt":
        raise ValueError("Unexpected label file format.")

    try:
        labels = parse_label_lines(
            label_text, expected_count=df.shape[1], source=label_gz_path
        )
    except ValueError as exc:
        print(
            f"Warning: {exc} Column relabeling skipped; keeping original headers.",
            file=sys.stderr,
        )
        return df
    df.columns = labels
    return df


def collect_fcs_inputs(raw_input: str) -> List[Path]:
    """
    Accept a single FCS path or a directory of FCS files (optionally gzipped) and return a sorted list of paths.
    """
    path = Path(raw_input)
    if path.is_dir():
        candidates = sorted(
            [
                p
                for p in path.iterdir()
                if any(suffix.lower() == ".fcs" for suffix in p.suffixes)
            ]
            + [
                p
                for p in path.iterdir()
                if tuple(s.lower() for s in p.suffixes[-2:]) == (".fcs", ".gz")
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No FCS files found in directory: {raw_input}")
        return candidates
    if not path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_input}")
    return [path]


def prepared_fcs_paths(fcs_paths: Sequence[Path]) -> Iterable[List[Path]]:
    """
    Ensure every FCS is an on-disk uncompressed file so FlowJo parsers can load them.
    Returns a list of usable paths and cleans up any temporary files afterwards.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    prepared: List[Path] = []
    try:
        for fcs_path in fcs_paths:
            suffixes = [s.lower() for s in fcs_path.suffixes]
            if suffixes and suffixes[-1] == ".gz":
                target_name = fcs_path.name
                if target_name.lower().endswith(".gz"):
                    target_name = target_name[: -len(".gz")]
                target_path = Path(tmp_dir.name) / target_name
                with gzip.open(fcs_path, "rb") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                prepared.append(target_path)
            else:
                prepared.append(fcs_path)
        yield prepared
    finally:
        tmp_dir.cleanup()


def _flowjo_leaf_labels(
    gating_result, leaves: Sequence[str], event_count: int
) -> pd.Series:
    """
    Convert FlowJo gating results into a label Series by assigning the leaf gate name to each event.
    Unassigned events are labeled 'unlabeled'.
    """
    labels = np.full(event_count, "unlabeled", dtype=object)
    for gate_name in leaves:
        if hasattr(gating_result, "get_gate_membership"):
            mask = gating_result.get_gate_membership(gate_name)
        elif hasattr(gating_result, "get_population_mask"):
            mask = gating_result.get_population_mask(gate_name)
        else:
            raise RuntimeError(
                "FlowJo gating result does not expose gate membership accessors."
            )
        labels[np.asarray(mask, dtype=bool)] = gate_name
    return pd.Series(labels, name="label")


def label_samples_from_flowjo_workspace(
    workspace_path: str, fcs_paths: Sequence[Path]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Use a FlowJo workspace (.wsp/.wps) to gate a collection of FCS files and emit per-event labels.

    The FlowJo file typically references original sample paths; we match samples by basename. A missing
    FlowKit dependency raises a clear error.
    """
    try:
        import flowkit as fk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FlowJo workspace inputs require the 'flowkit' package. "
            "Install it (e.g., pip install flowkit) and re-run this step."
        ) from exc

    workspace = fk.FlowJoWorkspace(workspace_path)
    feature_frames: List[pd.DataFrame] = []
    label_frames: List[pd.Series] = []

    # Build a lookup of sample identifiers exposed by the workspace so we can find the matching gating strategy.
    workspace_samples = {}
    if hasattr(workspace, "sample_ids"):
        workspace_samples = {sid: sid for sid in workspace.sample_ids}
    elif hasattr(workspace, "samples") and isinstance(workspace.samples, dict):
        workspace_samples = {sid: sid for sid in workspace.samples.keys()}

    for fcs_path in fcs_paths:
        sample_key = fcs_path.stem
        matching_id: Optional[str] = None
        for candidate in workspace_samples:
            if (
                Path(candidate).stem == sample_key
                or Path(candidate).name == fcs_path.name
            ):
                matching_id = candidate
                break
        # Fall back to using the stem directly if FlowKit handles name resolution internally.
        if matching_id is None:
            matching_id = sample_key

        gating_strategy = workspace.get_gating_strategy(matching_id)

        sample = fk.Sample(str(fcs_path), sample_id=matching_id)
        # FlowKit exposes either gate_sample function or method on gating strategy depending on version.
        if hasattr(fk, "gate_sample"):
            gating_result = fk.gate_sample(sample, gating_strategy)
        elif hasattr(gating_strategy, "gate_sample"):
            gating_result = gating_strategy.gate_sample(sample)
        else:
            raise RuntimeError(
                "Unable to gate sample with FlowKit; expected gate_sample helper or method."
            )

        leaf_gate_names: Sequence[str]
        if hasattr(gating_strategy, "get_population_ids"):
            leaf_gate_names = gating_strategy.get_population_ids(leaves_only=True)
        elif hasattr(gating_strategy, "population_ids"):
            leaf_gate_names = list(gating_strategy.population_ids)
        else:
            raise RuntimeError(
                "FlowKit gating strategy missing population identifiers."
            )

        if hasattr(sample, "as_dataframe"):
            sample_df = sample.as_dataframe()
        elif hasattr(sample, "data"):
            sample_df = pd.DataFrame(sample.data)
        else:
            raise RuntimeError("FlowKit sample object does not expose data accessors.")

        label_series = _flowjo_leaf_labels(
            gating_result=gating_result,
            leaves=leaf_gate_names,
            event_count=len(sample_df),
        )

        feature_frames.append(sample_df)
        label_frames.append(label_series)

    features_df = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    return features_df, labels


def split_features_and_labels(df) -> Tuple:
    """
    Split the loaded dataframe into features and labels if a label column exists.

    The column named 'label' (case-insensitive) is treated as the target vector.
    Returns (features_df, labels_series_or_None).
    """
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        print(
            "Warning: no label column found; writing all data as features.",
            file=sys.stderr,
        )
        return df, None

    labels = df[label_col]
    features = df.drop(columns=[label_col])
    return features, labels


def split_train_test(
    features_df: pd.DataFrame,
    labels: Optional[pd.Series],
    method: str,
    seed: int,
    test_fraction: float = 0.2,
) -> Tuple[
    Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[pd.DataFrame, Optional[pd.Series]]
]:
    """
    Split features (and labels, when available) into train/test partitions.
    """
    if method != "default":
        raise ValueError(
            f"Unsupported split method '{method}'. Only 'default' is implemented."
        )

    if features_df.empty:
        raise ValueError("No data rows found; cannot perform train/test split.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(features_df))
    rng.shuffle(indices)

    test_size = max(1, int(len(indices) * test_fraction))
    if test_size >= len(indices):
        test_size = 1
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if train_idx.size == 0:
        # Ensure a non-empty train split.
        train_idx, test_idx = indices[:-1], indices[-1:]

    train_features = features_df.iloc[train_idx]
    test_features = features_df.iloc[test_idx]

    if labels is None:
        return (train_features, None), (test_features, None)

    train_labels = labels.iloc[train_idx]
    test_labels = labels.iloc[test_idx]
    return (train_features, train_labels), (test_features, test_labels)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess gzipped FCS data into CSV."
    )
    parser.add_argument(
        "--data.raw",
        type=str,
        required=True,
        help="Gz-compressed FCS data file.",
    )
    parser.add_argument(
        "--data.labels",
        type=str,
        required=True,
        help="Gz-compressed labels file. Text replaces FCS headers; XML is not supported.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the resulting CSV file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Dataset name used for output filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic train/test splits.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="default",
        help="Train/test split method. Only 'default' is supported.",
    )
    return parser


def main(argv: Iterable[str] = None):
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = getattr(args, "data.raw")
    label_path = getattr(args, "data.labels")
    output_dir = args.output_dir
    name = args.name
    seed = args.seed
    method = args.method

    if is_flowjo_workspace(label_path):
        fcs_files = collect_fcs_inputs(raw_path)
        with prepared_fcs_paths(fcs_files) as ready_fcs:
            features_df, labels = label_samples_from_flowjo_workspace(
                label_path, ready_fcs
            )
    else:
        data_df = parse_fcs_to_dataframe(raw_path)
        data_df = apply_labels(label_path, data_df)
        features_df, labels = split_features_and_labels(data_df)

    os.makedirs(output_dir, exist_ok=True)
    (train_features, train_labels), (test_features, test_labels) = split_train_test(
        features_df, labels, method=method, seed=seed
    )

    # Test split keeps the legacy filenames for downstream compatibility.
    test_data_output_path = os.path.join(output_dir, f"{name}.matrix.gz")
    test_features.to_csv(test_data_output_path, index=False, compression="gzip")
    if test_labels is not None:
        test_label_output_path = os.path.join(output_dir, f"{name}.true_labels.gz")
        test_labels.to_csv(
            test_label_output_path, index=False, header=False, compression="gzip"
        )

    # Training split uses the new suffixes.
    train_data_output_path = os.path.join(output_dir, f"{name}.matrix.training.gz")
    train_features.to_csv(train_data_output_path, index=False, compression="gzip")
    if train_labels is not None:
        train_label_output_path = os.path.join(
            output_dir, f"{name}.true_labels.training.gz"
        )
        train_labels.to_csv(
            train_label_output_path, index=False, header=False, compression="gzip"
        )


if __name__ == "__main__":
    main()
