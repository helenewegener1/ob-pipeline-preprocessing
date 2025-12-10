#!/usr/bin/env python

"""
CLI utility to convert gzipped FCS data into gzipped CSV outputs with optional column relabeling.

Args:
    --data.raw      Path to a gz-compressed FCS file.
    --data.labels   Path to a gz-compressed labels file.
    --output_dir    Directory where the matrix/label CSV files will be written.
    --name          Dataset name used for the output filenames.
"""

import pandas
import numpy as np
import argparse
import gzip
import os
import sys
import fcsparser
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

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


# import requests
# import gzip
# def read_bytes_handling_gzip(path_or_url: str) -> bytes:
#     """Reads plain or gzipped bytes from local file or URL."""
    
#     # URL case
#     if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
#         resp = requests.get(path_or_url)
#         resp.raise_for_status()
#         raw = resp.content
#     else:
#         with open(path_or_url, "rb") as f:
#             raw = f.read()

#     # Gunzip if needed
#     if path_or_url.endswith(".gz"):
#         return gzip.decompress(raw)
#     else:
#         return raw


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


# parse_fcs_to_dataframe(raw_gz_path = "/Users/srz223/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/FlowCAP_WNV.fcs")
# parse_fcs_to_dataframe(raw_gz_path = "/Users/srz223/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/FlowCAP_ND.fcs")
parse_fcs_to_dataframe(raw_gz_path = "/Users/srz223/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/Levine_13dim_notransform.fcs.gz")

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


def apply_labels(label_gz_path: str, df):
    """Apply labels to DataFrame columns according to the provided rules."""
    try:
        label_text = read_bytes_handling_gzip(label_gz_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unexpected label file format: unable to decode as UTF-8 text.") from exc

    if not label_text.strip():
        raise ValueError("Unexpected label file format: file is empty after decompression.")

    label_format = detect_label_format(label_gz_path, label_text)

    if label_format == "xml":
        raise NotImplementedError("XML label handling not implemented.")
    if label_format != "txt":
        raise ValueError("Unexpected label file format.")

    try:
        labels = parse_label_lines(label_text, expected_count=df.shape[1], source=label_gz_path)
    except ValueError as exc:
        print(
            f"Warning: {exc} Column relabeling skipped; keeping original headers.",
            file=sys.stderr,
        )
        return df
    df.columns = labels
    return df

def replace_NAs(df):
    """
    Replace NAs of label column to ""
    """
    df["label"] = df["label"].fillna(99)

    return df

def split_features_and_labels(df) -> Tuple:
    """
    Split the loaded dataframe into features and labels if a label column exists.

    The column named 'label' (case-insensitive) is treated as the target vector.
    Returns (features_df, labels_series_or_None).
    """
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        print("Warning: no label column found; writing all data as features.", file=sys.stderr)
        return df, None

    labels = df[label_col]
    features = df.drop(columns=[label_col])
    return features, labels

# def get_unique_samples(df):
#     samples_unique = df["sample"].unique()

#     return samples_unique

# def train_test_sample_split(df, samples_unique):
#     """
#     Split features and labels into train/test subsets based on sample column + eliminate sample column after this.

#     For now, we are testing on the first sample.

#     Returns:    
#         train_set, test_set
#     """

#     training_sample = samples_unique[0]

#     train_set = df[df["sample"] == training_sample]
#     test_set = df[df["sample"] != training_sample]

#     nrow_df = df.shape[0]
#     nrow_train = train_set.shape[0]
#     nrow_test = test_set.shape[0]

#     if nrow_train + nrow_test != nrow_df:
#         print(
#             "Rows in training or test set do not match the original dataset.",
#             file=sys.stderr,
#         )

#     # remove sample column
#     train_set = train_set.drop("sample", axis=1)
#     test_set = test_set.drop("sample", axis=1)

#     return train_set, test_set

# def split_features_and_labels(df) -> Tuple:
#     """
#     Split the loaded dataframe into features and labels if a label column exists.

#     The column named 'label' (case-insensitive) is treated as the target vector.
#     Returns (features_df, labels_series_or_None).
#     """

#     label_col = next((c for c in df.columns if c.lower() == "label"), None)
#     if label_col is None:
#         print("Warning: no label column found; writing all data as features.", file=sys.stderr)
#         return df, None

#     labels = df[label_col]
#     features = df.drop(columns=[label_col])
#     return features, labels

def train_test_split(df, labels, seed, test_size=0.2):
    """
    Split features and labels into train/test subsets.

    Returns:
        features_train, labels_train, features_test, labels_test
    """
    n = df.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    features_train = df.iloc[train_idx].reset_index(drop=True)
    features_test = df.iloc[test_idx].reset_index(drop=True)

    if labels is not None:
        labels_train = labels.iloc[train_idx].reset_index(drop=True)
        labels_test = labels.iloc[test_idx].reset_index(drop=True)
    else:
        labels_train = labels_test = None

    return features_train, labels_train, features_test, labels_test

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess gzipped FCS data into CSV.")
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Link to FCS file.",
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
    default=0,
    help="Random seed used for train/test split.",
    )

    return parser


def main(argv: Iterable[str] = None):

    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = "/Users/srz223/Documents/courses/Benchmarking/repos/ob-flow-datasets/data/Levine_13dim_notransform.fcs.gz"
    raw_path = getattr(args, "data.raw")
    label_path = getattr(args, "data.labels")
    output_dir = args.output_dir
    name = args.name

    data_df = parse_fcs_to_dataframe(raw_path)
    data_df = replace_NAs(data_df)
    data_df = apply_labels(label_path, data_df)
    features_df, labels = split_features_and_labels(data_df)
    features_train, labels_train, features_test, labels_test = train_test_split(features_df, labels, seed=args.seed)

    # parser = parse_args()
    # args = parser.parse_args(argv)
    
    # # raw_path = getattr(args, "raw_path")
    # # raw_path = args.raw_path
    # raw_path = getattr(args, "data.raw")
    # label_path = getattr(args, "data.labels")
    # output_dir = args.output_dir
    # name = args.name

    
    # # raw_path = "https://raw.githubusercontent.com/kaae-2/ob-flow-datasets/main/data/FlowCAP_ND.fcs.gz"
    # data_df = parse_fcs_to_dataframe(raw_path)
    # data_df = replace_NAs(data_df)
    # # samples_unique = get_unique_samples(data_df)
    # # train_set, test_set = train_test_sample_split(data_df, samples_unique)
    # features_train, labels_train, features_test, labels_test = train_test_split(data_df, labels, seed, test_size=0.2)

    # data_df = apply_labels(label_path, data_df)
    # features_train, labels_train = split_features_and_labels(train_set)
    # features_test, labels_test = split_features_and_labels(test_set)

    # output_dir="/Users/srz223/Documents/courses/Benchmarking/out"
    # name = "covid"
    os.makedirs(output_dir, exist_ok=True)

    # Training set
    features_train.to_csv(
        os.path.join(output_dir, f"{name}.train.matrix.gz"),
        index=False,
        compression="gzip",
    )
    
    if labels_train is not None:
        labels_train.to_csv(
            os.path.join(output_dir, f"{name}.train.labels.gz"),
            index=False,
            header=False,
            compression="gzip",
        )

    # Test set
    features_test.to_csv(
        os.path.join(output_dir, f"{name}.test.matrix.gz"),
        index=False,
        compression="gzip",
    )
    if labels_test is not None:
        labels_test.to_csv(
            os.path.join(output_dir, f"{name}.test.labels.gz"),
            index=False,
            header=False,
            compression="gzip",
        )


if __name__ == "__main__":
    main()
    
