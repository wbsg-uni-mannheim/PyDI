import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_integer_dtype

from .constants import (
    BINARY_VALUES,
    KEY_REPRESENTATIONS,
    NULL_REPRESENTATIONS,
)

lm_map = {
    "roberta": "roberta-base",
    "mpnet": "microsoft/mpnet-base",
    "distilbert": "distilbert-base-uncased",
    "arctic": "Snowflake/snowflake-arctic-embed-m-v1.5",
}

sentence_transformer_map = {
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def get_dataset_paths(dataset):
    dataset_map = {
        "gdc": "GDC",
        "chembl": "ChEMBL",
        "opendata": "OpenData",
        "tpcdi": "TPC-DI",
        "wikidata": "Wikidata",
    }

    task_map = {
        "joinable": "Joinable",
        "semjoinable": "Semantically-Joinable",
        "unionable": "Unionable",
        "viewunion": "View-Unionable",
    }

    if "-" in dataset:
        task = dataset.split("-")[1]
        dataset = dataset.split("-")[0]
        data_dir = f"datasets/{dataset_map[dataset]}/{task_map[task]}"
    else:
        data_dir = f"datasets/{dataset_map[dataset]}"

    return (
        f"{data_dir}/source-tables",
        f"{data_dir}/target-tables",
        f"{data_dir}/matches.csv",
    )


def to_lowercase(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()
    return df


def process_tables(source_table, target_table):
    # print(source_table.columns)
    # processed_source_table = to_lowercase(source_table)
    # processed_target_table = to_lowercase(target_table)
    # processed_source_table.columns = processed_source_table.columns.str.strip().str.lower()
    # processed_target_table.columns = processed_target_table.columns.str.strip().str.lower()
    # return processed_source_table, processed_target_table
    processed_source_table = clean_df(source_table)
    processed_target_table = clean_df(target_table)
    return processed_source_table, processed_target_table


def get_samples(values, n=15, random=False):
    unique_values = values.dropna().unique()
    if random:
        tokens = np.random.choice(
            unique_values, min(15, len(unique_values)), replace=False
        )
    else:
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n)
        tokens = most_frequent_values.index.tolist()
    return [str(token) for token in tokens]


def get_samples_2(values, n=15, random=False):
    unique_values = values.dropna().unique()
    if random:
        tokens = np.random.choice(
            unique_values, min(15, len(unique_values)), replace=False
        )
    else:
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n)
        tokens = most_frequent_values.index.tolist()
    return [str(token) for token in tokens]


def infer_column_dtype(column, datetime_threshold=0.9):
    if column.isnull().all():
        return "unknown"

    # Try converting to numeric (int or float)
    temp_col = pd.to_numeric(column, errors="coerce")
    if not temp_col.isnull().all():
        if is_integer_dtype(temp_col.dtype):
            return "integer"
        elif is_float_dtype(temp_col.dtype):
            return "float"

    # Suppress warnings from datetime conversion to avoid user confusion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            temp_col = pd.to_datetime(column, errors="coerce")
            if not temp_col.isnull().all() and (
                temp_col.notna().sum() / len(temp_col) >= datetime_threshold
            ):
                return "datetime"
        except Exception:
            pass

    # Default to categorical if other conversions fail
    return "categorical"


def is_null_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in NULL_REPRESENTATIONS


def is_binary_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in BINARY_VALUES


def detect_column_type(col, key_threshold=0.8, numeric_threshold=0.90):
    # Try converting to numeric (int or float)
    temp_col = pd.to_numeric(col, errors="coerce")
    if not temp_col.isnull().all():
        return "numerical"

    if "gene" in col.name.lower():
        # TODO, implement a less naive approach
        return "gene"

    if "date" in col.name.lower():
        # TODO, implement a less naive approach
        return "date"

    unique_values = col.dropna().unique()
    if len(unique_values) / len(col) > key_threshold and col.dtype not in [
        np.float64,
        np.float32,
        np.float16,
    ]:
        # columns with many distinct values are considered as "keys"
        return "key"

    if len(unique_values) == 0:
        return "unknown"

    col_name = col.name.lower()
    if any(
        col_name.startswith(rep) or col_name.endswith(rep)
        for rep in KEY_REPRESENTATIONS
    ):
        return "key"

    if col.dtype in [np.float64, np.int64]:
        return "numerical"

    numeric_unique_values = pd.Series(pd.to_numeric(unique_values, errors="coerce"))
    numeric_unique_values = numeric_unique_values.dropna()

    if not numeric_unique_values.empty:
        if len(numeric_unique_values) / len(unique_values) > numeric_threshold:
            if len(numeric_unique_values) > 2:
                return "numerical"
            else:
                unique_values_as_int = set(map(int, unique_values))
                if unique_values_as_int.issubset({0, 1}):
                    return "binary"
                else:
                    return "numerical"

    if len(unique_values) == 2 and all(is_binary_value(val) for val in unique_values):
        return "binary"
    else:
        return "categorical"

    raise ValueError(f"Could not detect type for column {col.name}")


def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def remove_invalid_characters(input_string):
    # Remove any character that is not a letter, digit, or whitespace
    pattern = r"[^a-zA-Z0-9\s]"
    cleaned_string = re.sub(pattern, " ", input_string)
    return cleaned_string


def split_camel_case(input_string):
    # Split camel case by adding a space before any uppercase letter that is followed by a lowercase letter
    split_string = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", input_string)
    return split_string


def clean_column_name(col_name):
    # Strip leading/trailing spaces, convert to lowercase, split camel case, and remove invalid characters
    col_name = col_name.strip()
    col_name = split_camel_case(col_name)
    col_name = col_name.lower()
    col_name = remove_invalid_characters(col_name)
    # Reduce multiple spaces to a single space
    col_name = re.sub(r"\s+", " ", col_name)
    return col_name


def clean_element(x):
    if is_null_value(x):
        return None
    if isinstance(x, str):
        val = split_camel_case(x)
        val = remove_invalid_characters(val.strip().lower())

        if val != "":
            return val
        else:
            return None
    return x


def clean_df(df):
    df = df.apply(lambda col: col.apply(clean_element))
    return df
