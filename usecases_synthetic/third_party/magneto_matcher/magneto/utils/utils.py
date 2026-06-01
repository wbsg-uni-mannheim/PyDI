import re

import mmh3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from valentine import MatcherResults
from dateutil.parser import parse

from .constants import (
    BINARY_VALUES,
    KEY_REPRESENTATIONS,
    NULL_REPRESENTATIONS,
)

PHI_FRACTION = 0.6180339887  # φ - 1
np.random.seed(42)


def convert_to_valentine_format(matched_columns, source_table, target_table):
    valentine_format = {}
    for source_column, matches in matched_columns.items():
        for target_column, score in matches:
            key = (source_table, source_column), (target_table, target_column)
            valentine_format[key] = score
    if isinstance(valentine_format, MatcherResults):
        return valentine_format
    return MatcherResults(valentine_format)


def common_prefix(strings):
    if not strings:
        return ""

    # Sort the list, the common prefix of the whole list would be the common prefix of the first and last string
    strings.sort()
    first = strings[0]
    last = strings[-1]

    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1

    return first[:i]


def common_ngrams(strings, threshold=0.3):
    most_common_ngrams = {}

    for n in range(3, 9):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(n, n))

        tfidf_matrix = vectorizer.fit_transform(strings)

        scores = tfidf_matrix.sum(axis=0)

        ngram_scores = [
            (ngram, scores[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()
        ]

        filtered_ngrams = [ngram for ngram in ngram_scores if ngram[1] > threshold]

        most_common_ngrams[n] = sorted(
            filtered_ngrams, key=lambda x: x[1], reverse=True
        )

    return most_common_ngrams


def preprocess_string(s):
    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


def is_null_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in NULL_REPRESENTATIONS


def is_binary_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in BINARY_VALUES


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


def get_type2columns_map(df):
    # TODO: add more types, maybe semantic types
    types2columns_map = {}
    types2columns_map["key"] = []
    types2columns_map["numerical"] = []
    types2columns_map["categorical"] = []
    types2columns_map["binary"] = []
    types2columns_map["gene"] = []
    types2columns_map["date"] = []
    types2columns_map["Unknown"] = []

    for col in df.columns:
        col_type = detect_column_type(df[col])
        types2columns_map[col_type].append(col)

    return types2columns_map


def fibonacci_hash(x):
    result = (x * PHI_FRACTION) % 1  # Take fractional part
    return result


def get_samples(values, n=15, mode="priority_sampling"):
    """
    Sample values from a pandas Series using different strategies.

    Args:
        values: pandas Series containing the values to sample
        n: number of samples to return (default: 15)
        mode: sampling strategy ('random', 'frequent', or 'mixed')
            - 'random': completely random sampling from unique values
            - 'frequent': only the most frequent values
            - 'mixed': combination of frequent and diverse values
            - 'weighted': weighted sampling based on value counts
            - 'priority_sampling': uses priority sampling based on frequency and hash of the values
            - 'consistent_sampling': consistent uniform sampling based on hash of the values

    Returns:
        List of string representations of sampled values
    """
    unique_values = values.dropna().unique()
    total_unique = len(unique_values)

    # If total unique values are fewer than n, return them all
    if total_unique <= n:
        return sorted([str(val) for val in unique_values])

    if mode == "random":
        # Completely random sampling
        random_indices = np.random.choice(total_unique, size=n, replace=False)
        sampled_values = unique_values[random_indices]
        tokens = sorted(sampled_values)

    elif mode == "frequent":
        # Only most frequent values
        value_counts = values.dropna().value_counts()
        tokens = value_counts.head(n).index.tolist()
        tokens.sort()

    elif mode == "mixed":
        # Mix of most frequent and evenly spaced values
        n_frequent = n // 2
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n_frequent).index.tolist()

        # Calculate evenly spaced samples for diversity
        n_diverse = n - n_frequent
        spacing_interval = max(1, total_unique // n_diverse)
        diverse_values = unique_values[::spacing_interval][:n_diverse]

        # Combine frequent and diverse samples, remove duplicates
        # tokens = sorted(set(most_frequent_values + list(diverse_values)))
        tokens = sorted(set(map(str, most_frequent_values + list(diverse_values))))

    elif mode == "weighted":
        # Weighted sampling based on value counts
        value_counts = values.dropna().value_counts(sort=False)
        weights = value_counts / value_counts.sum()
        sampled_indices = np.random.choice(
            total_unique, size=n, replace=False, p=weights
        )
        sampled_values = unique_values[sampled_indices]
        tokens = sampled_values

    elif mode == "priority_sampling":
        value_counts = values.dropna().value_counts(sort=False)

        # Calculate priorities: qi = freq / hash(value)
        priorities = pd.Series(
            {
                val: freq / fibonacci_hash(mmh3.hash(str(val), 42))
                for val, freq in value_counts.items()
            }
        )

        # Select the top elements based on priority scores
        sampled_values = priorities.nlargest(n).index.tolist()
        tokens = sampled_values

    elif mode == "consistent_sampling":
        value_counts = values.dropna().value_counts(sort=False)

        priorities = pd.Series(
            {
                val: fibonacci_hash(mmh3.hash(str(val), 42))
                for val in value_counts.keys()
            }
        )

        # Select the top elements based on priority scores
        sampled_values = priorities.nlargest(n).index.tolist()
        tokens = sampled_values

    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Use 'random', 'frequent', 'mixed','weighted', 'priority_sampling' or 'consistent_sampling'"
        )

    return [str(token) for token in tokens]


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(str(string), fuzzy=fuzzy)
        return True
    except Exception:
        return False
