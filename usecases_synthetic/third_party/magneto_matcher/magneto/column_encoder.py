from .utils.utils import detect_column_type, get_samples

modes = [
    "header_values_default",
    "header_values_prefix",
    "header_values_repeat",
    "header_values_verbose",
    "header_only",
    "header_values_verbose_notype",
    "header_values_columnvaluepair_notype",
    "header_header_values_repeat_notype",
    "header_values_default_notype",
]

sampling_modes = [
    "random",
    "frequent",
    "mixed",
    "weighted",
    "priority_sampling",
    "consistent_sampling",
]


class ColumnEncoder:
    def __init__(
        self,
        tokenizer,
        encoding_mode="header_values_repeat",
        sampling_mode="mixed",
        n_samples=10,
    ):
        self._tokenizer = tokenizer
        self.cls_token = getattr(tokenizer, "cls_token", "")
        self.sep_token = getattr(tokenizer, "sep_token", "")
        self.eos_token = getattr(tokenizer, "eos_token", "")

        self._serialization_methods = {
            "header_values_default": self._serialize_header_values_default,
            "header_values_prefix": self._serialize_header_values_prefix,
            "header_values_repeat": self._serialize_header_values_repeat,
            "header_values_verbose": self._serialize_header_values_verbose,
            "header_only": self._serialize_header_only,
            "header_values_verbose_notype": self._serialize_header_values_verbose_notype,
            "header_values_columnvaluepair_notype": self._serialize_header_values_columnvaluepair_notype,
            "header_header_values_repeat_notype": self._serialize_header_values_repeat_notype,
            "header_values_default_notype": self._serialize_header_values_default,
        }

        if encoding_mode not in self._serialization_methods:
            raise ValueError(
                f"Unsupported encoding mode: {encoding_mode}. Supported modes are: {list(self._serialization_methods.keys())}"
            )
        if sampling_mode not in sampling_modes:
            raise ValueError(
                f"Unsupported sampling mode: {sampling_mode}. Supported modes are: {sampling_modes}"
            )

        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.n_samples = n_samples

    def encode(self, df, col):
        """Encodes the column of a DataFrame using the selected serialization method."""
        header = col
        tokens = get_samples(df[col], n=self.n_samples, mode=self.sampling_mode)
        data_type = detect_column_type(df[col])
        return self._serialization_methods[self.encoding_mode](
            header, data_type, tokens
        )

    def _serialize_header_values_verbose(self, header, data_type, tokens):
        """Serializes with detailed column header, type, and token values."""
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Type: {data_type}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
        )

    def _serialize_header_values_default(self, header, data_type, tokens):
        """Serializes with default format including header, type, and tokens."""
        return (
            f"{self.cls_token}"
            f"{header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_values_prefix(self, header, data_type, tokens):
        """Serializes with prefixed labels for header, datatype, and values."""
        return (
            f"{self.cls_token}"
            f"header:{header}{self.sep_token}"
            f"datatype:{data_type}{self.sep_token}"
            f"values:{', '.join(tokens)}"
        )

    def _serialize_header_values_repeat(self, header, data_type, tokens):
        """Serializes with repeated header for emphasis."""
        repeated_header = self.sep_token.join([header] * 5)
        return (
            f"{self.cls_token}"
            f"{repeated_header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_only(self, header, data_type, tokens):
        """Serializes with header only."""
        return f"{self.cls_token}" f"{header}" f"{self.eos_token}"

    def _serialize_header_values_verbose_notype(self, header, data_type, tokens):
        """Serializes with simple format including header and tokens."""
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
            f"{self.eos_token}"
        )

    def _serialize_header_values_columnvaluepair_notype(
        self, header, data_type, tokens
    ):
        tokens = [f"{header}:{token}" for token in tokens]
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
            f"{self.eos_token}"
        )

    def _serialize_header_values_repeat_notype(self, header, data_type, tokens):
        """Serializes with repeated header for emphasis."""
        repeated_header = self.sep_token.join([header] * 5)
        return (
            f"{self.cls_token}"
            f"{repeated_header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_values_default_notype(self, header, data_type, tokens):
        return (
            f"{self.cls_token}"
            f"{header}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )
