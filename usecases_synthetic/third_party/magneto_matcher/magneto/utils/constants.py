import numpy as np
import pandas as pd

NULL_REPRESENTATIONS = {
    "not allowed to collect",
    "not reported",
    "unknown",
    "not otherwise specified",
    "nos",
    "not applicable",
    "na",
    "not available",
    "n/a",
    "none",
    "null",
    "",
    " ",
    "missing",
    "unspecified",
    "undetermined",
    "not collected",
    "not recorded",
    "not provided",
    "no data",
    "unavailable",
    "empty",
    "undefined",
    "not defined",
    "other, specify",
    "other",
    "exposure to secondhand smoke history not available",
    "exposure to secondhand smoke history not available.",
    "indeterminate",
    "staging incomplete",
    "no pathologic evidence of distant metastasis",
    "medical record does not state",
    "patient not interviewed",
    "medical record does not state.",
    "patient not interviewed.",
    None,
    np.nan,
    pd.NaT,
    pd.NA,
    pd.NaT,
}

KEY_REPRESENTATIONS = [
    "id",
    "identifier",
    "key",
    "uuid",  # Universally Unique Identifier
    "gid",  # Global ID
    "sid",  # System ID
    "token",  # Often used in authentication contexts
    "serial",  # Can represent a serial number or code
    "code",  # General code used for identification
    "hash",  # Hash-based unique identifier
    "primary_key",  # Common in databases
    "foreign_key",  # Reference to a primary key in another table
    "access_key",  # Common in APIs
    "unique_id",  # Explicitly stating uniqueness
    "slug",  # URL-friendly identifier often used in web contexts
    "auth_token",  # Used for authentication
    "apikey",  # API key (alternative spelling)
    "object_id",  # Frequently used in object-oriented databases
    "record_id",  # General identifier for records in a dataset
]

BINARY_VALUES = {
    "yes",
    "no",
    "true",
    "false",
    "t",
    "f",
    "y",
    "n",
    "1",
    "0",
    "1.0",
    "0.0",
    "1.00",
    "0.00",
    "0.",
    "1.",
    "present",
    "absent",
    "positive",
    "negative",
    "detected",
    "not detected",
    "normal",
    "abnormal",
    "enabled",
    "disabled",
    "active",
    "inactive",
    "open",
    "closed",
    "success",
    "failure",
    "on",
    "off",
    "approved",
    "rejected",
    "included",
    "excluded",
    "passed",
    "failed",
    "accepted",
    "denied",
    "smoker",
    "non-smoker",
    "present",
    "not identified",
    "no or minimal exposure to secondhand smoke",
    "no or minimal exposure to secondhand smoke.",
}
