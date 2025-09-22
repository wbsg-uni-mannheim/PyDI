"""Built-in rules library for information extraction."""

import re
from typing import Any, Callable, Dict, List, Optional, Union

# Import normalization components
from PyDI.normalization.types import NumericParser, LinkNormalizer, parse_coordinate as _parse_coord_tuple
from PyDI.normalization.values import normalize_date
from PyDI.normalization.units import normalize_units, parse_quantity


def parse_money(text: str) -> Optional[float]:
    """Parse monetary value from text using NumericParser."""
    if not text or not isinstance(text, str):
        return None
    parser = NumericParser(handle_currency=True)
    return parser.parse_numeric(text)


def parse_number(text: str) -> Optional[float]:
    """Parse numeric value from text using NumericParser."""
    if not text or not isinstance(text, str):
        return None
    parser = NumericParser()
    return parser.parse_numeric(text)


def parse_percent(text: str) -> Optional[float]:
    """Parse percentage and return as fraction (0.0-1.0)."""
    if not text or not isinstance(text, str):
        return None
    parser = NumericParser(handle_percentages=True)
    result = parser.parse_numeric(text)
    # Convert percentage to fraction if needed
    if result is not None and result > 1.0:
        result = result / 100.0
    return result


def parse_date(text: str) -> Optional[str]:
    """Parse date using normalization module."""
    if not text or not isinstance(text, str):
        return None
    return normalize_date(text)


def normalize_url(text: str) -> Optional[str]:
    """Normalize URL using LinkNormalizer."""
    if not text or not isinstance(text, str):
        return None
    normalizer = LinkNormalizer()
    return normalizer.normalize_url(text)


def extract_domain(text: str) -> Optional[str]:
    """Extract domain from URL using LinkNormalizer."""
    if not text or not isinstance(text, str):
        return None
    normalizer = LinkNormalizer()
    return normalizer.extract_domain(text)


def parse_coordinate(text: str) -> Optional[str]:
    """Parse coordinate and format as "lat, lon" string."""
    if not text or not isinstance(text, str):
        return None
    tpl = _parse_coord_tuple(text)
    if not tpl:
        return None
    lat, lon = tpl
    return f"{lat:.6f}, {lon:.6f}"


def normalize_units_wrapper(text: str) -> Optional[str]:
    """Normalize units using normalization module."""
    if not text or not isinstance(text, str):
        return None
    return normalize_units(text)


def parse_quantity_scalar(text: str) -> Optional[float]:
    """Parse quantity and return scalar value."""
    if not text or not isinstance(text, str):
        return None
    try:
        result = parse_quantity(text)
        return result.value if result else None
    except Exception:
        return None


def parse_storage_gb(text: str) -> Optional[float]:
    """Parse storage capacity and convert to GB."""
    if not text or not isinstance(text, str):
        return None
    try:
        # Extract numeric value and convert common units to GB
        match = re.search(r'(\d+(?:\.\d+)?)\s*(TB|GB|MB|KB|B)\b', text, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            value = float(value)
            unit = unit.upper()
            
            # Convert to GB
            if unit == 'TB':
                return value * 1024
            elif unit == 'GB':
                return value
            elif unit == 'MB':
                return value / 1024
            elif unit == 'KB':
                return value / (1024 * 1024)
            elif unit == 'B':
                return value / (1024 * 1024 * 1024)
        
        return None
    except Exception:
        return None


def parse_power_w(text: str) -> Optional[float]:
    """Parse power and convert to watts."""
    if not text or not isinstance(text, str):
        return None
    try:
        # Extract power value and convert to watts
        match = re.search(r'(\d+(?:\.\d+)?)\s*(MW|kW|W|mW)\b', text, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            value = float(value)
            unit = unit.upper()
            
            if unit == 'MW':
                return value * 1000000
            elif unit == 'KW':
                return value * 1000
            elif unit == 'W':
                return value
            elif unit == 'MW':
                return value / 1000
        
        return None
    except Exception:
        return None


def parse_frequency_hz(text: str) -> Optional[float]:
    """Parse frequency and convert to Hz."""
    if not text or not isinstance(text, str):
        return None
    try:
        # Extract frequency value and convert to Hz
        match = re.search(r'(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)\b', text, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            value = float(value)
            unit = unit.upper()
            
            if unit == 'GHZ':
                return value * 1000000000
            elif unit == 'MHZ':
                return value * 1000000
            elif unit == 'KHZ':
                return value * 1000
            elif unit == 'HZ':
                return value
        
        return None
    except Exception:
        return None


def normalize_whitespace(text: Any) -> str:
    """Normalize whitespace in text."""
    if not isinstance(text, str):
        return str(text)
    return ' '.join(text.split())


def strip_text(text: Any) -> str:
    """Strip whitespace from text."""
    if not isinstance(text, str):
        return str(text)
    return text.strip()


def lowercase_text(text: Any) -> str:
    """Convert text to lowercase."""
    if not isinstance(text, str):
        return str(text)
    return text.lower()


def parse_employee_count(text: str) -> Optional[float]:
    """Parse employee count with k/M multipliers."""
    if not text or not isinstance(text, str):
        return None
    
    match = re.search(r'\b(\d+(?:,\d{3})*)\s*([kKmM])?\s*(?:employees?|staff|workers?)', text)
    if match:
        number_str, multiplier = match.groups()
        try:
            # Remove commas and convert to float
            number = float(number_str.replace(',', ''))
            
            # Apply multiplier
            if multiplier:
                if multiplier.lower() == 'k':
                    number *= 1000
                elif multiplier.lower() == 'm':
                    number *= 1000000
            
            return number
        except ValueError:
            return None
    
    return None


# Built-in transformation functions
TRANSFORMATIONS: Dict[str, Callable[[str], Any]] = {
    # Numeric transformations (using normalization)
    'parse_money': parse_money,
    'parse_number': parse_number,
    'parse_percent': parse_percent,
    
    # Date transformations
    'parse_date': parse_date,
    
    # URL transformations
    'normalize_url': normalize_url,
    'extract_domain': extract_domain,
    
    # Unit transformations
    'normalize_units': normalize_units_wrapper,
    'parse_quantity': parse_quantity_scalar,
    'parse_storage_gb': parse_storage_gb,
    'parse_power_w': parse_power_w,
    'parse_frequency_hz': parse_frequency_hz,
    
    # Coordinate transformations
    'parse_coordinate': parse_coordinate,
    
    # Text transformations
    'normalize_whitespace': normalize_whitespace,
    'strip': strip_text,
    'lower': lowercase_text,
    
    # Specialized parsers
    'parse_employee_count': parse_employee_count,
}


# Built-in regex patterns organized by category
built_in_rules: Dict[str, Dict[str, Dict[str, Union[str, List[str], int, Callable]]]] = {
    "identifiers": {
        "uuid4": {
            "pattern": r"\b[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "imdb_id": {
            "pattern": r"\btt\d{7,8}\b",
            "flags": 0,
            "group": 0,
        },
        "isbn_13": {
            "pattern": r"\b978\d{10}\b|\b979\d{10}\b",
            "flags": 0,
            "group": 0,
        },
        "isbn_10": {
            "pattern": r"\b\d{9}[\dX]\b",
            "flags": 0,
            "group": 0,
        },
        "ean_13": {
            "pattern": r"\b\d{13}\b",
            "flags": 0,
            "group": 0,
        },
        "asin": {
            "pattern": r"\b[A-Z0-9]{10}\b",
            "flags": 0,
            "group": 0,
        },
        "isin": {
            "pattern": r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b",
            "flags": 0,
            "group": 0,
        },
        "lei": {
            "pattern": r"\b[A-Z0-9]{18}[0-9]{2}\b",
            "flags": 0,
            "group": 0,
        },
        "iban": {
            "pattern": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b",
            "flags": 0,
            "group": 0,
        },
        "duns": {
            "pattern": r"\b\d{9}\b",
            "flags": 0,
            "group": 0,
        },
        "vat_generic": {
            "pattern": r"\b[A-Z]{2}\d{8,12}\b",
            "flags": 0,
            "group": 0,
        },
        "swift_bic": {
            "pattern": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
            "flags": 0,
            "group": 0,
        },
        "upc_a": {
            "pattern": r"\b\d{12}\b",
            "flags": 0,
            "group": 0,
        },
    },
    "contact": {
        "email": {
            "pattern": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "url": {
            "pattern": r"https?://[^\s<>\"]+",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "domain": {
            "pattern": r"\b[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z]{2,})\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "phone_us": {
            "pattern": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "flags": 0,
            "group": 0,
        },
        "twitter_handle": {
            "pattern": r"@([a-zA-Z0-9_]+)",
            "flags": 0,
            "group": 1,
        },
        "phone_e164": {
            "pattern": r"\+[1-9]\d{1,14}\b",
            "flags": 0,
            "group": 0,
        },
    },
    "money": {
        "price_symbol": {
            "pattern": r"[\$£€¥₹][\d,]+(?:\.\d{2})?",
            "flags": 0,
            "group": 0,
            "postprocess": "parse_money",
        },
        "price_iso": {
            "pattern": r"\b(?:USD|EUR|GBP|JPY|CAD|AUD)\s+[\d,]+(?:\.\d{2})?\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "parse_money",
        },
        "percent": {
            "pattern": r"\b\d+(?:\.\d+)?%",
            "flags": 0,
            "group": 0,
            "postprocess": "parse_number",
        },
    },
    "dates": {
        "iso_date": {
            "pattern": r"\b\d{4}-\d{2}-\d{2}\b",
            "flags": 0,
            "group": 0,
        },
        "us_date": {
            "pattern": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            "flags": 0,
            "group": 0,
        },
        "year": {
            "pattern": r"\b(19|20)\d{2}\b",
            "flags": 0,
            "group": 0,
            "postprocess": "parse_number",
        },
    },
    "geo": {
        "postal_us": {
            "pattern": r"\b\d{5}(?:-\d{4})?\b",
            "flags": 0,
            "group": 0,
        },
        "postal_uk": {
            "pattern": r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "postal_de": {
            "pattern": r"\b\d{5}\b",
            "flags": 0,
            "group": 0,
        },
        "postal_fr": {
            "pattern": r"\b\d{5}\b",
            "flags": 0,
            "group": 0,
        },
        "postal_ca": {
            "pattern": r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "lat_long": {
            "pattern": r"\b-?\d{1,3}\.\d+[°]?\s*[,\s]\s*-?\d{1,3}\.\d+[°]?\b",
            "flags": 0,
            "group": 0,
            "postprocess": "parse_coordinate",
        },
    },
    "measurements": {
        "length_metric": {
            "pattern": r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "length_imperial": {
            "pattern": r"\b\d+(?:\.\d+)?\s*(?:in|ft|yd|mi)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "weight_metric": {
            "pattern": r"\b\d+(?:\.\d+)?\s*(?:g|kg|mg)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "weight_imperial": {
            "pattern": r"\b\d+(?:\.\d+)?\s*(?:oz|lb|lbs)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "temperature": {
            "pattern": r"\b\d+(?:\.\d+)?\s*°[CFR]\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "dimensions": {
            "pattern": r"\b\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?\s*(?:mm|cm|m|in|ft)?\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
    },
    "product": {
        "model_number": {
            "pattern": r"\b[A-Z0-9][\w\-/]{2,20}\b",
            "flags": 0,
            "group": 0,
        },
        "color": {
            "pattern": r"\b(?:black|white|red|blue|green|yellow|orange|purple|pink|brown|gray|grey|silver|gold|beige|navy|maroon|teal|olive|lime|aqua|fuchsia)\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "lower",
        },
        "size_clothing": {
            "pattern": r"\b(?:XXS|XS|S|M|L|XL|XXL|XXXL)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "warranty": {
            "pattern": r"\b\d+\s*(?:year|month|yr|mo)s?\s*warranty\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "storage_gb_tb": {
            "pattern": r"\b(\d+(?:\.\d+)?)\s*(GB|TB)\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "parse_storage_gb",
        },
        "battery_mah": {
            "pattern": r"\b(\d+(?:,\d{3})*)\s*mAh\b",
            "flags": re.IGNORECASE,
            "group": 1,
            "postprocess": "parse_number",
        },
        "clock_ghz": {
            "pattern": r"\b(\d+(?:\.\d+)?)\s*GHz\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "parse_frequency_hz",
        },
        "resolution_px": {
            "pattern": r"\b(\d+)\s*[x×]\s*(\d+)\s*(?:px|pixels?)?\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "video_resolution_tag": {
            "pattern": r"\b(?:480p|720p|1080p|1440p|2160p|4K|8K)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "bitrate_kbps_mbps": {
            "pattern": r"\b(\d+(?:\.\d+)?)\s*(kbps|Mbps|Gbps)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "power_w_kw": {
            "pattern": r"\b(\d+(?:\.\d+)?)\s*(W|kW|MW)\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "parse_power_w",
        },
    },
    "media": {
        "runtime": {
            "pattern": r"\b(\d{1,3})\s*(?:min|minutes?|hrs?|hours?)\b",
            "flags": re.IGNORECASE,
            "group": 1,
            "postprocess": "parse_number",
        },
        "age_rating": {
            "pattern": r"\b(?:G|PG|PG-13|R|NC-17|NR|Not Rated)\b",
            "flags": 0,
            "group": 0,
        },
        "language": {
            "pattern": r"\b(?:English|Spanish|French|German|Italian|Portuguese|Russian|Chinese|Japanese|Korean|Arabic|Hindi)\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "lower",
        },
        "genre": {
            "pattern": r"\b(?:Action|Adventure|Animation|Biography|Comedy|Crime|Documentary|Drama|Family|Fantasy|Film-Noir|History|Horror|Music|Musical|Mystery|News|Romance|Sci-Fi|Sport|Thriller|War|Western)\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "lower",
        },
    },
    "company": {
        "legal_suffix": {
            "pattern": r"\b(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|GmbH|SA|AG|Oy|Pty Ltd)\b",
            "flags": re.IGNORECASE,
            "group": 0,
        },
        "employee_count": {
            "pattern": r"\b(\d+(?:,\d{3})*)\s*([kKmM])?\s*(?:employees?|staff|workers?)\b",
            "flags": re.IGNORECASE,
            "group": 0,
            "postprocess": "parse_employee_count",
        },
        "registration_number_generic": {
            "pattern": r"\b[A-Z0-9]{7,12}\b",
            "flags": 0,
            "group": 0,
        },
    },
    "key_value": {
        "colon_separator": {
            "pattern": r"([^:\n]+):\s*([^\n]+)",
            "flags": 0,
            "group": (1, 2),
            "postprocess": "strip",
        },
        "equals_separator": {
            "pattern": r"([^=\n]+)=\s*([^\n]+)",
            "flags": 0,
            "group": (1, 2),
            "postprocess": "strip",
        },
        "dash_separator": {
            "pattern": r"([^-\n]+?)\s*-\s*([^\n]+)",
            "flags": 0,
            "group": (1, 2),
            "postprocess": "strip",
        },
        "semicolon_separator": {
            "pattern": r"([^;\n]+);\s*([^\n]+)",
            "flags": 0,
            "group": (1, 2),
            "postprocess": "strip",
        },
    },
}
