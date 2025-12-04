"""
Unit tests for the PyDI normalization module.

Tests cover the main functionalities demonstrated in the normalization workflow:
1. Scale modifier detection and expansion (MEO, MEUR, kEUR, million, etc.)
2. Unit parsing and conversion (km, miles, m², etc.)
3. Country/currency/language normalization
4. Phone number parsing and formatting
5. Email validation and normalization
6. Standard number validation (IBAN, ISBN, VAT)
7. Profiling and type detection
8. NormalizationSpec and DataFrame transformation
"""

import pytest
import pandas as pd

from PyDI.normalization import (
    # Scale modifiers
    detect_scale_modifier,
    expand_scale,
    parse_scaled_number,
    ScaleModifier,
    # Units
    parse_quantity,
    convert_units,
    normalize_quantity,
    are_compatible,
    is_valid_unit,
    # Country/currency/language
    normalize_country,
    normalize_currency,
    normalize_language,
    lookup_country,
    lookup_currency,
    # Phone
    parse_phone,
    format_phone,
    validate_phone,
    # Email
    validate_email,
    normalize_email,
    is_valid_email,
    # Standard numbers
    is_valid_iban,
    is_valid_isbn,
    is_valid_vat,
    detect_stdnum_type,
    format_stdnum,
    # Profiling
    profile_dataframe,
    profile_column,
    # Spec and transform
    NormalizationSpec,
    ColumnSpec,
    transform_dataframe,
    normalize_dataframe,
)


class TestScaleModifiers:
    """Tests for scale modifier detection and expansion."""

    def test_detect_meo(self):
        """Test detection of MEO (Portuguese style millions)."""
        modifier = detect_scale_modifier("5.2 MEO")
        assert modifier is not None
        assert modifier.name == "MEO"
        assert modifier.multiplier == 1_000_000

    def test_detect_meur(self):
        """Test detection of MEUR (millions of euros)."""
        modifier = detect_scale_modifier("2.5 MEUR")
        assert modifier is not None
        assert modifier.name == "MEUR"
        assert modifier.multiplier == 1_000_000

    def test_detect_keur(self):
        """Test detection of kEUR (thousands of euros)."""
        modifier = detect_scale_modifier("750 kEUR")
        assert modifier is not None
        assert modifier.name == "kEUR"
        assert modifier.multiplier == 1_000

    def test_detect_teur(self):
        """Test detection of TEUR (thousands of euros, German style)."""
        modifier = detect_scale_modifier("500 TEUR")
        assert modifier is not None
        assert modifier.name == "TEUR"
        assert modifier.multiplier == 1_000

    def test_detect_million(self):
        """Test detection of 'million' word."""
        modifier = detect_scale_modifier("50 million")
        assert modifier is not None
        assert modifier.name == "million"
        assert modifier.multiplier == 1_000_000

    def test_detect_billion(self):
        """Test detection of 'billion' word."""
        modifier = detect_scale_modifier("1.2 billion SEK")
        assert modifier is not None
        assert modifier.name == "billion"
        assert modifier.multiplier == 1_000_000_000

    def test_detect_m_abbreviation(self):
        """Test detection of 'M' abbreviation for millions."""
        modifier = detect_scale_modifier("10M")
        assert modifier is not None
        assert modifier.name == "M"
        assert modifier.multiplier == 1_000_000

    def test_no_modifier_for_units(self):
        """Test that km is not detected as a scale modifier."""
        modifier = detect_scale_modifier("5 km distance")
        assert modifier is None

    def test_parse_scaled_number_meo(self):
        """Test parsing number with MEO modifier."""
        result = parse_scaled_number("5.2 MEO")
        assert result is not None
        value, modifier = result
        assert value == 5_200_000
        assert modifier.name == "MEO"

    def test_parse_scaled_number_million(self):
        """Test parsing number with 'million' word."""
        result = parse_scaled_number("¥50 million")
        assert result is not None
        value, modifier = result
        assert value == 50_000_000

    def test_parse_scaled_number_plain(self):
        """Test parsing plain number without modifier."""
        result = parse_scaled_number("100")
        assert result is not None
        value, modifier = result
        assert value == 100.0
        assert modifier is None

    def test_expand_scale(self):
        """Test expanding scale modifier."""
        result = expand_scale("5 MEO", 5.0)
        assert result.scaled_value == 5_000_000
        assert result.modifier.name == "MEO"


class TestUnitParsing:
    """Tests for unit parsing and conversion."""

    def test_parse_quantity_km(self):
        """Test parsing kilometers."""
        result = parse_quantity("45 km")
        assert result is not None
        assert result.magnitude == 45.0
        assert "km" in result.unit or "kilometer" in result.unit

    def test_parse_quantity_miles(self):
        """Test parsing miles."""
        result = parse_quantity("100 miles")
        assert result is not None
        assert result.magnitude == 100.0
        assert "mi" in result.unit or "mile" in result.unit

    def test_parse_quantity_sqm(self):
        """Test parsing square meters."""
        result = parse_quantity("2500 m²")
        assert result is not None
        assert result.magnitude == 2500.0

    def test_convert_km_to_m(self):
        """Test converting kilometers to meters."""
        result = convert_units(5, "km", "m")
        assert result == pytest.approx(5000.0)

    def test_convert_miles_to_km(self):
        """Test converting miles to kilometers."""
        result = convert_units(100, "miles", "km")
        assert result == pytest.approx(160.93, rel=0.01)

    def test_convert_sqft_to_sqm(self):
        """Test converting square feet to square meters."""
        result = convert_units(50000, "sq ft", "m**2")
        assert result == pytest.approx(4645.15, rel=0.01)

    def test_convert_fahrenheit_to_celsius(self):
        """Test converting Fahrenheit to Celsius."""
        # Use degF and degC which are the Pint-recognized temperature unit aliases
        result = convert_units(68, "degF", "degC")
        assert result == pytest.approx(20.0, rel=0.01)

    def test_normalize_quantity_with_conversion(self):
        """Test normalizing quantity with unit conversion."""
        result = normalize_quantity("5 km", target_unit="m")
        assert result is not None
        value, unit = result
        assert value == pytest.approx(5000.0)

    def test_are_compatible_length(self):
        """Test compatibility check for length units."""
        assert are_compatible("km", "miles") is True
        assert are_compatible("km", "m") is True

    def test_are_compatible_different_dimensions(self):
        """Test incompatibility of different dimensions."""
        assert are_compatible("km", "kg") is False

    def test_is_valid_unit(self):
        """Test unit validation."""
        assert is_valid_unit("km") is True
        assert is_valid_unit("miles") is True
        assert is_valid_unit("invalid_unit_xyz") is False


class TestCountryNormalization:
    """Tests for country code normalization."""

    def test_normalize_country_name_to_alpha2(self):
        """Test normalizing country name to alpha-2 code."""
        assert normalize_country("Germany") == "DE"
        assert normalize_country("United Kingdom") == "GB"
        assert normalize_country("Portugal") == "PT"

    def test_normalize_country_alpha3_to_alpha2(self):
        """Test normalizing alpha-3 to alpha-2 code."""
        assert normalize_country("DEU") == "DE"
        assert normalize_country("ESP") == "ES"

    def test_normalize_country_to_name(self):
        """Test normalizing to country name."""
        assert normalize_country("DE", output_format="name") == "Germany"
        assert normalize_country("DEU", output_format="name") == "Germany"

    def test_normalize_country_to_alpha3(self):
        """Test normalizing to alpha-3 code."""
        assert normalize_country("Germany", output_format="alpha_3") == "DEU"

    def test_normalize_country_invalid(self):
        """Test that invalid country returns None."""
        assert normalize_country("InvalidCountry") is None

    def test_lookup_country(self):
        """Test country lookup returns full info."""
        info = lookup_country("Germany")
        assert info is not None
        assert info.alpha_2 == "DE"
        assert info.alpha_3 == "DEU"
        assert info.name == "Germany"


class TestCurrencyNormalization:
    """Tests for currency code normalization."""

    def test_normalize_currency_name_to_alpha3(self):
        """Test normalizing currency name to alpha-3 code."""
        assert normalize_currency("Euro") == "EUR"
        assert normalize_currency("euro") == "EUR"

    def test_normalize_currency_code(self):
        """Test that currency codes are returned as-is."""
        assert normalize_currency("EUR") == "EUR"
        assert normalize_currency("USD") == "USD"
        assert normalize_currency("GBP") == "GBP"

    def test_normalize_currency_to_name(self):
        """Test normalizing to currency name."""
        assert normalize_currency("EUR", output_format="name") == "Euro"

    def test_normalize_currency_invalid(self):
        """Test that invalid currency returns None."""
        assert normalize_currency("InvalidCurrency") is None

    def test_lookup_currency(self):
        """Test currency lookup returns full info."""
        info = lookup_currency("EUR")
        assert info is not None
        assert info.alpha_3 == "EUR"
        assert info.name == "Euro"


class TestLanguageNormalization:
    """Tests for language code normalization."""

    def test_normalize_language_name_to_alpha2(self):
        """Test normalizing language name to alpha-2 code."""
        assert normalize_language("English") == "en"
        assert normalize_language("German") == "de"

    def test_normalize_language_alpha3_to_alpha2(self):
        """Test normalizing alpha-3 to alpha-2."""
        assert normalize_language("eng") == "en"
        assert normalize_language("deu") == "de"

    def test_normalize_language_to_name(self):
        """Test normalizing to language name."""
        assert normalize_language("en", output_format="name") == "English"


class TestPhoneNumberHandling:
    """Tests for phone number parsing and formatting."""

    def test_parse_phone_german(self):
        """Test parsing German phone number."""
        info = parse_phone("+49 30 12345678")
        assert info is not None
        assert info.country_code == 49
        assert info.region == "DE"

    def test_parse_phone_uk(self):
        """Test parsing UK phone number."""
        info = parse_phone("+44 20 7946 0958")
        assert info is not None
        assert info.country_code == 44
        assert info.region == "GB"

    def test_parse_phone_with_00_prefix(self):
        """Test parsing phone with 00 international prefix."""
        info = parse_phone("00351 21 1234567")
        assert info is not None
        assert info.country_code == 351
        assert info.region == "PT"

    def test_format_phone_e164(self):
        """Test formatting phone to E.164."""
        result = format_phone("+49 30 12345678", "E164")
        assert result == "+493012345678"

    def test_format_phone_international(self):
        """Test formatting phone to international format."""
        result = format_phone("+49 30 12345678", "INTERNATIONAL")
        assert result is not None
        assert result.startswith("+49")

    def test_validate_phone_valid(self):
        """Test phone validation for valid number."""
        assert validate_phone("+49 30 12345678") is True

    def test_validate_phone_invalid(self):
        """Test phone validation for invalid string."""
        assert validate_phone("invalid") is False


class TestEmailHandling:
    """Tests for email validation and normalization."""

    def test_validate_email_valid(self):
        """Test email validation for valid email."""
        info = validate_email("user@example.com")
        assert info.is_valid is True
        assert info.normalized == "user@example.com"

    def test_validate_email_uppercase(self):
        """Test email validation normalizes domain but preserves local part case."""
        info = validate_email("USER@EXAMPLE.COM")
        assert info.is_valid is True
        # Note: email-validator preserves local part case per RFC 5321
        # Only the domain is lowercased
        assert info.normalized == "USER@example.com"

    def test_validate_email_invalid(self):
        """Test email validation for invalid email."""
        info = validate_email("invalid")
        assert info.is_valid is False

    def test_normalize_email(self):
        """Test email normalization."""
        assert normalize_email("INFO@ACME.DE") == "info@acme.de"
        assert normalize_email("invalid") is None

    def test_is_valid_email(self):
        """Test is_valid_email helper."""
        assert is_valid_email("user@example.com") is True
        assert is_valid_email("invalid") is False


class TestStandardNumbers:
    """Tests for standard number validation (IBAN, ISBN, VAT)."""

    def test_is_valid_iban_valid(self):
        """Test IBAN validation for valid IBAN."""
        # German IBAN
        assert is_valid_iban("DE89370400440532013000") is True

    def test_is_valid_iban_invalid(self):
        """Test IBAN validation for invalid IBAN."""
        assert is_valid_iban("invalid-iban") is False
        assert is_valid_iban("DE12345") is False

    def test_is_valid_isbn_valid(self):
        """Test ISBN validation for valid ISBN."""
        assert is_valid_isbn("978-0-306-40615-7") is True
        assert is_valid_isbn("9780306406157") is True

    def test_is_valid_isbn_invalid(self):
        """Test ISBN validation for invalid ISBN."""
        assert is_valid_isbn("invalid-isbn") is False

    def test_detect_stdnum_type_iban(self):
        """Test detecting IBAN type."""
        result = detect_stdnum_type("DE89370400440532013000")
        assert result == "IBAN"

    def test_detect_stdnum_type_isbn(self):
        """Test detecting ISBN type."""
        result = detect_stdnum_type("978-0-306-40615-7")
        assert result == "ISBN"

    def test_format_stdnum_iban(self):
        """Test formatting IBAN."""
        result = format_stdnum("DE89370400440532013000", "IBAN")
        assert result is not None
        assert "DE89" in result


class TestProfiling:
    """Tests for DataFrame profiling and type detection."""

    def test_profile_dataframe_basic(self):
        """Test basic DataFrame profiling."""
        df = pd.DataFrame({
            "revenue": ["5 MEO", "10 MEO", "2.5 MEO"],
            "country": ["Germany", "France", "Spain"],
            "email": ["a@test.com", "b@test.com", "c@test.com"],
        })
        profile = profile_dataframe(df)

        assert profile.row_count == 3
        assert profile.column_count == 3
        assert "revenue" in profile.columns
        assert "country" in profile.columns
        assert "email" in profile.columns

    def test_profile_detects_scaled_number(self):
        """Test that profiling detects scaled numbers."""
        df = pd.DataFrame({
            "revenue": ["5 MEO", "10 MEUR", "2.5 million", "100 kEUR"],
        })
        profile = profile_dataframe(df)

        assert profile.columns["revenue"].detected_type == "scaled_number"
        assert profile.columns["revenue"].scale_info is not None

    def test_profile_detects_country(self):
        """Test that profiling detects country codes."""
        df = pd.DataFrame({
            "country": ["Germany", "France", "DEU", "ES", "United Kingdom"],
        })
        profile = profile_dataframe(df)

        assert profile.columns["country"].detected_type == "country"

    def test_profile_detects_email(self):
        """Test that profiling detects emails."""
        df = pd.DataFrame({
            "contact": ["user1@example.com", "user2@test.org", "info@company.de"],
        })
        profile = profile_dataframe(df)

        assert profile.columns["contact"].detected_type == "email"

    def test_profile_detects_phone(self):
        """Test that profiling detects phone numbers."""
        df = pd.DataFrame({
            "phone": ["+49 30 12345678", "+44 20 7946 0958", "+33 1 23 45 67 89"],
        })
        profile = profile_dataframe(df)

        assert profile.columns["phone"].detected_type == "phone"

    def test_profile_detects_unit_quantity(self):
        """Test that profiling detects quantities with units."""
        df = pd.DataFrame({
            "distance": ["45 km", "100 miles", "50 kilometers", "200 m"],
        })
        profile = profile_dataframe(df)

        assert profile.columns["distance"].detected_type == "unit_quantity"
        assert profile.columns["distance"].unit_info is not None

    def test_profile_column(self):
        """Test profiling a single column."""
        series = pd.Series(["DE", "FR", "ES", "GB", "IT"], name="country")
        profile = profile_column(series)

        assert profile.name == "country"
        assert profile.detected_type == "country"
        assert profile.total_count == 5
        assert profile.null_count == 0

    def test_profile_generates_summary(self):
        """Test that profile generates a readable summary."""
        df = pd.DataFrame({
            "revenue": ["5 MEO", "10 MEO"],
            "country": ["Germany", "France"],
        })
        profile = profile_dataframe(df)
        summary = profile.summary()

        assert "DataFrame Profile" in summary
        assert "revenue" in summary
        assert "country" in summary


class TestNormalizationSpec:
    """Tests for NormalizationSpec configuration."""

    def test_create_spec(self):
        """Test creating a normalization spec."""
        spec = NormalizationSpec()
        spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
        spec.set_column("country", country_format="alpha_2")

        assert "revenue" in spec.columns
        assert "country" in spec.columns
        assert spec.columns["revenue"].expand_scale_modifiers is True
        assert spec.columns["country"].country_format == "alpha_2"

    def test_spec_to_dict(self):
        """Test serializing spec to dict."""
        spec = NormalizationSpec()
        spec.set_column("revenue", expand_scale_modifiers=True)

        d = spec.to_dict()
        assert "columns" in d
        assert "revenue" in d["columns"]

    def test_spec_to_json(self):
        """Test serializing spec to JSON."""
        spec = NormalizationSpec()
        spec.set_column("country", country_format="alpha_2")

        json_str = spec.to_json()
        assert "country" in json_str
        assert "alpha_2" in json_str

    def test_spec_from_dict(self):
        """Test deserializing spec from dict."""
        d = {
            "columns": {
                "revenue": {"expand_scale_modifiers": True, "output_type": "float"},
            }
        }
        spec = NormalizationSpec.from_dict(d)

        assert "revenue" in spec.columns
        assert spec.columns["revenue"].expand_scale_modifiers is True

    def test_spec_from_profile(self):
        """Test creating spec from profile."""
        df = pd.DataFrame({
            "revenue": ["5 MEO", "10 MEO", "2.5 MEO", "100 MEO"],
            "country": ["Germany", "France", "Spain", "Italy"],
        })
        profile = profile_dataframe(df)
        spec = NormalizationSpec.from_profile(profile)

        # Should auto-detect normalizations
        assert len(spec.columns) > 0

    def test_column_spec_defaults(self):
        """Test ColumnSpec default values."""
        spec = ColumnSpec()
        assert spec.output_type == "keep"
        assert spec.on_failure == "keep"
        assert spec.expand_scale_modifiers is False
        assert spec.normalize_email is False


class TestDataFrameTransformation:
    """Tests for DataFrame transformation."""

    def test_transform_scale_modifiers(self):
        """Test transforming scale modifiers."""
        df = pd.DataFrame({
            "revenue": ["5 MEO", "2.5 MEUR", "750 kEUR", "10M"],
        })
        spec = NormalizationSpec()
        spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")

        result = transform_dataframe(df, spec)

        assert result.dataframe["revenue"].iloc[0] == pytest.approx(5_000_000)
        assert result.dataframe["revenue"].iloc[1] == pytest.approx(2_500_000)
        assert result.dataframe["revenue"].iloc[2] == pytest.approx(750_000)
        assert result.dataframe["revenue"].iloc[3] == pytest.approx(10_000_000)

    def test_transform_country_codes(self):
        """Test transforming country codes."""
        df = pd.DataFrame({
            "country": ["Germany", "DEU", "United Kingdom", "ES"],
        })
        spec = NormalizationSpec()
        spec.set_column("country", country_format="alpha_2")

        result = transform_dataframe(df, spec)

        assert result.dataframe["country"].iloc[0] == "DE"
        assert result.dataframe["country"].iloc[1] == "DE"
        assert result.dataframe["country"].iloc[2] == "GB"
        assert result.dataframe["country"].iloc[3] == "ES"

    def test_transform_currency_codes(self):
        """Test transforming currency codes."""
        df = pd.DataFrame({
            "currency": ["Euro", "GBP", "US Dollar"],
        })
        spec = NormalizationSpec()
        spec.set_column("currency", currency_format="alpha_3")

        result = transform_dataframe(df, spec)

        assert result.dataframe["currency"].iloc[0] == "EUR"
        assert result.dataframe["currency"].iloc[1] == "GBP"
        assert result.dataframe["currency"].iloc[2] == "USD"

    def test_transform_email(self):
        """Test transforming emails."""
        df = pd.DataFrame({
            "email": ["USER@EXAMPLE.COM", "  info@test.de  "],
        })
        spec = NormalizationSpec()
        spec.set_column("email", normalize_email=True)

        result = transform_dataframe(df, spec)

        # Note: email-validator preserves local part case per RFC 5321
        assert result.dataframe["email"].iloc[0] == "USER@example.com"
        assert result.dataframe["email"].iloc[1] == "info@test.de"

    def test_transform_phone(self):
        """Test transforming phone numbers."""
        df = pd.DataFrame({
            "phone": ["+49 30 12345678", "+44 20 7946 0958"],
        })
        spec = NormalizationSpec()
        spec.set_column("phone", phone_format="e164")

        result = transform_dataframe(df, spec)

        assert result.dataframe["phone"].iloc[0] == "+493012345678"
        assert result.dataframe["phone"].iloc[1] == "+442079460958"

    def test_transform_unit_conversion(self):
        """Test transforming with unit conversion."""
        df = pd.DataFrame({
            "distance": ["45 km", "100 miles", "2500 m"],
        })
        spec = NormalizationSpec()
        spec.set_column("distance", target_unit="km", output_type="float")

        result = transform_dataframe(df, spec)

        assert result.dataframe["distance"].iloc[0] == pytest.approx(45.0)
        assert result.dataframe["distance"].iloc[1] == pytest.approx(160.93, rel=0.01)
        assert result.dataframe["distance"].iloc[2] == pytest.approx(2.5)

    def test_transform_text_case(self):
        """Test transforming text case."""
        df = pd.DataFrame({
            "name": ["acme industries", "TECHSTART LTD"],
        })
        spec = NormalizationSpec()
        spec.set_column("name", case="title", strip_whitespace=True)

        result = transform_dataframe(df, spec)

        assert result.dataframe["name"].iloc[0] == "Acme Industries"
        assert result.dataframe["name"].iloc[1] == "Techstart Ltd"

    def test_transform_percentage_to_decimal(self):
        """Test transforming percentages to decimals."""
        df = pd.DataFrame({
            "growth": ["12.5%", "8%", "+5.7%"],
        })
        spec = NormalizationSpec()
        spec.set_column("growth", convert_percentage="to_decimal", output_type="float")

        result = transform_dataframe(df, spec)

        assert result.dataframe["growth"].iloc[0] == pytest.approx(0.125)
        assert result.dataframe["growth"].iloc[1] == pytest.approx(0.08)
        assert result.dataframe["growth"].iloc[2] == pytest.approx(0.057)

    def test_transform_result_metadata(self):
        """Test that transform result contains correct metadata."""
        df = pd.DataFrame({
            "country": ["Germany", "InvalidCountry", "France"],
        })
        spec = NormalizationSpec()
        spec.set_column("country", country_format="alpha_2")

        result = transform_dataframe(df, spec)

        assert result.total_transformed == 2
        assert result.total_failed == 1
        assert "country" in result.columns
        assert result.columns["country"].values_transformed == 2
        assert result.columns["country"].values_failed == 1

    def test_transform_on_failure_null(self):
        """Test on_failure='null' sets failed values to None."""
        df = pd.DataFrame({
            "country": ["Germany", "InvalidCountry"],
        })
        spec = NormalizationSpec()
        spec.set_column("country", country_format="alpha_2", on_failure="null")

        result = transform_dataframe(df, spec)

        assert result.dataframe["country"].iloc[0] == "DE"
        assert pd.isna(result.dataframe["country"].iloc[1])

    def test_transform_preserves_nulls(self):
        """Test that null values are preserved."""
        df = pd.DataFrame({
            "country": ["Germany", None, "France"],
        })
        spec = NormalizationSpec()
        spec.set_column("country", country_format="alpha_2")

        result = transform_dataframe(df, spec)

        assert result.dataframe["country"].iloc[0] == "DE"
        assert pd.isna(result.dataframe["country"].iloc[1])
        assert result.dataframe["country"].iloc[2] == "FR"


class TestNormalizeDataFrame:
    """Tests for the normalize_dataframe convenience function."""

    def test_normalize_with_spec(self):
        """Test normalizing with explicit spec."""
        df = pd.DataFrame({
            "country": ["Germany", "France"],
        })
        spec = NormalizationSpec()
        spec.set_column("country", country_format="alpha_2")

        result = normalize_dataframe(df, spec)

        assert result["country"].iloc[0] == "DE"
        assert result["country"].iloc[1] == "FR"

    def test_normalize_auto(self):
        """Test auto-normalization."""
        df = pd.DataFrame({
            "country": ["Germany", "France", "Spain", "Italy"],
            "email": ["A@TEST.COM", "B@TEST.COM", "C@TEST.COM", "D@TEST.COM"],
        })

        result = normalize_dataframe(df, auto=True)

        # Country should be normalized to alpha-2
        assert result["country"].iloc[0] == "DE"
        # Email domain should be lowercase (local part case preserved per RFC 5321)
        assert result["email"].iloc[0] == "A@test.com"

    def test_normalize_no_spec_returns_copy(self):
        """Test that no spec and auto=False returns a copy."""
        df = pd.DataFrame({
            "value": [1, 2, 3],
        })

        result = normalize_dataframe(df, auto=False)

        # Should be a copy, not the same object
        assert result is not df
        assert result.equals(df)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling empty DataFrame."""
        df = pd.DataFrame()
        profile = profile_dataframe(df)

        assert profile.row_count == 0
        assert profile.column_count == 0

    def test_all_null_column(self):
        """Test handling column with all nulls."""
        df = pd.DataFrame({
            "empty": [None, None, None],
        })
        profile = profile_dataframe(df)

        assert profile.columns["empty"].detected_type == "empty"
        assert profile.columns["empty"].null_count == 3

    def test_parse_empty_string(self):
        """Test parsing empty strings."""
        assert parse_scaled_number("") is None
        assert parse_quantity("") is None
        assert normalize_country("") is None

    def test_parse_none(self):
        """Test parsing None values."""
        assert parse_scaled_number(None) is None
        assert normalize_country(None) is None

    def test_transform_missing_column(self):
        """Test transforming spec with missing column."""
        df = pd.DataFrame({
            "existing": [1, 2, 3],
        })
        spec = NormalizationSpec()
        spec.set_column("missing_column", output_type="float")

        # Should not raise, just skip the missing column
        result = transform_dataframe(df, spec)
        assert "existing" in result.dataframe.columns
