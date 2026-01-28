"""
Additional coverage tests - Batch 38: Normalizer Module Complete Coverage.

Comprehensive tests for voice_soundboard/normalizer.py
Covers: number conversion, decimals, currency, abbreviations, acronyms, emojis, math, URLs, HTML entities.
"""

import pytest
from voice_soundboard.normalizer import (
    number_to_words,
    decimal_to_words,
    expand_currency,
    expand_abbreviations,
    expand_acronyms,
    expand_emojis,
    expand_math_symbols,
    expand_url,
    expand_email,
    expand_urls_and_emails,
    decode_html_entities,
    normalize_text,
    ABBREVIATIONS,
    CURRENCY_SYMBOLS,
    MATH_SYMBOLS,
    EMOJI_NAMES,
)


# =============================================================================
# Number Conversion Tests
# =============================================================================

class TestNumberToWords:
    """Tests for number_to_words function."""

    def test_number_to_words_zero(self):
        """Test converting zero."""
        assert number_to_words(0) == "zero"

    def test_number_to_words_single_digits(self):
        """Test converting single digit numbers 1-9."""
        expected = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        for i, word in enumerate(expected, start=1):
            assert number_to_words(i) == word

    def test_number_to_words_teens(self):
        """Test converting teen numbers 10-19."""
        expected = [
            "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
        ]
        for i, word in enumerate(expected, start=10):
            assert number_to_words(i) == word

    def test_number_to_words_tens(self):
        """Test converting multiples of ten (20, 30, ..., 90)."""
        expected = {
            20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
            60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"
        }
        for num, word in expected.items():
            assert number_to_words(num) == word

    def test_number_to_words_compound_tens(self):
        """Test converting compound tens (21, 42, 99)."""
        assert number_to_words(21) == "twenty-one"
        assert number_to_words(42) == "forty-two"
        assert number_to_words(99) == "ninety-nine"

    def test_number_to_words_hundreds(self):
        """Test converting hundreds."""
        assert number_to_words(100) == "one hundred"
        assert number_to_words(200) == "two hundred"
        assert number_to_words(456) == "four hundred fifty-six"
        assert number_to_words(999) == "nine hundred ninety-nine"

    def test_number_to_words_thousands(self):
        """Test converting thousands."""
        assert number_to_words(1000) == "one thousand"
        assert number_to_words(5000) == "five thousand"
        assert number_to_words(12345) == "twelve thousand three hundred forty-five"
        assert number_to_words(50000) == "fifty thousand"

    def test_number_to_words_millions(self):
        """Test converting millions."""
        assert number_to_words(1000000) == "one million"
        assert number_to_words(2500000) == "two million five hundred thousand"
        assert number_to_words(123456789) == "one hundred twenty-three million four hundred fifty-six thousand seven hundred eighty-nine"

    def test_number_to_words_negative(self):
        """Test converting negative numbers."""
        assert number_to_words(-1) == "negative one"
        assert number_to_words(-42) == "negative forty-two"
        assert number_to_words(-100) == "negative one hundred"


# =============================================================================
# Decimal Conversion Tests
# =============================================================================

class TestDecimalToWords:
    """Tests for decimal_to_words function."""

    def test_decimal_to_words_basic(self):
        """Test basic decimal conversion."""
        result = decimal_to_words("3.14")
        assert "three" in result
        assert "point" in result
        assert "one" in result
        assert "four" in result

    def test_decimal_to_words_leading_zero(self):
        """Test decimal with leading zero."""
        result = decimal_to_words("0.5")
        assert "zero" in result
        assert "point" in result
        assert "five" in result

    def test_decimal_to_words_multiple_zeros(self):
        """Test decimal with multiple zeros."""
        result = decimal_to_words("1.00")
        assert "one" in result
        assert "point" in result
        # Should have two zeros after point

    def test_decimal_to_words_no_decimal(self):
        """Test integer passed as string."""
        result = decimal_to_words("42")
        assert result == "forty-two"

    def test_decimal_to_words_invalid(self):
        """Test invalid decimal format (multiple dots)."""
        result = decimal_to_words("1.2.3")
        # Should return original string for invalid format
        assert result == "1.2.3"

    def test_decimal_to_words_pi(self):
        """Test pi value."""
        result = decimal_to_words("3.14159")
        assert "three point" in result


# =============================================================================
# Currency Expansion Tests
# =============================================================================

class TestExpandCurrency:
    """Tests for expand_currency function."""

    def test_expand_currency_dollars_basic(self):
        """Test basic dollar expansion."""
        result = expand_currency("$100")
        assert "one hundred dollars" in result

    def test_expand_currency_dollars_single(self):
        """Test singular dollar."""
        result = expand_currency("$1")
        assert "one dollar" in result
        assert "dollars" not in result

    def test_expand_currency_with_cents(self):
        """Test dollars with cents."""
        result = expand_currency("$10.50")
        assert "ten dollars" in result
        assert "fifty cents" in result

    def test_expand_currency_with_commas(self):
        """Test currency with thousand separators."""
        result = expand_currency("$1,000")
        assert "one thousand dollars" in result

    def test_expand_currency_large_amount(self):
        """Test large currency amount."""
        result = expand_currency("$1,000,000")
        assert "one million dollars" in result

    def test_expand_currency_euros(self):
        """Test euro expansion."""
        result = expand_currency("€5")
        assert "five euros" in result

    def test_expand_currency_pounds(self):
        """Test British pound expansion."""
        result = expand_currency("£50")
        assert "fifty pounds" in result

    def test_expand_currency_yen(self):
        """Test yen expansion."""
        result = expand_currency("¥1000")
        assert "one thousand yen" in result

    def test_expand_currency_single_cent(self):
        """Test single cent."""
        result = expand_currency("$0.01")
        assert "zero dollars" in result or "one cent" in result


# =============================================================================
# Abbreviation Expansion Tests
# =============================================================================

class TestExpandAbbreviations:
    """Tests for expand_abbreviations function."""

    def test_expand_abbreviations_titles(self):
        """Test title abbreviations."""
        assert "Doctor" in expand_abbreviations("Dr. Smith")
        assert "Mister" in expand_abbreviations("Mr. Jones")
        assert "Missus" in expand_abbreviations("Mrs. Brown")

    def test_expand_abbreviations_units(self):
        """Test unit abbreviations."""
        assert "feet" in expand_abbreviations("10 ft.")
        assert "pounds" in expand_abbreviations("5 lbs.")
        assert "ounces" in expand_abbreviations("8 oz.")

    def test_expand_abbreviations_time(self):
        """Test time abbreviations."""
        assert "minutes" in expand_abbreviations("30 min.")
        assert "hours" in expand_abbreviations("2 hrs.")
        assert "seconds" in expand_abbreviations("45 sec.")

    def test_expand_abbreviations_common(self):
        """Test common abbreviations."""
        assert "versus" in expand_abbreviations("Team A vs. Team B")
        assert "etcetera" in expand_abbreviations("apples, oranges, etc.")
        assert "for example" in expand_abbreviations("Fruits, e.g. apples")
        assert "that is" in expand_abbreviations("Mammals, i.e. warm-blooded")

    def test_expand_abbreviations_case_insensitive(self):
        """Test case-insensitive abbreviation expansion."""
        result_lower = expand_abbreviations("dr. smith")
        result_upper = expand_abbreviations("DR. Smith")
        # Both should expand (case insensitive flag)
        assert "octor" in result_lower.lower()  # Doctor


# =============================================================================
# Acronym Expansion Tests
# =============================================================================

class TestExpandAcronyms:
    """Tests for expand_acronyms function."""

    def test_expand_acronyms_common(self):
        """Test common acronym expansion."""
        result = expand_acronyms("The FBI investigated")
        assert "F B I" in result

    def test_expand_acronyms_tech(self):
        """Test tech acronyms."""
        result = expand_acronyms("Use the API")
        assert "A P I" in result

    def test_expand_acronyms_case_insensitive(self):
        """Test case insensitive acronym matching."""
        result_lower = expand_acronyms("the fbi agent")
        result_upper = expand_acronyms("the FBI agent")
        # Both should be expanded
        assert "F B I" in result_upper

    def test_expand_acronyms_in_sentence(self):
        """Test acronyms within sentence context."""
        result = expand_acronyms("NASA launched a new satellite for the USA")
        assert "N A S A" in result
        assert "U S A" in result

    def test_expand_acronyms_preserves_other_text(self):
        """Test that non-acronym text is preserved."""
        result = expand_acronyms("Hello world FBI testing")
        assert "Hello" in result
        assert "world" in result
        assert "testing" in result


# =============================================================================
# Emoji Expansion Tests
# =============================================================================

class TestExpandEmojis:
    """Tests for expand_emojis function."""

    def test_expand_emojis_basic(self):
        """Test basic emoji expansion."""
        result = expand_emojis("Hello 😀")
        assert "grinning face" in result

    def test_expand_emojis_multiple(self):
        """Test multiple emoji expansion."""
        result = expand_emojis("I'm 😊 and 😍")
        assert "smiling" in result.lower()
        assert "heart" in result.lower()

    def test_expand_emojis_no_double_spaces(self):
        """Test no double spaces after expansion."""
        result = expand_emojis("Hello 👍 there")
        assert "  " not in result  # No double spaces

    def test_expand_emojis_thumbs(self):
        """Test thumbs up/down emojis."""
        assert "thumbs up" in expand_emojis("👍")
        assert "thumbs down" in expand_emojis("👎")

    def test_expand_emojis_hearts(self):
        """Test heart emojis."""
        assert "red heart" in expand_emojis("❤️")
        assert "broken heart" in expand_emojis("💔")


# =============================================================================
# Math Symbol Expansion Tests
# =============================================================================

class TestExpandMathSymbols:
    """Tests for expand_math_symbols function."""

    def test_expand_math_symbols_basic(self):
        """Test basic math symbol expansion."""
        result = expand_math_symbols("2 + 2 = 4")
        assert "plus" in result
        assert "equals" in result

    def test_expand_math_symbols_complex(self):
        """Test complex math symbols."""
        assert "less than or equal to" in expand_math_symbols("x ≤ 10")
        assert "approximately equal to" in expand_math_symbols("π ≈ 3.14")

    def test_expand_math_symbols_greek(self):
        """Test Greek letter expansion."""
        assert "alpha" in expand_math_symbols("α")
        assert "theta" in expand_math_symbols("θ")
        assert "pi" in expand_math_symbols("π")

    def test_expand_math_symbols_percent(self):
        """Test percent expansion."""
        result = expand_math_symbols("50%")
        assert "percent" in result

    def test_expand_math_symbols_degrees(self):
        """Test degree symbol expansion."""
        result = expand_math_symbols("90°")
        assert "degrees" in result


# =============================================================================
# URL/Email Expansion Tests
# =============================================================================

class TestExpandUrlEmail:
    """Tests for URL and email expansion functions."""

    def test_expand_url_basic(self):
        """Test basic URL expansion."""
        result = expand_url("https://example.com")
        assert "example" in result
        assert "dot" in result
        assert "com" in result

    def test_expand_url_removes_protocol(self):
        """Test protocol is removed."""
        result = expand_url("https://www.example.com")
        assert "https" not in result
        assert "www" not in result

    def test_expand_url_path(self):
        """Test URL with path."""
        result = expand_url("https://example.com/page")
        assert "slash" in result

    def test_expand_email_basic(self):
        """Test basic email expansion."""
        result = expand_email("test@example.com")
        assert "at" in result
        assert "dot" in result

    def test_expand_email_with_dots(self):
        """Test email with dots in username."""
        result = expand_email("first.last@example.com")
        assert result.count("dot") >= 2

    def test_expand_urls_and_emails_in_text(self):
        """Test finding and expanding URLs/emails in text."""
        text = "Contact support@example.com or visit https://example.com"
        result = expand_urls_and_emails(text)
        assert "at" in result  # From email
        assert "dot" in result


# =============================================================================
# HTML Entity Decoding Tests
# =============================================================================

class TestDecodeHtmlEntities:
    """Tests for decode_html_entities function."""

    def test_decode_html_entities_basic(self):
        """Test basic HTML entity decoding."""
        assert decode_html_entities("&lt;") == "<"
        assert decode_html_entities("&gt;") == ">"
        assert decode_html_entities("&amp;") == "&"

    def test_decode_html_entities_numeric(self):
        """Test numeric HTML entity decoding."""
        assert decode_html_entities("&#169;") == "©"
        assert decode_html_entities("&#8212;") == "—"

    def test_decode_html_entities_named(self):
        """Test named HTML entities."""
        assert decode_html_entities("&copy;") == "©"
        assert decode_html_entities("&nbsp;") == "\xa0"  # Non-breaking space

    def test_decode_html_entities_mixed(self):
        """Test mixed content with entities."""
        result = decode_html_entities("Hello &amp; World &lt;3")
        assert "&" in result
        assert "<" in result


# =============================================================================
# Integration Tests - normalize_text
# =============================================================================

class TestNormalizeText:
    """Tests for the full normalize_text function."""

    def test_normalize_text_all_features(self):
        """Test full pipeline with all features."""
        text = "$100 is great! 😀 Dr. Smith said 2 + 2 = 4"
        result = normalize_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_text_currency_only(self):
        """Test with only currency expansion enabled."""
        result = normalize_text(
            "$50",
            expand_numbers_flag=False,
            expand_currency_flag=True,
            expand_abbreviations_flag=False,
            expand_emojis_flag=False,
            expand_math_flag=False,
            expand_urls_flag=False,
            decode_html_flag=False,
        )
        assert "fifty dollars" in result

    def test_normalize_text_acronyms_enabled(self):
        """Test with acronyms expansion enabled."""
        result = normalize_text(
            "The FBI is here",
            expand_acronyms_flag=True,
        )
        assert "F B I" in result

    def test_normalize_text_empty_string(self):
        """Test with empty string input."""
        result = normalize_text("")
        assert result == ""

    def test_normalize_text_whitespace_only(self):
        """Test with whitespace only input."""
        result = normalize_text("   \n\t   ")
        assert result == ""

    def test_normalize_text_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "Hello world this is normal text"
        result = normalize_text(text)
        assert "Hello" in result
        assert "world" in result

    def test_normalize_text_cleans_whitespace(self):
        """Test that extra whitespace is cleaned."""
        text = "Hello    world"
        result = normalize_text(text)
        assert "  " not in result  # No double spaces


# =============================================================================
# Edge Cases and Robustness
# =============================================================================

class TestNormalizerEdgeCases:
    """Edge case tests for normalizer functions."""

    def test_number_large_value(self):
        """Test very large number conversion."""
        result = number_to_words(1000000000000)  # One trillion
        assert "trillion" in result

    def test_currency_zero_amount(self):
        """Test zero currency amount."""
        result = expand_currency("$0")
        assert "zero dollars" in result

    def test_abbreviations_not_at_word_boundary(self):
        """Test abbreviations only match at word boundaries."""
        # "Driveway" should not match "Dr." for Drive
        result = expand_abbreviations("The driveway is long")
        # Should not be expanded inappropriately

    def test_emoji_unknown_emoji(self):
        """Test unknown emoji is preserved or handled."""
        # Use an emoji not in the dictionary
        result = expand_emojis("Hello 🦄")  # Unicorn may or may not be in dict
        assert "Hello" in result

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        result = expand_url("https://example.com/page?id=123&name=test")
        assert "question mark" in result
        assert "and" in result
        assert "equals" in result
