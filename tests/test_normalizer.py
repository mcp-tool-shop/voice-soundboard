"""Tests for text normalization module."""

import pytest
from voice_soundboard.normalizer import (
    number_to_words,
    decimal_to_words,
    expand_currency,
    expand_abbreviations,
    expand_acronyms,
    expand_emojis,
    expand_math_symbols,
    decode_html_entities,
    expand_url,
    expand_email,
    expand_urls_and_emails,
    normalize_text,
    ABBREVIATIONS,
    CURRENCY_SYMBOLS,
    MATH_SYMBOLS,
    EMOJI_NAMES,
)


class TestNumberToWords:
    """Tests for number_to_words function."""

    def test_zero(self):
        assert number_to_words(0) == "zero"

    def test_single_digits(self):
        assert number_to_words(1) == "one"
        assert number_to_words(5) == "five"
        assert number_to_words(9) == "nine"

    def test_teens(self):
        assert number_to_words(10) == "ten"
        assert number_to_words(11) == "eleven"
        assert number_to_words(13) == "thirteen"
        assert number_to_words(19) == "nineteen"

    def test_tens(self):
        assert number_to_words(20) == "twenty"
        assert number_to_words(25) == "twenty-five"
        assert number_to_words(42) == "forty-two"
        assert number_to_words(99) == "ninety-nine"

    def test_hundreds(self):
        assert number_to_words(100) == "one hundred"
        assert number_to_words(123) == "one hundred twenty-three"
        assert number_to_words(500) == "five hundred"
        assert number_to_words(999) == "nine hundred ninety-nine"

    def test_thousands(self):
        assert number_to_words(1000) == "one thousand"
        assert number_to_words(1234) == "one thousand two hundred thirty-four"
        assert number_to_words(10000) == "ten thousand"
        assert number_to_words(99999) == "ninety-nine thousand nine hundred ninety-nine"

    def test_millions(self):
        assert number_to_words(1000000) == "one million"
        assert number_to_words(1234567) == "one million two hundred thirty-four thousand five hundred sixty-seven"

    def test_billions(self):
        assert number_to_words(1000000000) == "one billion"
        assert "billion" in number_to_words(5000000000)

    def test_trillions(self):
        assert number_to_words(1000000000000) == "one trillion"

    def test_negative_numbers(self):
        assert number_to_words(-5) == "negative five"
        assert number_to_words(-100) == "negative one hundred"
        assert number_to_words(-1234) == "negative one thousand two hundred thirty-four"


class TestDecimalToWords:
    """Tests for decimal_to_words function."""

    def test_simple_decimal(self):
        assert decimal_to_words("3.14") == "three point one four"

    def test_zero_decimal(self):
        assert decimal_to_words("0.5") == "zero point five"

    def test_large_decimal(self):
        result = decimal_to_words("123.456")
        assert "one hundred twenty-three point" in result
        assert "four" in result
        assert "five" in result
        assert "six" in result

    def test_integer_string(self):
        assert decimal_to_words("42") == "forty-two"

    def test_zero_in_decimal(self):
        assert decimal_to_words("1.01") == "one point zero one"


class TestExpandCurrency:
    """Tests for expand_currency function."""

    def test_dollars(self):
        assert expand_currency("$100") == "one hundred dollars"
        assert expand_currency("$1") == "one dollar"

    def test_dollars_with_cents(self):
        result = expand_currency("$19.99")
        assert "nineteen dollars" in result
        assert "ninety-nine cents" in result

    def test_euros(self):
        assert expand_currency("€50") == "fifty euros"
        assert expand_currency("€1") == "one euro"

    def test_pounds(self):
        assert expand_currency("£100") == "one hundred pounds"

    def test_yen(self):
        assert expand_currency("¥1000") == "one thousand yen"

    def test_currency_with_commas(self):
        assert expand_currency("$1,000") == "one thousand dollars"
        assert expand_currency("$1,000,000") == "one million dollars"

    def test_mixed_text_with_currency(self):
        result = expand_currency("The price is $50 for the item.")
        assert "fifty dollars" in result
        assert "The price is" in result

    def test_multiple_currencies(self):
        result = expand_currency("$100 and €50")
        assert "one hundred dollars" in result
        assert "fifty euros" in result


class TestExpandAbbreviations:
    """Tests for expand_abbreviations function."""

    def test_titles(self):
        assert "Doctor" in expand_abbreviations("Dr. Smith")
        assert "Mister" in expand_abbreviations("Mr. Jones")
        assert "Missus" in expand_abbreviations("Mrs. Johnson")
        assert "Professor" in expand_abbreviations("Prof. Williams")

    def test_addresses(self):
        assert "Street" in expand_abbreviations("123 Main St.")
        assert "Avenue" in expand_abbreviations("5th Ave.")
        assert "Boulevard" in expand_abbreviations("Sunset Blvd.")

    def test_units(self):
        assert "feet" in expand_abbreviations("10 ft.")
        assert "pounds" in expand_abbreviations("5 lbs.")
        assert "kilometers" in expand_abbreviations("100 km.")

    def test_common_abbrevs(self):
        assert "versus" in expand_abbreviations("Team A vs. Team B")
        assert "etcetera" in expand_abbreviations("apples, oranges, etc.")
        assert "for example" in expand_abbreviations("Fruits, e.g. apples")

    def test_case_insensitive(self):
        assert "Doctor" in expand_abbreviations("dr. smith")
        assert "Doctor" in expand_abbreviations("DR. SMITH")


class TestExpandAcronyms:
    """Tests for expand_acronyms function."""

    def test_common_acronyms(self):
        assert expand_acronyms("FBI") == "F B I"
        assert expand_acronyms("CIA") == "C I A"
        assert expand_acronyms("NASA") == "N A S A"

    def test_tech_acronyms(self):
        assert expand_acronyms("API") == "A P I"
        assert expand_acronyms("HTML") == "H T M L"
        assert expand_acronyms("URL") == "U R L"

    def test_acronym_in_sentence(self):
        result = expand_acronyms("The FBI investigated the case.")
        assert "F B I" in result
        assert "The" in result

    def test_multiple_acronyms(self):
        result = expand_acronyms("Use the API and URL")
        assert "A P I" in result
        assert "U R L" in result

    def test_case_insensitive(self):
        assert expand_acronyms("fbi") == "F B I"
        assert expand_acronyms("Fbi") == "F B I"


class TestExpandEmojis:
    """Tests for expand_emojis function."""

    def test_single_emoji(self):
        assert "grinning face" in expand_emojis("😀")
        assert "thumbs up" in expand_emojis("👍")
        assert "fire" in expand_emojis("🔥")

    def test_emoji_in_text(self):
        result = expand_emojis("Great job! 👍")
        assert "thumbs up" in result
        assert "Great job!" in result

    def test_multiple_emojis(self):
        result = expand_emojis("😀😍🎉")
        assert "grinning face" in result
        assert "heart eyes" in result
        assert "party popper" in result

    def test_hearts(self):
        assert "red heart" in expand_emojis("❤️")
        assert "broken heart" in expand_emojis("💔")


class TestExpandMathSymbols:
    """Tests for expand_math_symbols function."""

    def test_basic_operators(self):
        assert "plus" in expand_math_symbols("2 + 2")
        assert "minus" in expand_math_symbols("5 - 3")
        assert "times" in expand_math_symbols("2 × 3")
        assert "divided by" in expand_math_symbols("10 ÷ 2")

    def test_comparisons(self):
        assert "equals" in expand_math_symbols("2 + 2 = 4")
        assert "less than" in expand_math_symbols("x < 5")
        assert "greater than" in expand_math_symbols("y > 10")

    def test_percent(self):
        assert "percent" in expand_math_symbols("50%")

    def test_degrees(self):
        assert "degrees" in expand_math_symbols("90°")

    def test_greek_letters(self):
        assert "pi" in expand_math_symbols("π")
        assert "alpha" in expand_math_symbols("α")
        assert "beta" in expand_math_symbols("β")


class TestDecodeHtmlEntities:
    """Tests for decode_html_entities function."""

    def test_basic_entities(self):
        assert decode_html_entities("&amp;") == "&"
        assert decode_html_entities("&lt;") == "<"
        assert decode_html_entities("&gt;") == ">"
        assert decode_html_entities("&quot;") == '"'

    def test_numeric_entities(self):
        assert decode_html_entities("&#60;") == "<"
        assert decode_html_entities("&#x3C;") == "<"

    def test_mixed_text(self):
        result = decode_html_entities("Tom &amp; Jerry")
        assert result == "Tom & Jerry"

    def test_special_chars(self):
        assert decode_html_entities("&copy;") == "©"
        assert decode_html_entities("&nbsp;") == "\xa0"  # Non-breaking space


class TestExpandUrl:
    """Tests for expand_url function."""

    def test_simple_url(self):
        result = expand_url("https://example.com")
        assert "example" in result
        assert "dot" in result
        assert "com" in result

    def test_url_with_path(self):
        result = expand_url("https://example.com/page")
        assert "slash" in result

    def test_url_with_www(self):
        result = expand_url("www.example.com")
        assert "example" in result
        # www should be removed

    def test_url_with_params(self):
        result = expand_url("https://example.com?id=123")
        assert "question mark" in result
        assert "equals" in result


class TestExpandEmail:
    """Tests for expand_email function."""

    def test_simple_email(self):
        result = expand_email("user@example.com")
        assert "at" in result
        assert "dot" in result
        assert "user" in result
        assert "example" in result

    def test_email_with_subdomain(self):
        result = expand_email("user@mail.example.com")
        assert result.count("dot") == 2

    def test_email_with_special_chars(self):
        result = expand_email("user-name@example.com")
        assert "dash" in result


class TestExpandUrlsAndEmails:
    """Tests for expand_urls_and_emails function."""

    def test_url_in_text(self):
        result = expand_urls_and_emails("Visit https://example.com for more info.")
        assert "dot" in result
        assert "Visit" in result

    def test_email_in_text(self):
        result = expand_urls_and_emails("Contact us at support@example.com")
        assert "at" in result
        assert "Contact us" in result

    def test_both_url_and_email(self):
        result = expand_urls_and_emails("Go to https://site.com or email test@site.com")
        assert result.count("at") >= 1
        assert "dot" in result


class TestNormalizeText:
    """Tests for the main normalize_text function."""

    def test_comprehensive_normalization(self):
        text = "Dr. Smith said the price is $100 and the temperature is 75°"
        result = normalize_text(text)
        assert "Doctor" in result
        assert "one hundred dollars" in result
        assert "degrees" in result

    def test_with_emojis(self):
        text = "Great work! 👍"
        result = normalize_text(text)
        assert "thumbs up" in result

    def test_with_html_entities(self):
        text = "Tom &amp; Jerry"
        result = normalize_text(text)
        assert "&" in result or "and" in result

    def test_disable_features(self):
        text = "$100"
        result = normalize_text(text, expand_currency_flag=False)
        assert result == "$100"

    def test_acronym_expansion_off_by_default(self):
        text = "The FBI is here"
        result = normalize_text(text, expand_acronyms_flag=False)
        assert "FBI" in result  # Should not be expanded

    def test_acronym_expansion_when_enabled(self):
        text = "The FBI is here"
        result = normalize_text(text, expand_acronyms_flag=True)
        assert "F B I" in result

    def test_whitespace_cleanup(self):
        text = "Hello   world    test"
        result = normalize_text(text)
        assert "  " not in result  # No double spaces


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   ") == ""

    def test_special_characters_only(self):
        # Should handle without crashing
        result = normalize_text("@#$%^&*()")
        assert isinstance(result, str)

    def test_unicode_text(self):
        result = normalize_text("Café résumé naïve")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_very_large_number(self):
        # Should handle gracefully
        result = number_to_words(999999999999999)
        assert isinstance(result, str)
        assert "trillion" in result or "quadrillion" in result

    def test_mixed_content(self):
        text = "Dr. Smith paid $99.99 for 👍 and said 2 + 2 = 4"
        result = normalize_text(text)
        assert "Doctor" in result
        assert "dollars" in result
        assert "thumbs up" in result
        assert "equals" in result

    def test_currency_in_middle_of_word(self):
        # Should not break when currency symbol is part of other text
        text = "non$tandardtext"
        result = expand_currency(text)
        assert isinstance(result, str)

    def test_no_match_text(self):
        # Text without any patterns to normalize
        text = "The quick brown fox jumps over the lazy dog."
        result = normalize_text(text)
        assert result == text  # Should remain unchanged


class TestDataIntegrity:
    """Tests to ensure data dictionaries are properly defined."""

    def test_abbreviations_not_empty(self):
        assert len(ABBREVIATIONS) > 50

    def test_currency_symbols_not_empty(self):
        assert len(CURRENCY_SYMBOLS) > 10

    def test_math_symbols_not_empty(self):
        assert len(MATH_SYMBOLS) > 20

    def test_emoji_names_not_empty(self):
        assert len(EMOJI_NAMES) > 100

    def test_all_abbreviations_have_expansions(self):
        for abbrev, expansion in ABBREVIATIONS.items():
            assert isinstance(abbrev, str)
            assert isinstance(expansion, str)
            assert len(expansion) > len(abbrev) - 1  # Expansion should be longer or equal

    def test_all_currencies_have_singular_plural(self):
        for symbol, (singular, plural) in CURRENCY_SYMBOLS.items():
            assert isinstance(symbol, str)
            assert isinstance(singular, str)
            assert isinstance(plural, str)
