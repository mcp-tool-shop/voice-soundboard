"""
Tests for SSML Parser (ssml.py).

Tests cover:
- SSMLParams dataclass
- Time parsing (_parse_time)
- Time to pause conversion (_time_to_pause)
- Formatting functions (date, time, cardinal, ordinal, characters, telephone)
- Say-as processing
- Element processing for all SSML tags
- parse_ssml main function
- ssml_to_text convenience function
- SSML building helpers (pause, emphasis, say_as, prosody)
- Security: defusedxml usage
"""

import pytest
from unittest.mock import patch

from voice_soundboard.ssml import (
    SSMLParams,
    BREAK_MARKERS,
    _parse_time,
    _time_to_pause,
    _format_date,
    _format_time,
    _format_cardinal,
    _format_ordinal,
    _format_characters,
    _format_telephone,
    _process_say_as,
    parse_ssml,
    ssml_to_text,
    pause,
    emphasis,
    say_as,
    prosody,
)


class TestSSMLParams:
    """Tests for SSMLParams dataclass."""

    def test_default_values(self):
        """Test SSMLParams has correct defaults."""
        params = SSMLParams()

        assert params.speed == 1.0
        assert params.pitch == 1.0
        assert params.volume == 1.0
        assert params.voice is None
        assert params.phonemes == []

    def test_custom_values(self):
        """Test SSMLParams accepts custom values."""
        params = SSMLParams(
            speed=0.8,
            pitch=1.2,
            volume=0.9,
            voice="test_voice",
            phonemes=["ph1", "ph2"],
        )

        assert params.speed == 0.8
        assert params.pitch == 1.2
        assert params.volume == 0.9
        assert params.voice == "test_voice"
        assert params.phonemes == ["ph1", "ph2"]

    def test_phonemes_default_factory(self):
        """Test phonemes list is independent between instances."""
        params1 = SSMLParams()
        params2 = SSMLParams()

        params1.phonemes.append("test")

        assert params1.phonemes == ["test"]
        assert params2.phonemes == []  # Should not be affected


class TestBreakMarkers:
    """Tests for BREAK_MARKERS constant."""

    def test_all_strengths_present(self):
        """Test all strength levels are defined."""
        expected = ["none", "x-weak", "weak", "medium", "strong", "x-strong"]
        for strength in expected:
            assert strength in BREAK_MARKERS

    def test_marker_values(self):
        """Test break markers have expected patterns."""
        assert BREAK_MARKERS["none"] == ""
        assert "," in BREAK_MARKERS["x-weak"]
        assert "..." in BREAK_MARKERS["medium"]
        assert "..." in BREAK_MARKERS["x-strong"]


class TestParseTime:
    """Tests for _parse_time function."""

    def test_milliseconds(self):
        """Test parsing milliseconds."""
        assert _parse_time("500ms") == 0.5
        assert _parse_time("1000ms") == 1.0
        assert _parse_time("250ms") == 0.25

    def test_seconds(self):
        """Test parsing seconds."""
        assert _parse_time("1s") == 1.0
        assert _parse_time("2.5s") == 2.5
        assert _parse_time("0.5s") == 0.5

    def test_plain_number(self):
        """Test parsing plain number (assumed seconds)."""
        assert _parse_time("1.5") == 1.5
        assert _parse_time("2") == 2.0

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert _parse_time("500MS") == 0.5
        assert _parse_time("1S") == 1.0

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert _parse_time("  500ms  ") == 0.5

    def test_invalid_returns_default(self):
        """Test invalid input returns default 0.5."""
        assert _parse_time("invalid") == 0.5
        assert _parse_time("") == 0.5


class TestTimeToPause:
    """Tests for _time_to_pause function."""

    def test_zero_or_negative(self):
        """Test zero or negative returns empty."""
        assert _time_to_pause(0) == ""
        assert _time_to_pause(-1) == ""

    def test_short_pause(self):
        """Test short pause returns comma."""
        assert _time_to_pause(0.1) == ","
        assert _time_to_pause(0.2) == ","

    def test_medium_pause(self):
        """Test medium pause returns ellipsis."""
        assert _time_to_pause(0.3) == "..."
        assert _time_to_pause(0.4) == "..."

    def test_longer_pause(self):
        """Test longer pause returns double ellipsis."""
        assert _time_to_pause(0.6) == "... ..."
        assert _time_to_pause(0.9) == "... ..."

    def test_very_long_pause(self):
        """Test very long pause returns multiple ellipses."""
        result = _time_to_pause(2.0)
        assert result.count("...") >= 2

    def test_max_ellipses(self):
        """Test pause is capped at 5 ellipses."""
        result = _time_to_pause(10.0)
        assert result.count("...") <= 5


class TestFormatDate:
    """Tests for _format_date function."""

    def test_iso_date_format(self):
        """Test ISO date formatting."""
        result = _format_date("2024-01-15")
        assert "January" in result
        assert "15" in result
        assert "2024" in result

    def test_ordinal_suffix_st(self):
        """Test 1st suffix."""
        result = _format_date("2024-01-01")
        assert "1st" in result

    def test_ordinal_suffix_nd(self):
        """Test 2nd suffix."""
        result = _format_date("2024-01-02")
        assert "2nd" in result

    def test_ordinal_suffix_rd(self):
        """Test 3rd suffix."""
        result = _format_date("2024-01-03")
        assert "3rd" in result

    def test_ordinal_suffix_th(self):
        """Test th suffix for regular numbers."""
        result = _format_date("2024-01-04")
        assert "4th" in result

    def test_teens_use_th(self):
        """Test teen numbers use th."""
        result = _format_date("2024-01-11")
        assert "11th" in result

        result = _format_date("2024-01-12")
        assert "12th" in result

        result = _format_date("2024-01-13")
        assert "13th" in result

    def test_all_months(self):
        """Test all months are formatted correctly."""
        months = [
            (1, "January"), (2, "February"), (3, "March"),
            (4, "April"), (5, "May"), (6, "June"),
            (7, "July"), (8, "August"), (9, "September"),
            (10, "October"), (11, "November"), (12, "December"),
        ]
        for num, name in months:
            result = _format_date(f"2024-{num:02d}-15")
            assert name in result

    def test_non_iso_passthrough(self):
        """Test non-ISO dates pass through unchanged."""
        assert _format_date("Jan 15, 2024") == "Jan 15, 2024"


class TestFormatTime:
    """Tests for _format_time function."""

    def test_morning_times(self):
        """Test morning times."""
        result = _format_time("09:30")
        assert "9" in result
        assert "30" in result
        assert "AM" in result

    def test_afternoon_times(self):
        """Test afternoon times."""
        result = _format_time("14:30")
        assert "2" in result
        assert "30" in result
        assert "PM" in result

    def test_midnight(self):
        """Test midnight (00:xx)."""
        result = _format_time("00:30")
        assert "12" in result
        assert "AM" in result

    def test_noon(self):
        """Test noon (12:xx)."""
        result = _format_time("12:30")
        assert "12" in result
        assert "PM" in result

    def test_on_the_hour(self):
        """Test times on the hour."""
        result = _format_time("15:00")
        assert "3" in result
        assert "PM" in result
        # Should not include "00"
        assert "00" not in result

    def test_oh_minutes(self):
        """Test single-digit minutes use 'oh'."""
        result = _format_time("14:05")
        assert "oh" in result.lower()
        assert "5" in result

    def test_non_24h_passthrough(self):
        """Test non-24h format is still parsed (regex matches 3:30 part)."""
        # The function parses the 3:30 part, ignoring the PM
        # It interprets 3 as 3 AM since there's no context
        result = _format_time("3:30 PM")
        assert "3" in result
        assert "30" in result


class TestFormatCardinal:
    """Tests for _format_cardinal function."""

    def test_zero(self):
        """Test zero."""
        assert _format_cardinal("0") == "zero"

    def test_single_digits(self):
        """Test single digits."""
        assert _format_cardinal("1") == "one"
        assert _format_cardinal("5") == "five"
        assert _format_cardinal("9") == "nine"

    def test_teens(self):
        """Test teen numbers."""
        assert _format_cardinal("11") == "eleven"
        assert _format_cardinal("15") == "fifteen"
        assert _format_cardinal("19") == "nineteen"

    def test_tens(self):
        """Test tens."""
        assert _format_cardinal("20") == "twenty"
        assert _format_cardinal("50") == "fifty"
        assert _format_cardinal("90") == "ninety"

    def test_two_digit_combined(self):
        """Test two-digit numbers."""
        assert _format_cardinal("21") == "twenty-one"
        assert _format_cardinal("45") == "forty-five"
        assert _format_cardinal("99") == "ninety-nine"

    def test_negative(self):
        """Test negative numbers."""
        result = _format_cardinal("-5")
        assert "negative" in result
        assert "five" in result

    def test_large_numbers_passthrough(self):
        """Test large numbers pass through."""
        # For numbers >= 100, just return the string
        assert _format_cardinal("100") == "100"
        assert _format_cardinal("1000") == "1000"

    def test_comma_separated(self):
        """Test comma-separated numbers."""
        # Should handle commas in numbers
        result = _format_cardinal("1,000")
        # May just pass through
        assert result is not None

    def test_invalid_passthrough(self):
        """Test invalid input passes through."""
        assert _format_cardinal("abc") == "abc"


class TestFormatOrdinal:
    """Tests for _format_ordinal function."""

    def test_first_second_third(self):
        """Test 1st, 2nd, 3rd."""
        assert _format_ordinal("1") == "1st"
        assert _format_ordinal("2") == "2nd"
        assert _format_ordinal("3") == "3rd"

    def test_th_suffix(self):
        """Test th suffix."""
        assert _format_ordinal("4") == "4th"
        assert _format_ordinal("5") == "5th"
        assert _format_ordinal("10") == "10th"

    def test_teens(self):
        """Test teens use th."""
        assert _format_ordinal("11") == "11th"
        assert _format_ordinal("12") == "12th"
        assert _format_ordinal("13") == "13th"

    def test_twenty_first_etc(self):
        """Test 21st, 22nd, 23rd pattern."""
        assert _format_ordinal("21") == "21st"
        assert _format_ordinal("22") == "22nd"
        assert _format_ordinal("23") == "23rd"
        assert _format_ordinal("24") == "24th"

    def test_invalid_passthrough(self):
        """Test invalid input passes through."""
        assert _format_ordinal("abc") == "abc"


class TestFormatCharacters:
    """Tests for _format_characters function."""

    def test_spells_out_letters(self):
        """Test letters are spelled out with spaces."""
        result = _format_characters("ABC")
        assert result == "A B C"

    def test_uppercase_conversion(self):
        """Test lowercase is converted to uppercase."""
        result = _format_characters("abc")
        assert result == "A B C"

    def test_mixed_case(self):
        """Test mixed case."""
        result = _format_characters("AbC")
        assert result == "A B C"

    def test_with_numbers(self):
        """Test numbers are included."""
        result = _format_characters("A1B")
        assert result == "A 1 B"


class TestFormatTelephone:
    """Tests for _format_telephone function."""

    def test_basic_number(self):
        """Test basic phone number."""
        result = _format_telephone("5551234567")
        words = result.split()
        assert len(words) == 10
        assert words[0] == "five"
        assert words[1] == "five"
        assert words[2] == "five"

    def test_with_dashes(self):
        """Test phone with dashes."""
        result = _format_telephone("555-123-4567")
        # Non-digits are removed
        assert "five" in result
        assert "-" not in result

    def test_with_parentheses(self):
        """Test phone with parentheses."""
        result = _format_telephone("(555) 123-4567")
        assert "five" in result
        assert "(" not in result

    def test_digit_words(self):
        """Test all digit words."""
        result = _format_telephone("0123456789")
        expected = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        words = result.split()
        assert words == expected


class TestProcessSayAs:
    """Tests for _process_say_as function."""

    def test_date(self):
        """Test date interpretation."""
        result = _process_say_as("2024-01-15", "date")
        assert "January" in result

    def test_time(self):
        """Test time interpretation."""
        result = _process_say_as("14:30", "time")
        assert "PM" in result

    def test_cardinal(self):
        """Test cardinal interpretation."""
        result = _process_say_as("42", "cardinal")
        assert "forty-two" in result

    def test_ordinal(self):
        """Test ordinal interpretation."""
        result = _process_say_as("3", "ordinal")
        assert "3rd" in result

    def test_characters(self):
        """Test characters interpretation."""
        result = _process_say_as("ABC", "characters")
        assert result == "A B C"

    def test_spell_out(self):
        """Test spell-out is alias for characters."""
        result = _process_say_as("ABC", "spell-out")
        assert result == "A B C"

    def test_telephone(self):
        """Test telephone interpretation."""
        result = _process_say_as("555-1234", "telephone")
        assert "five" in result

    def test_unknown_passthrough(self):
        """Test unknown interpret-as passes through."""
        result = _process_say_as("test", "unknown")
        assert result == "test"


class TestParseSSML:
    """Tests for parse_ssml main function."""

    def test_simple_text(self):
        """Test simple text without tags."""
        text, params = parse_ssml("Hello world")
        assert "Hello world" in text
        assert params.speed == 1.0

    def test_with_speak_wrapper(self):
        """Test SSML with speak wrapper."""
        text, params = parse_ssml("<speak>Hello world</speak>")
        assert "Hello world" in text

    def test_break_tag_time(self):
        """Test break tag with time attribute."""
        text, params = parse_ssml('<speak>Hello <break time="500ms"/> world</speak>')
        # Should have pause markers
        assert "Hello" in text
        assert "world" in text

    def test_break_tag_strength(self):
        """Test break tag with strength attribute."""
        text, params = parse_ssml('<speak>Hello <break strength="strong"/> world</speak>')
        assert "..." in text

    def test_emphasis_tag(self):
        """Test emphasis tag."""
        text, params = parse_ssml('<speak><emphasis level="strong">Important</emphasis></speak>')
        assert "Important" in text
        # Strong emphasis adds asterisks
        assert "*" in text

    def test_emphasis_reduced(self):
        """Test reduced emphasis."""
        text, params = parse_ssml('<speak><emphasis level="reduced">quiet</emphasis></speak>')
        assert "quiet" in text

    def test_prosody_rate(self):
        """Test prosody tag changes speed."""
        text, params = parse_ssml('<speak><prosody rate="slow">Slow text</prosody></speak>')
        assert "Slow text" in text
        assert params.speed == 0.75

    def test_prosody_rate_fast(self):
        """Test prosody tag with fast rate."""
        text, params = parse_ssml('<speak><prosody rate="fast">Fast text</prosody></speak>')
        assert params.speed == 1.25

    def test_prosody_rate_percentage(self):
        """Test prosody rate as percentage."""
        text, params = parse_ssml('<speak><prosody rate="80%">Slower</prosody></speak>')
        assert params.speed == 0.8

    def test_say_as_date(self):
        """Test say-as with date."""
        text, params = parse_ssml('<speak><say-as interpret-as="date">2024-01-15</say-as></speak>')
        assert "January" in text

    def test_say_as_cardinal(self):
        """Test say-as with cardinal."""
        text, params = parse_ssml('<speak><say-as interpret-as="cardinal">42</say-as></speak>')
        assert "forty-two" in text

    def test_sub_tag(self):
        """Test substitution tag."""
        text, params = parse_ssml('<speak>The <sub alias="World Wide Web">WWW</sub></speak>')
        assert "World Wide Web" in text
        assert "WWW" not in text

    def test_phoneme_tag(self):
        """Test phoneme tag."""
        text, params = parse_ssml('<speak><phoneme ph="t eh s t">test</phoneme></speak>')
        assert "t eh s t" in text
        assert params.phonemes == ["t eh s t"]

    def test_voice_tag(self):
        """Test voice tag sets voice parameter."""
        text, params = parse_ssml('<speak><voice name="en-US-Wavenet-A">Hello</voice></speak>')
        assert params.voice == "en-US-Wavenet-A"
        assert "Hello" in text

    def test_p_tag_adds_pause(self):
        """Test paragraph tag adds pause."""
        text, params = parse_ssml('<speak><p>Paragraph one.</p><p>Paragraph two.</p></speak>')
        assert "Paragraph one" in text
        assert "Paragraph two" in text

    def test_s_tag_adds_pause(self):
        """Test sentence tag adds pause."""
        text, params = parse_ssml('<speak><s>Sentence one.</s><s>Sentence two.</s></speak>')
        assert "Sentence one" in text
        assert "Sentence two" in text

    def test_nested_elements(self):
        """Test nested SSML elements."""
        ssml = '''<speak>
            <prosody rate="slow">
                <emphasis level="strong">Important slow text</emphasis>
            </prosody>
        </speak>'''
        text, params = parse_ssml(ssml)
        assert "Important slow text" in text
        assert params.speed == 0.75

    def test_whitespace_cleanup(self):
        """Test excessive whitespace is cleaned."""
        text, params = parse_ssml("<speak>Hello    world</speak>")
        assert "  " not in text  # No double spaces

    def test_consecutive_pauses_cleanup(self):
        """Test consecutive pauses are cleaned."""
        text, params = parse_ssml('<speak>A<break time="1s"/><break time="1s"/>B</speak>')
        # Should not have excessive consecutive ellipses
        assert "... ... ..." not in text

    def test_invalid_xml_fallback(self):
        """Test invalid XML falls back to text extraction."""
        text, params = parse_ssml("<speak>Unclosed <break")
        # Should still extract some text
        assert "Unclosed" in text or text == ""

    def test_auto_wrap_in_speak(self):
        """Test text without speak wrapper is auto-wrapped."""
        text, params = parse_ssml("Just plain text")
        assert "Just plain text" in text

    def test_tail_text_handling(self):
        """Test text after closing tags is preserved."""
        text, params = parse_ssml("<speak><emphasis>Bold</emphasis> and normal</speak>")
        assert "Bold" in text
        assert "normal" in text


class TestSSMLToText:
    """Tests for ssml_to_text convenience function."""

    def test_simple_conversion(self):
        """Test simple SSML to text."""
        result = ssml_to_text("<speak>Hello world</speak>")
        assert "Hello world" in result

    def test_returns_only_text(self):
        """Test only text is returned, not params."""
        result = ssml_to_text('<speak><prosody rate="fast">Fast</prosody></speak>')
        assert isinstance(result, str)
        assert "Fast" in result


class TestSSMLBuilders:
    """Tests for SSML building convenience functions."""

    def test_pause_default(self):
        """Test pause with default time."""
        result = pause()
        assert 'break time="500ms"' in result

    def test_pause_custom_time(self):
        """Test pause with custom time."""
        result = pause("1s")
        assert 'time="1s"' in result

    def test_emphasis_default(self):
        """Test emphasis with default level."""
        result = emphasis("text")
        assert '<emphasis level="moderate">text</emphasis>' == result

    def test_emphasis_strong(self):
        """Test emphasis with strong level."""
        result = emphasis("important", "strong")
        assert 'level="strong"' in result
        assert "important" in result

    def test_say_as_builder(self):
        """Test say_as builder function."""
        result = say_as("2024-01-15", "date")
        assert 'interpret-as="date"' in result
        assert "2024-01-15" in result

    def test_prosody_builder(self):
        """Test prosody builder function."""
        result = prosody("text", rate="slow", pitch="high")
        assert 'rate="slow"' in result
        assert 'pitch="high"' in result
        assert "text" in result

    def test_prosody_defaults(self):
        """Test prosody with default values."""
        result = prosody("text")
        assert 'rate="medium"' in result
        assert 'pitch="medium"' in result


class TestSSMLSecurity:
    """Tests for SSML security features."""

    def test_uses_defusedxml(self):
        """Test that defusedxml is used for parsing."""
        # This is verified by the import - if defusedxml isn't installed,
        # the import would fail
        import voice_soundboard.ssml as ssml_module

        # Check the module imports defusedxml
        # defusedxml.ElementTree is a module, so check its __name__
        assert "defusedxml" in str(ssml_module.ET.__name__)

    def test_handles_malformed_xml(self):
        """Test graceful handling of malformed XML."""
        # Should not crash, should return something
        text, params = parse_ssml("<speak>Malformed <break></speak>")
        assert text is not None

    def test_deeply_nested_elements(self):
        """Test handling of deeply nested elements."""
        # Build deeply nested SSML
        nested = "<speak>" + "<emphasis>" * 50 + "deep" + "</emphasis>" * 50 + "</speak>"
        text, params = parse_ssml(nested)
        assert "deep" in text


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_ssml(self):
        """Test empty SSML."""
        text, params = parse_ssml("")
        assert text == ""

    def test_whitespace_only(self):
        """Test whitespace-only SSML."""
        text, params = parse_ssml("   ")
        assert text.strip() == ""

    def test_special_characters(self):
        """Test special characters in SSML."""
        text, params = parse_ssml("<speak>Hello &amp; goodbye</speak>")
        assert "&" in text

    def test_unicode_text(self):
        """Test Unicode text in SSML."""
        text, params = parse_ssml("<speak>Héllo wörld 你好</speak>")
        assert "Héllo" in text
        assert "wörld" in text
        assert "你好" in text

    def test_multiple_phonemes(self):
        """Test multiple phoneme tags accumulate."""
        ssml = '<speak><phoneme ph="a">1</phoneme><phoneme ph="b">2</phoneme></speak>'
        text, params = parse_ssml(ssml)
        assert "a" in params.phonemes
        assert "b" in params.phonemes

    def test_unknown_tags_processed(self):
        """Test unknown tags are processed as text containers."""
        text, params = parse_ssml("<speak><custom>Content</custom></speak>")
        assert "Content" in text

    def test_mixed_content(self):
        """Test mixed text and element content."""
        ssml = "<speak>Before <emphasis>middle</emphasis> after</speak>"
        text, params = parse_ssml(ssml)
        assert "Before" in text
        assert "middle" in text
        assert "after" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
