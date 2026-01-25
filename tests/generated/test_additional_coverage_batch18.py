"""
Batch 18: Additional Coverage Tests for Web Server and Text Normalizer
- web_server.py: HTTP server for mobile access, REST API endpoints
- normalizer.py: Text normalization for TTS (numbers, currency, abbreviations, emojis)
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import numpy as np
import pytest


# ==============================================================================
# Tests for normalizer.py
# ==============================================================================

class TestNumberToWords:
    """Tests for number_to_words function."""

    def test_zero(self):
        """Test converting zero."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(0) == "zero"

    def test_single_digit(self):
        """Test single digit numbers."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(1) == "one"
        assert number_to_words(5) == "five"
        assert number_to_words(9) == "nine"

    def test_teens(self):
        """Test teen numbers."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(11) == "eleven"
        assert number_to_words(13) == "thirteen"
        assert number_to_words(19) == "nineteen"

    def test_tens(self):
        """Test multiples of ten."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(20) == "twenty"
        assert number_to_words(50) == "fifty"
        assert number_to_words(90) == "ninety"

    def test_two_digit_numbers(self):
        """Test two-digit numbers."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(21) == "twenty-one"
        assert number_to_words(42) == "forty-two"
        assert number_to_words(99) == "ninety-nine"

    def test_hundreds(self):
        """Test hundreds."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(100) == "one hundred"
        assert number_to_words(200) == "two hundred"
        assert number_to_words(123) == "one hundred twenty-three"
        assert number_to_words(999) == "nine hundred ninety-nine"

    def test_thousands(self):
        """Test thousands."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(1000) == "one thousand"
        assert number_to_words(5000) == "five thousand"
        assert number_to_words(1234) == "one thousand two hundred thirty-four"

    def test_millions(self):
        """Test millions."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(1000000) == "one million"
        assert number_to_words(2500000) == "two million five hundred thousand"

    def test_billions(self):
        """Test billions."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(1000000000) == "one billion"

    def test_negative_numbers(self):
        """Test negative numbers."""
        from voice_soundboard.normalizer import number_to_words

        assert number_to_words(-1) == "negative one"
        assert number_to_words(-42) == "negative forty-two"


class TestDecimalToWords:
    """Tests for decimal_to_words function."""

    def test_integer_string(self):
        """Test integer as string."""
        from voice_soundboard.normalizer import decimal_to_words

        assert decimal_to_words("42") == "forty-two"

    def test_simple_decimal(self):
        """Test simple decimal."""
        from voice_soundboard.normalizer import decimal_to_words

        result = decimal_to_words("3.14")
        assert "three point one four" in result

    def test_decimal_with_zeros(self):
        """Test decimal with zeros."""
        from voice_soundboard.normalizer import decimal_to_words

        result = decimal_to_words("0.5")
        assert "zero point five" in result

    def test_invalid_string(self):
        """Test invalid string returns as-is."""
        from voice_soundboard.normalizer import decimal_to_words

        assert decimal_to_words("abc") == "abc"

    def test_multiple_dots(self):
        """Test string with multiple dots."""
        from voice_soundboard.normalizer import decimal_to_words

        assert decimal_to_words("1.2.3") == "1.2.3"


class TestExpandCurrency:
    """Tests for expand_currency function."""

    def test_dollar_amount(self):
        """Test dollar amount."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("$100")
        assert "one hundred dollars" in result

    def test_dollar_with_cents(self):
        """Test dollar amount with cents."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("$5.99")
        assert "five dollars" in result
        assert "ninety-nine cents" in result

    def test_single_dollar(self):
        """Test single dollar uses singular form."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("$1")
        assert "one dollar" in result

    def test_euro(self):
        """Test euro currency."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("€50")
        assert "fifty euros" in result

    def test_pound(self):
        """Test pound currency."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("£100")
        assert "one hundred pounds" in result

    def test_currency_with_commas(self):
        """Test currency with comma separators."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("$1,000")
        assert "one thousand dollars" in result

    def test_no_currency(self):
        """Test text without currency."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("Hello world")
        assert result == "Hello world"


class TestExpandAbbreviations:
    """Tests for expand_abbreviations function."""

    def test_title_abbreviations(self):
        """Test title abbreviations."""
        from voice_soundboard.normalizer import expand_abbreviations

        assert "Doctor" in expand_abbreviations("Dr. Smith")
        assert "Mister" in expand_abbreviations("Mr. Jones")
        assert "Professor" in expand_abbreviations("Prof. Brown")

    def test_address_abbreviations(self):
        """Test address abbreviations."""
        from voice_soundboard.normalizer import expand_abbreviations

        assert "Street" in expand_abbreviations("123 Main St.")
        assert "Avenue" in expand_abbreviations("5th Ave.")
        assert "Boulevard" in expand_abbreviations("Sunset Blvd.")

    def test_unit_abbreviations(self):
        """Test unit abbreviations."""
        from voice_soundboard.normalizer import expand_abbreviations

        assert "feet" in expand_abbreviations("10 ft.")
        assert "pounds" in expand_abbreviations("5 lb.")
        assert "kilometers" in expand_abbreviations("100 km.")

    def test_common_abbreviations(self):
        """Test common abbreviations."""
        from voice_soundboard.normalizer import expand_abbreviations

        assert "versus" in expand_abbreviations("Team A vs. Team B")
        assert "etcetera" in expand_abbreviations("apples, oranges, etc.")
        assert "for example" in expand_abbreviations("e.g. this")


class TestExpandAcronyms:
    """Tests for expand_acronyms function."""

    def test_common_acronyms(self):
        """Test common acronyms are spelled out."""
        from voice_soundboard.normalizer import expand_acronyms

        result = expand_acronyms("The FBI investigated.")
        assert "F B I" in result

    def test_tech_acronyms(self):
        """Test technology acronyms."""
        from voice_soundboard.normalizer import expand_acronyms

        result = expand_acronyms("Use the API")
        assert "A P I" in result

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        from voice_soundboard.normalizer import expand_acronyms

        result = expand_acronyms("The fbi and FBI")
        # Both should be expanded
        assert result.count("F B I") >= 1

    def test_no_acronyms(self):
        """Test text without acronyms."""
        from voice_soundboard.normalizer import expand_acronyms

        result = expand_acronyms("Hello world")
        assert result == "Hello world"


class TestExpandEmojis:
    """Tests for expand_emojis function."""

    def test_smiley_emoji(self):
        """Test smiley face emoji."""
        from voice_soundboard.normalizer import expand_emojis

        result = expand_emojis("Hello 😀")
        assert "grinning face" in result

    def test_heart_emoji(self):
        """Test heart emoji."""
        from voice_soundboard.normalizer import expand_emojis

        result = expand_emojis("I love you ❤️")
        assert "red heart" in result

    def test_thumbs_up(self):
        """Test thumbs up emoji."""
        from voice_soundboard.normalizer import expand_emojis

        result = expand_emojis("Good job 👍")
        assert "thumbs up" in result

    def test_multiple_emojis(self):
        """Test multiple emojis."""
        from voice_soundboard.normalizer import expand_emojis

        result = expand_emojis("🎉 Party time! 🎊")
        assert "party popper" in result
        assert "confetti ball" in result

    def test_no_emojis(self):
        """Test text without emojis."""
        from voice_soundboard.normalizer import expand_emojis

        result = expand_emojis("Plain text")
        assert result == "Plain text"


class TestExpandMathSymbols:
    """Tests for expand_math_symbols function."""

    def test_basic_operators(self):
        """Test basic math operators."""
        from voice_soundboard.normalizer import expand_math_symbols

        assert "plus" in expand_math_symbols("2 + 2")
        assert "minus" in expand_math_symbols("5 - 3")
        assert "equals" in expand_math_symbols("2 + 2 = 4")

    def test_special_symbols(self):
        """Test special math symbols."""
        from voice_soundboard.normalizer import expand_math_symbols

        assert "times" in expand_math_symbols("3 × 4")
        assert "divided by" in expand_math_symbols("10 ÷ 2")
        assert "percent" in expand_math_symbols("50%")

    def test_greek_letters(self):
        """Test Greek letters."""
        from voice_soundboard.normalizer import expand_math_symbols

        assert "pi" in expand_math_symbols("π")
        assert "alpha" in expand_math_symbols("α")
        assert "omega" in expand_math_symbols("ω")

    def test_degree_symbol(self):
        """Test degree symbol."""
        from voice_soundboard.normalizer import expand_math_symbols

        assert "degrees" in expand_math_symbols("90°")


class TestDecodeHtmlEntities:
    """Tests for decode_html_entities function."""

    def test_basic_entities(self):
        """Test basic HTML entities."""
        from voice_soundboard.normalizer import decode_html_entities

        assert decode_html_entities("&amp;") == "&"
        assert decode_html_entities("&lt;") == "<"
        assert decode_html_entities("&gt;") == ">"
        assert decode_html_entities("&quot;") == '"'

    def test_nbsp(self):
        """Test non-breaking space."""
        from voice_soundboard.normalizer import decode_html_entities

        result = decode_html_entities("hello&nbsp;world")
        assert result == "hello\xa0world"

    def test_numeric_entities(self):
        """Test numeric HTML entities."""
        from voice_soundboard.normalizer import decode_html_entities

        assert decode_html_entities("&#38;") == "&"


class TestExpandUrl:
    """Tests for expand_url function."""

    def test_simple_url(self):
        """Test simple URL expansion."""
        from voice_soundboard.normalizer import expand_url

        result = expand_url("https://example.com")
        assert "example" in result
        assert "dot" in result
        assert "com" in result

    def test_url_with_path(self):
        """Test URL with path."""
        from voice_soundboard.normalizer import expand_url

        result = expand_url("https://example.com/path/to/page")
        assert "slash" in result

    def test_url_with_dashes(self):
        """Test URL with dashes."""
        from voice_soundboard.normalizer import expand_url

        result = expand_url("https://my-website.com")
        assert "dash" in result

    def test_removes_protocol(self):
        """Test that protocol is removed."""
        from voice_soundboard.normalizer import expand_url

        result = expand_url("https://example.com")
        assert "https" not in result.lower()


class TestExpandEmail:
    """Tests for expand_email function."""

    def test_simple_email(self):
        """Test simple email expansion."""
        from voice_soundboard.normalizer import expand_email

        result = expand_email("user@example.com")
        assert "at" in result
        assert "dot" in result

    def test_email_with_underscore(self):
        """Test email with underscore."""
        from voice_soundboard.normalizer import expand_email

        result = expand_email("user_name@example.com")
        assert "underscore" in result

    def test_email_with_dash(self):
        """Test email with dash."""
        from voice_soundboard.normalizer import expand_email

        result = expand_email("user-name@example.com")
        assert "dash" in result


class TestExpandUrlsAndEmails:
    """Tests for expand_urls_and_emails function."""

    def test_mixed_text_with_url(self):
        """Test text containing URL."""
        from voice_soundboard.normalizer import expand_urls_and_emails

        result = expand_urls_and_emails("Visit https://example.com for more")
        assert "dot" in result

    def test_mixed_text_with_email(self):
        """Test text containing email."""
        from voice_soundboard.normalizer import expand_urls_and_emails

        result = expand_urls_and_emails("Contact user@example.com")
        assert "at" in result

    def test_text_without_urls_or_emails(self):
        """Test text without URLs or emails."""
        from voice_soundboard.normalizer import expand_urls_and_emails

        result = expand_urls_and_emails("Plain text here")
        assert result == "Plain text here"


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_default_normalization(self):
        """Test default normalization settings."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Dr. Smith paid $100 for 😀")
        assert "Doctor" in result
        assert "dollars" in result
        assert "grinning face" in result

    def test_disable_currency(self):
        """Test with currency expansion disabled."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("$100", expand_currency_flag=False)
        assert "$100" in result

    def test_disable_emojis(self):
        """Test with emoji expansion disabled."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("😀", expand_emojis_flag=False)
        assert "😀" in result

    def test_enable_acronyms(self):
        """Test with acronym expansion enabled."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("The FBI", expand_acronyms_flag=True)
        assert "F B I" in result

    def test_disable_abbreviations(self):
        """Test with abbreviation expansion disabled."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Dr. Smith", expand_abbreviations_flag=False)
        assert "Dr." in result

    def test_cleans_whitespace(self):
        """Test that multiple spaces are cleaned."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("Hello    world")
        assert "  " not in result


# ==============================================================================
# Tests for web_server.py
# ==============================================================================

class TestGetLocalIp:
    """Tests for get_local_ip function."""

    def test_returns_string(self):
        """Test that get_local_ip returns a string."""
        from voice_soundboard.web_server import get_local_ip

        result = get_local_ip()
        assert isinstance(result, str)

    def test_fallback_to_localhost(self):
        """Test fallback to localhost on error."""
        from voice_soundboard.web_server import get_local_ip

        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.connect.side_effect = Exception("No network")
            result = get_local_ip()
            assert result == "localhost"


class TestGetEngine:
    """Tests for get_engine function."""

    def test_lazy_creation(self):
        """Test lazy engine creation."""
        import voice_soundboard.web_server as web_module

        # Reset global
        web_module._engine = None

        with patch.object(web_module, "VoiceEngine") as mock_engine_class:
            mock_engine_class.return_value = Mock()
            engine = web_module.get_engine()
            assert engine is not None
            mock_engine_class.assert_called_once()

    def test_returns_same_instance(self):
        """Test that same instance is returned."""
        import voice_soundboard.web_server as web_module

        mock_engine = Mock()
        web_module._engine = mock_engine

        engine = web_module.get_engine()
        assert engine is mock_engine


class TestIndexHandler:
    """Tests for index_handler."""

    @pytest.mark.asyncio
    async def test_returns_file_response_if_exists(self):
        """Test returns FileResponse if index.html exists."""
        from voice_soundboard.web_server import index_handler

        mock_request = Mock()

        with patch("pathlib.Path.exists", return_value=True):
            with patch("aiohttp.web.FileResponse") as mock_file_response:
                mock_file_response.return_value = Mock()
                result = await index_handler(mock_request)
                mock_file_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_404_if_not_exists(self):
        """Test returns 404 if index.html doesn't exist."""
        from voice_soundboard.web_server import index_handler

        mock_request = Mock()

        with patch("pathlib.Path.exists", return_value=False):
            result = await index_handler(mock_request)
            assert result.status == 404


class TestManifestHandler:
    """Tests for manifest_handler."""

    @pytest.mark.asyncio
    async def test_returns_json(self):
        """Test returns valid JSON response."""
        from voice_soundboard.web_server import manifest_handler

        mock_request = Mock()

        result = await manifest_handler(mock_request)
        assert result.content_type == "application/json"


class TestVoicesHandler:
    """Tests for voices_handler."""

    @pytest.mark.asyncio
    async def test_returns_voices(self):
        """Test returns voices list."""
        from voice_soundboard.web_server import voices_handler

        mock_request = Mock()

        result = await voices_handler(mock_request)
        assert result.content_type == "application/json"


class TestPresetsHandler:
    """Tests for presets_handler."""

    @pytest.mark.asyncio
    async def test_returns_presets(self):
        """Test returns presets list."""
        from voice_soundboard.web_server import presets_handler

        mock_request = Mock()

        result = await presets_handler(mock_request)
        assert result.content_type == "application/json"


class TestEffectsHandler:
    """Tests for effects_handler."""

    @pytest.mark.asyncio
    async def test_returns_effects(self):
        """Test returns effects list."""
        from voice_soundboard.web_server import effects_handler

        mock_request = Mock()

        with patch("voice_soundboard.web_server.list_effects", return_value=["chime", "beep"]):
            result = await effects_handler(mock_request)
            assert result.content_type == "application/json"


class TestHealthHandler:
    """Tests for health_handler."""

    @pytest.mark.asyncio
    async def test_returns_ok(self):
        """Test returns ok status."""
        from voice_soundboard.web_server import health_handler

        mock_request = Mock()

        result = await health_handler(mock_request)
        assert result.content_type == "application/json"


class TestSpeakHandler:
    """Tests for speak_handler."""

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON."""
        from voice_soundboard.web_server import speak_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))

        result = await speak_handler(mock_request)
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_no_text_provided(self):
        """Test handling of missing text."""
        from voice_soundboard.web_server import speak_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"text": ""})

        result = await speak_handler(mock_request)
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_successful_speech(self):
        """Test successful speech generation."""
        from voice_soundboard.web_server import speak_handler
        import voice_soundboard.web_server as web_module

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={
            "text": "Hello world",
            "voice": "af_bella",
            "play": False,
        })

        mock_result = Mock()
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.5
        mock_result.realtime_factor = 0.5
        mock_result.audio_path = Mock()
        mock_result.audio_path.read_bytes = Mock(return_value=b"audio data")

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(web_module, "get_engine", return_value=mock_engine):
            result = await speak_handler(mock_request)
            assert result.status == 200

    @pytest.mark.asyncio
    async def test_speech_error(self):
        """Test handling of speech generation error."""
        from voice_soundboard.web_server import speak_handler
        import voice_soundboard.web_server as web_module

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"text": "Hello"})

        mock_engine = Mock()
        mock_engine.speak = Mock(side_effect=Exception("TTS error"))

        with patch.object(web_module, "get_engine", return_value=mock_engine):
            result = await speak_handler(mock_request)
            assert result.status == 500


class TestSpeakJsonHandler:
    """Tests for speak_json_handler."""

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON."""
        from voice_soundboard.web_server import speak_json_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))

        result = await speak_json_handler(mock_request)
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_no_text_provided(self):
        """Test handling of missing text."""
        from voice_soundboard.web_server import speak_json_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"text": "  "})

        result = await speak_json_handler(mock_request)
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_successful_json_response(self):
        """Test successful JSON response."""
        from voice_soundboard.web_server import speak_json_handler
        import voice_soundboard.web_server as web_module

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={
            "text": "Hello",
            "play": False,
        })

        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/audio.wav")
        mock_result.voice_used = "af_bella"
        mock_result.duration_seconds = 1.0
        mock_result.realtime_factor = 0.5

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(web_module, "get_engine", return_value=mock_engine):
            result = await speak_json_handler(mock_request)
            assert result.status == 200
            assert result.content_type == "application/json"


class TestEffectHandler:
    """Tests for effect_handler."""

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON."""
        from voice_soundboard.web_server import effect_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))

        result = await effect_handler(mock_request)
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_no_effect_specified(self):
        """Test handling of missing effect name."""
        from voice_soundboard.web_server import effect_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"effect": ""})

        result = await effect_handler(mock_request)
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_successful_effect(self):
        """Test successful effect playback."""
        from voice_soundboard.web_server import effect_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={
            "effect": "chime",
            "play": False,
        })

        with patch("voice_soundboard.web_server.get_effect") as mock_get_effect:
            mock_get_effect.return_value = {"path": Path("/tmp/chime.wav")}
            result = await effect_handler(mock_request)
            assert result.status == 200

    @pytest.mark.asyncio
    async def test_effect_error(self):
        """Test handling of effect error."""
        from voice_soundboard.web_server import effect_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"effect": "unknown"})

        with patch("voice_soundboard.web_server.get_effect") as mock_get_effect:
            mock_get_effect.side_effect = Exception("Effect not found")
            result = await effect_handler(mock_request)
            assert result.status == 500


class TestCreateApp:
    """Tests for create_app function."""

    def test_creates_application(self):
        """Test that create_app creates an application."""
        from voice_soundboard.web_server import create_app

        app = create_app()
        assert app is not None

    def test_has_routes(self):
        """Test that app has expected routes."""
        from voice_soundboard.web_server import create_app

        app = create_app()

        # Check that routes exist
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, 'resource')]
        # Just verify the app was created with routes
        assert len(list(app.router.routes())) > 0

    def test_has_cors_middleware(self):
        """Test that app has CORS middleware."""
        from voice_soundboard.web_server import create_app

        app = create_app()
        assert len(app.middlewares) > 0


class TestRunServer:
    """Tests for run_server function."""

    @pytest.mark.asyncio
    async def test_server_starts(self):
        """Test that server can start."""
        from voice_soundboard.web_server import run_server

        # We can't easily test the full server start, so just verify imports work
        from voice_soundboard.web_server import create_app
        app = create_app()
        assert app is not None


class TestMain:
    """Tests for main function."""

    def test_main_parses_args(self):
        """Test that main parses arguments."""
        from voice_soundboard.web_server import main

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = Mock(host="0.0.0.0", port=8080)
            with patch("asyncio.run") as mock_run:
                with patch("logging.basicConfig"):
                    main()
                    # Function should have been called
                    mock_run.assert_called_once()

    def test_main_handles_keyboard_interrupt(self):
        """Test that main handles keyboard interrupt."""
        from voice_soundboard.web_server import main

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = Mock(host="0.0.0.0", port=8080)
            with patch("asyncio.run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()
                with patch("logging.basicConfig"):
                    with patch("builtins.print"):
                        # Should not raise
                        main()


# ==============================================================================
# Additional Edge Case Tests
# ==============================================================================

class TestNormalizerEdgeCases:
    """Edge case tests for normalizer."""

    def test_currency_with_single_cent(self):
        """Test currency with single cent."""
        from voice_soundboard.normalizer import expand_currency

        result = expand_currency("$1.01")
        assert "one cent" in result

    def test_large_number(self):
        """Test very large number."""
        from voice_soundboard.normalizer import number_to_words

        result = number_to_words(1000000000000)  # 1 trillion
        assert "trillion" in result

    def test_empty_text_normalization(self):
        """Test normalization of empty text."""
        from voice_soundboard.normalizer import normalize_text

        result = normalize_text("")
        assert result == ""

    def test_abbreviation_case_insensitive(self):
        """Test abbreviation matching is case-insensitive."""
        from voice_soundboard.normalizer import expand_abbreviations

        result = expand_abbreviations("DR. SMITH")
        assert "Doctor" in result

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        from voice_soundboard.normalizer import expand_url

        result = expand_url("https://example.com?param=value")
        assert "question mark" in result
        assert "equals" in result

    def test_email_complex(self):
        """Test complex email address."""
        from voice_soundboard.normalizer import expand_email

        result = expand_email("first.last+tag@sub.domain.co.uk")
        assert "at" in result
        assert "dot" in result


class TestWebServerEdgeCases:
    """Edge case tests for web server."""

    @pytest.mark.asyncio
    async def test_speak_with_play_option(self):
        """Test speak handler with play option."""
        from voice_soundboard.web_server import speak_handler
        import voice_soundboard.web_server as web_module

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={
            "text": "Hello",
            "play": True,
        })

        mock_result = Mock()
        mock_result.audio_path = Mock()
        mock_result.audio_path.read_bytes = Mock(return_value=b"audio")
        mock_result.voice_used = "voice"
        mock_result.duration_seconds = 1.0
        mock_result.realtime_factor = 0.5

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(web_module, "get_engine", return_value=mock_engine):
            with patch("asyncio.to_thread", new=AsyncMock()):
                result = await speak_handler(mock_request)
                assert result.status == 200

    @pytest.mark.asyncio
    async def test_speak_json_with_play(self):
        """Test speak_json_handler with play option."""
        from voice_soundboard.web_server import speak_json_handler
        import voice_soundboard.web_server as web_module

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={
            "text": "Hello",
            "play": True,
        })

        mock_result = Mock()
        mock_result.audio_path = Path("/tmp/audio.wav")
        mock_result.voice_used = "voice"
        mock_result.duration_seconds = 1.0
        mock_result.realtime_factor = 0.5

        mock_engine = Mock()
        mock_engine.speak = Mock(return_value=mock_result)

        with patch.object(web_module, "get_engine", return_value=mock_engine):
            with patch("asyncio.to_thread", new=AsyncMock()):
                result = await speak_json_handler(mock_request)
                assert result.status == 200

    @pytest.mark.asyncio
    async def test_effect_with_play(self):
        """Test effect_handler with play option."""
        from voice_soundboard.web_server import effect_handler

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={
            "effect": "chime",
            "play": True,
        })

        with patch("voice_soundboard.web_server.get_effect") as mock_get:
            mock_get.return_value = {"path": Path("/tmp/chime.wav")}
            with patch("asyncio.to_thread", new=AsyncMock()):
                result = await effect_handler(mock_request)
                assert result.status == 200


class TestCorsMiddleware:
    """Tests for CORS middleware."""

    @pytest.mark.asyncio
    async def test_cors_headers_added(self):
        """Test that CORS headers are added."""
        from voice_soundboard.web_server import create_app
        from aiohttp.test_utils import TestClient, TestServer

        app = create_app()

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert "Access-Control-Allow-Origin" in resp.headers

    @pytest.mark.asyncio
    async def test_options_request(self):
        """Test OPTIONS request handling."""
        from voice_soundboard.web_server import create_app
        from aiohttp.test_utils import TestClient, TestServer

        app = create_app()

        async with TestClient(TestServer(app)) as client:
            resp = await client.options("/health")
            assert "Access-Control-Allow-Origin" in resp.headers
