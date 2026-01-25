"""
Additional tests - Batch 7.

Covers remaining unchecked items from TEST_PLAN.md:
- llm/context.py (ProsodyHint, ContextConfig, ConversationContext, EmotionSelector, ContextAwareSpeaker)
- llm/interruption.py (InterruptionStrategy, InterruptionEvent, BargeInConfig, BargeInDetector, InterruptionHandler)
- cloning/cloner.py (CloningConfig, CloningResult, VoiceCloner)
"""

import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile


# =============================================================================
# Module: llm/context.py - ProsodyHint Tests
# =============================================================================

class TestProsodyHint:
    """Tests for ProsodyHint enum."""

    def test_prosody_hint_values(self):
        """Test ProsodyHint enum values."""
        from voice_soundboard.llm.context import ProsodyHint

        assert ProsodyHint.NEUTRAL.value == "neutral"
        assert ProsodyHint.EMPATHETIC.value == "empathetic"
        assert ProsodyHint.EXCITED.value == "excited"
        assert ProsodyHint.CALM.value == "calm"
        assert ProsodyHint.SERIOUS.value == "serious"
        assert ProsodyHint.PLAYFUL.value == "playful"
        assert ProsodyHint.APOLOGETIC.value == "apologetic"
        assert ProsodyHint.ENCOURAGING.value == "encouraging"
        assert ProsodyHint.CONCERNED.value == "concerned"
        assert ProsodyHint.PROFESSIONAL.value == "professional"


# =============================================================================
# Module: llm/context.py - ContextConfig Tests
# =============================================================================

class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_context_config_defaults(self):
        """Test ContextConfig default values."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig()

        assert config.enable_auto_emotion is True
        assert config.emotion_confidence_threshold == 0.5
        assert config.analyze_user_sentiment is True
        assert config.history_window == 5
        assert config.default_emotion == "neutral"
        assert config.empathy_speed_factor == 0.9
        assert config.excitement_speed_factor == 1.1

    def test_context_config_custom(self):
        """Test ContextConfig with custom values."""
        from voice_soundboard.llm.context import ContextConfig

        config = ContextConfig(
            enable_auto_emotion=False,
            emotion_confidence_threshold=0.7,
            history_window=10,
        )

        assert config.enable_auto_emotion is False
        assert config.emotion_confidence_threshold == 0.7
        assert config.history_window == 10


# =============================================================================
# Module: llm/context.py - ConversationContext Tests
# =============================================================================

class TestConversationContext:
    """Tests for ConversationContext dataclass."""

    def test_conversation_context_initialization(self):
        """Test ConversationContext initialization."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()

        assert context.messages == []
        assert context.turn_count == 0
        assert context.user_sentiment is None
        assert context.user_emotion is None

    def test_conversation_context_add_message(self):
        """Test ConversationContext.add_message()."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()

        context.add_message("user", "Hello!")
        context.add_message("assistant", "Hi there!")

        assert len(context.messages) == 2
        assert context.turn_count == 2
        assert context.messages[0]["role"] == "user"
        assert context.messages[0]["content"] == "Hello!"

    def test_conversation_context_get_recent_messages(self):
        """Test ConversationContext.get_recent_messages()."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()

        for i in range(10):
            context.add_message("user", f"Message {i}")

        recent = context.get_recent_messages(3)

        assert len(recent) == 3
        assert recent[0]["content"] == "Message 7"
        assert recent[2]["content"] == "Message 9"

    def test_conversation_context_get_last_user_message(self):
        """Test ConversationContext.get_last_user_message()."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()

        context.add_message("user", "First user message")
        context.add_message("assistant", "Response")
        context.add_message("user", "Second user message")

        last = context.get_last_user_message()

        assert last == "Second user message"

    def test_conversation_context_get_last_user_message_empty(self):
        """Test get_last_user_message with no user messages."""
        from voice_soundboard.llm.context import ConversationContext

        context = ConversationContext()
        context.add_message("assistant", "Hello!")

        assert context.get_last_user_message() is None


# =============================================================================
# Module: llm/context.py - EmotionSelector Tests
# =============================================================================

class TestEmotionSelector:
    """Tests for EmotionSelector class."""

    def test_emotion_selector_initialization(self):
        """Test EmotionSelector initialization."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        assert selector.config is not None

    def test_select_emotion_default(self):
        """Test selecting emotion returns default for neutral text."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        emotion, confidence = selector.select_emotion("This is a normal sentence.")

        assert emotion == "neutral"

    def test_select_emotion_happy_keywords(self):
        """Test selecting emotion based on happy keywords."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        emotion, confidence = selector.select_emotion(
            "That's wonderful! Great job, this is fantastic news!"
        )

        assert emotion == "happy"
        assert confidence > 0

    def test_select_emotion_sympathetic_keywords(self):
        """Test selecting emotion based on sympathetic keywords."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        emotion, confidence = selector.select_emotion(
            "I'm sorry to hear that. That sounds really difficult and frustrating."
        )

        assert emotion == "sympathetic"
        assert confidence > 0

    def test_select_emotion_with_hint(self):
        """Test selecting emotion with prosody hint."""
        from voice_soundboard.llm.context import EmotionSelector, ProsodyHint

        selector = EmotionSelector()

        emotion, confidence = selector.select_emotion(
            "Normal text",
            hint=ProsodyHint.EXCITED,
        )

        assert emotion == "excited"

    def test_detect_content_type_question(self):
        """Test detecting question content type."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        content_type = selector._detect_content_type("What is your name?")
        assert content_type == "question"

        content_type = selector._detect_content_type("How does this work?")
        assert content_type == "question"

    def test_detect_content_type_greeting(self):
        """Test detecting greeting content type."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        # Use greeting without question mark to avoid question detection
        content_type = selector._detect_content_type("Hello everyone!")
        assert content_type == "greeting"

        content_type = selector._detect_content_type("Good morning everyone!")
        assert content_type == "greeting"

    def test_detect_content_type_farewell(self):
        """Test detecting farewell content type."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        content_type = selector._detect_content_type("Goodbye, take care!")
        assert content_type == "farewell"

    def test_detect_content_type_apology(self):
        """Test detecting apology content type."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        content_type = selector._detect_content_type("I'm sorry for the confusion.")
        assert content_type == "apology"

    def test_detect_user_sentiment_frustrated(self):
        """Test detecting frustrated user sentiment."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        sentiment, emotion = selector.detect_user_sentiment(
            "This is so frustrating! It doesn't work at all."
        )

        assert sentiment == "negative"
        assert emotion == "frustrated"

    def test_detect_user_sentiment_happy(self):
        """Test detecting happy user sentiment."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        sentiment, emotion = selector.detect_user_sentiment(
            "Thank you so much! This is awesome!"
        )

        assert sentiment == "positive"
        assert emotion in ["happy", "excited"]

    def test_detect_user_sentiment_neutral(self):
        """Test detecting neutral user sentiment."""
        from voice_soundboard.llm.context import EmotionSelector

        selector = EmotionSelector()

        sentiment, emotion = selector.detect_user_sentiment(
            "Please process my request."
        )

        assert sentiment == "neutral"
        assert emotion == "neutral"


# =============================================================================
# Module: llm/context.py - ContextAwareSpeaker Tests
# =============================================================================

class TestContextAwareSpeaker:
    """Tests for ContextAwareSpeaker class."""

    def test_context_aware_speaker_initialization(self):
        """Test ContextAwareSpeaker initialization."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()

        assert speaker.config is not None
        assert speaker.context is not None
        assert speaker.emotion_selector is not None

    def test_context_aware_speaker_update_context(self):
        """Test ContextAwareSpeaker.update_context()."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()

        speaker.update_context("I'm really frustrated with this issue!")

        assert speaker.context.user_sentiment == "negative"
        assert speaker.context.user_emotion == "frustrated"

    def test_context_aware_speaker_reset_context(self):
        """Test ContextAwareSpeaker.reset_context()."""
        from voice_soundboard.llm.context import ContextAwareSpeaker

        speaker = ContextAwareSpeaker()
        speaker.update_context("Hello!")
        speaker.update_context("How are you?")

        assert speaker.context.turn_count == 2

        speaker.reset_context()

        assert speaker.context.turn_count == 0
        assert len(speaker.context.messages) == 0


# =============================================================================
# Module: llm/interruption.py - InterruptionStrategy Tests
# =============================================================================

class TestInterruptionStrategy:
    """Tests for InterruptionStrategy enum."""

    def test_interruption_strategy_values(self):
        """Test InterruptionStrategy enum values."""
        from voice_soundboard.llm.interruption import InterruptionStrategy

        assert InterruptionStrategy.IGNORE.value == "ignore"
        assert InterruptionStrategy.STOP_IMMEDIATE.value == "stop_immediate"
        assert InterruptionStrategy.STOP_SENTENCE.value == "stop_sentence"
        assert InterruptionStrategy.PAUSE.value == "pause"
        assert InterruptionStrategy.QUEUE.value == "queue"


# =============================================================================
# Module: llm/interruption.py - InterruptionEvent Tests
# =============================================================================

class TestInterruptionEvent:
    """Tests for InterruptionEvent dataclass."""

    def test_interruption_event_creation(self):
        """Test InterruptionEvent creation."""
        from voice_soundboard.llm.interruption import InterruptionEvent

        event = InterruptionEvent()

        assert event.trigger == "user"
        assert event.handled is False
        assert event.audio_level is None
        assert event.transcript is None

    def test_interruption_event_age_ms(self):
        """Test InterruptionEvent.age_ms property."""
        from voice_soundboard.llm.interruption import InterruptionEvent

        event = InterruptionEvent()

        # Age should be very small
        assert event.age_ms < 100

        time.sleep(0.05)
        assert event.age_ms >= 45

    def test_interruption_event_custom_values(self):
        """Test InterruptionEvent with custom values."""
        from voice_soundboard.llm.interruption import InterruptionEvent

        event = InterruptionEvent(
            trigger="voice_activity",
            audio_level=-30.0,
            transcript="stop",
        )

        assert event.trigger == "voice_activity"
        assert event.audio_level == -30.0
        assert event.transcript == "stop"


# =============================================================================
# Module: llm/interruption.py - BargeInConfig Tests
# =============================================================================

class TestBargeInConfig:
    """Tests for BargeInConfig dataclass."""

    def test_barge_in_config_defaults(self):
        """Test BargeInConfig default values."""
        from voice_soundboard.llm.interruption import BargeInConfig, InterruptionStrategy

        config = BargeInConfig()

        assert config.enabled is True
        assert config.vad_threshold_db == -35.0
        assert config.vad_duration_ms == 200.0
        assert config.vad_cooldown_ms == 500.0
        assert config.hotword_enabled is False
        assert config.default_strategy == InterruptionStrategy.STOP_IMMEDIATE

    def test_barge_in_config_hotwords(self):
        """Test BargeInConfig default hotwords."""
        from voice_soundboard.llm.interruption import BargeInConfig

        config = BargeInConfig()

        assert "hey" in config.hotwords
        assert "stop" in config.hotwords
        assert "wait" in config.hotwords


# =============================================================================
# Module: llm/interruption.py - BargeInDetector Tests
# =============================================================================

class TestBargeInDetector:
    """Tests for BargeInDetector class."""

    def test_barge_in_detector_initialization(self):
        """Test BargeInDetector initialization."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        assert detector.config is not None
        assert detector._is_listening is False

    def test_barge_in_detector_start_stop_listening(self):
        """Test BargeInDetector start/stop listening."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        detector.start_listening()
        assert detector._is_listening is True
        assert detector._speaking_started is not None

        detector.stop_listening()
        assert detector._is_listening is False
        assert detector._speaking_started is None

    def test_barge_in_detector_on_interrupt_callback(self):
        """Test BargeInDetector.on_interrupt() callback registration."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        callback_called = []

        def my_callback(event):
            callback_called.append(event)

        detector.on_interrupt(my_callback)

        assert len(detector._callbacks) == 1

    def test_barge_in_detector_trigger_manual(self):
        """Test BargeInDetector.trigger_manual()."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        event = detector.trigger_manual()

        assert event.trigger == "manual"

    def test_barge_in_detector_check_audio_level_not_listening(self):
        """Test check_audio_level when not listening."""
        from voice_soundboard.llm.interruption import BargeInDetector

        detector = BargeInDetector()

        result = detector.check_audio_level(-20.0)

        assert result is None

    def test_barge_in_detector_check_transcript_hotword(self):
        """Test check_transcript with hotword detection."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(hotword_enabled=True, hotwords=["stop", "wait"])
        detector = BargeInDetector(config)
        detector.start_listening()

        event = detector.check_transcript("Please stop talking")

        assert event is not None
        assert event.trigger == "hotword"
        assert event.transcript == "Please stop talking"

    def test_barge_in_detector_check_transcript_no_hotword(self):
        """Test check_transcript without hotword."""
        from voice_soundboard.llm.interruption import BargeInDetector, BargeInConfig

        config = BargeInConfig(hotword_enabled=True, hotwords=["stop", "wait"])
        detector = BargeInDetector(config)
        detector.start_listening()

        event = detector.check_transcript("Hello there")

        assert event is None


# =============================================================================
# Module: llm/interruption.py - InterruptionHandler Tests
# =============================================================================

class TestInterruptionHandler:
    """Tests for InterruptionHandler class."""

    def test_interruption_handler_initialization(self):
        """Test InterruptionHandler initialization."""
        from voice_soundboard.llm.interruption import InterruptionHandler, InterruptionStrategy

        handler = InterruptionHandler()

        assert handler.strategy == InterruptionStrategy.STOP_IMMEDIATE
        assert handler._is_active is False
        assert handler._is_paused is False

    def test_interruption_handler_start_end_session(self):
        """Test InterruptionHandler start/end session."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()

        handler.start_session()
        assert handler._is_active is True

        handler.end_session()
        assert handler._is_active is False

    def test_interruption_handler_on_interrupt_callback(self):
        """Test InterruptionHandler on_interrupt callback property."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()

        callback = Mock()
        handler.on_interrupt = callback

        assert handler.on_interrupt is callback

    def test_interruption_handler_force_interrupt(self):
        """Test InterruptionHandler.force_interrupt()."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        handler.start_session()

        callback = Mock()
        handler.on_interrupt = callback

        event = handler.force_interrupt()

        assert event.trigger == "manual"
        # Callback may be called multiple times due to event propagation
        assert callback.call_count >= 1

    def test_interruption_handler_stats(self):
        """Test InterruptionHandler.stats property."""
        from voice_soundboard.llm.interruption import InterruptionHandler

        handler = InterruptionHandler()
        handler.start_session()

        stats = handler.stats

        assert stats["is_active"] is True
        assert stats["is_paused"] is False
        assert "total_interruptions" in stats
        assert "strategy" in stats

    def test_interruption_handler_ignore_strategy(self):
        """Test InterruptionHandler with IGNORE strategy."""
        from voice_soundboard.llm.interruption import InterruptionHandler, InterruptionStrategy

        handler = InterruptionHandler(strategy=InterruptionStrategy.IGNORE)
        handler.start_session()

        callback = Mock()
        handler.on_interrupt = callback

        handler.force_interrupt()

        # Callback should not be called with IGNORE strategy
        callback.assert_not_called()

    def test_interruption_handler_pause_strategy(self):
        """Test InterruptionHandler with PAUSE strategy."""
        from voice_soundboard.llm.interruption import InterruptionHandler, InterruptionStrategy

        handler = InterruptionHandler(strategy=InterruptionStrategy.PAUSE)
        handler.start_session()

        pause_callback = Mock()
        handler.on_pause = pause_callback

        handler.force_interrupt()

        assert handler._is_paused is True
        pause_callback.assert_called_once()

    def test_interruption_handler_resume(self):
        """Test InterruptionHandler.resume()."""
        from voice_soundboard.llm.interruption import InterruptionHandler, InterruptionStrategy

        handler = InterruptionHandler(strategy=InterruptionStrategy.PAUSE)
        handler.start_session()

        resume_callback = Mock()
        handler.on_resume = resume_callback

        handler.force_interrupt()  # Pause
        handler.resume()

        assert handler._is_paused is False
        resume_callback.assert_called_once()

    def test_interruption_handler_queue_strategy(self):
        """Test InterruptionHandler with QUEUE strategy."""
        from voice_soundboard.llm.interruption import InterruptionHandler, InterruptionStrategy

        handler = InterruptionHandler(strategy=InterruptionStrategy.QUEUE)
        handler.start_session()

        # Simulate interruption with transcript
        handler.detector.start_listening()
        handler.detector._speaking_started = time.time() - 2  # 2 seconds ago

        # Force through detector
        event = handler.detector.trigger_manual()
        event.transcript = "user message"
        handler._handle_event(event)

        messages = handler.get_queued_messages()
        assert "user message" in messages

    def test_interruption_handler_get_queued_messages_clears(self):
        """Test that get_queued_messages clears the queue."""
        from voice_soundboard.llm.interruption import InterruptionHandler, InterruptionStrategy

        handler = InterruptionHandler(strategy=InterruptionStrategy.QUEUE)
        handler._queue = ["msg1", "msg2"]

        messages = handler.get_queued_messages()
        assert len(messages) == 2

        # Queue should be cleared
        assert len(handler._queue) == 0


# =============================================================================
# Module: cloning/cloner.py - CloningConfig Tests
# =============================================================================

class TestCloningConfig:
    """Tests for CloningConfig dataclass."""

    def test_cloning_config_defaults(self):
        """Test CloningConfig default values."""
        from voice_soundboard.cloning.cloner import CloningConfig
        from voice_soundboard.cloning.extractor import ExtractorBackend

        config = CloningConfig()

        assert config.extractor_backend == ExtractorBackend.MOCK
        assert config.device == "cpu"
        assert config.min_audio_seconds == 1.0
        assert config.max_audio_seconds == 30.0
        assert config.optimal_audio_seconds == 5.0
        assert config.min_quality_score == 0.3
        assert config.require_consent is True
        assert config.add_watermark is False

    def test_cloning_config_custom(self):
        """Test CloningConfig with custom values."""
        from voice_soundboard.cloning.cloner import CloningConfig
        from voice_soundboard.cloning.extractor import ExtractorBackend

        config = CloningConfig(
            extractor_backend=ExtractorBackend.MOCK,
            device="cuda",
            min_audio_seconds=2.0,
            require_consent=False,
        )

        assert config.device == "cuda"
        assert config.min_audio_seconds == 2.0
        assert config.require_consent is False


# =============================================================================
# Module: cloning/cloner.py - CloningResult Tests
# =============================================================================

class TestCloningResult:
    """Tests for CloningResult dataclass."""

    def test_cloning_result_success(self):
        """Test CloningResult for successful cloning."""
        from voice_soundboard.cloning.cloner import CloningResult

        result = CloningResult(
            success=True,
            voice_id="my_voice",
            quality_score=0.9,
            snr_db=25.0,
            audio_duration=5.0,
        )

        assert result.success is True
        assert result.voice_id == "my_voice"
        assert result.error is None

    def test_cloning_result_failure(self):
        """Test CloningResult for failed cloning."""
        from voice_soundboard.cloning.cloner import CloningResult

        result = CloningResult(
            success=False,
            voice_id="failed_voice",
            error="Audio too short",
        )

        assert result.success is False
        assert result.error == "Audio too short"

    def test_cloning_result_warnings_recommendations(self):
        """Test CloningResult with warnings and recommendations."""
        from voice_soundboard.cloning.cloner import CloningResult

        result = CloningResult(
            success=True,
            voice_id="test",
            warnings=["Low quality audio"],
            recommendations=["Use cleaner audio"],
        )

        assert len(result.warnings) == 1
        assert len(result.recommendations) == 1


# =============================================================================
# Module: cloning/cloner.py - VoiceCloner Tests
# =============================================================================

class TestVoiceCloner:
    """Tests for VoiceCloner class."""

    def test_voice_cloner_initialization(self, tmp_path):
        """Test VoiceCloner initialization."""
        from voice_soundboard.cloning.cloner import VoiceCloner
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        cloner = VoiceCloner(library=library)

        assert cloner.config is not None
        assert cloner.library is library

    def test_voice_cloner_extractor_property(self, tmp_path):
        """Test VoiceCloner.extractor property."""
        from voice_soundboard.cloning.cloner import VoiceCloner
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        cloner = VoiceCloner(library=library)

        extractor = cloner.extractor

        assert extractor is not None
        assert cloner._extractor is extractor  # Cached

    def test_voice_cloner_clone_requires_consent(self, tmp_path):
        """Test VoiceCloner.clone() requires consent."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=True)
        cloner = VoiceCloner(config=config, library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.clone(audio, "test_voice", consent_given=False)

        assert result.success is False
        assert "consent" in result.error.lower()

    def test_voice_cloner_clone_success(self, tmp_path):
        """Test VoiceCloner.clone() success."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary
        import soundfile as sf

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=True, use_segment_averaging=False)
        cloner = VoiceCloner(config=config, library=library)

        # Create audio file
        audio = np.random.randn(16000 * 5).astype(np.float32)
        audio_path = tmp_path / "source.wav"
        sf.write(str(audio_path), audio, 16000)

        result = cloner.clone(
            audio_path,
            "test_voice",
            name="Test Voice",
            consent_given=True,
            consent_notes="Testing",
        )

        assert result.success is True
        assert result.voice_id == "test_voice"
        assert result.profile is not None
        assert result.embedding is not None

    def test_voice_cloner_clone_duplicate(self, tmp_path):
        """Test VoiceCloner.clone() rejects duplicate."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary
        import soundfile as sf

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False, use_segment_averaging=False)
        cloner = VoiceCloner(config=config, library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        audio_path = tmp_path / "source.wav"
        sf.write(str(audio_path), audio, 16000)

        # First clone
        result1 = cloner.clone(audio_path, "dup_voice")
        assert result1.success is True

        # Second clone with same ID
        result2 = cloner.clone(audio_path, "dup_voice")
        assert result2.success is False
        assert "already exists" in result2.error

    def test_voice_cloner_clone_file_not_found(self, tmp_path):
        """Test VoiceCloner.clone() with missing file."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False)
        cloner = VoiceCloner(config=config, library=library)

        result = cloner.clone("/nonexistent/audio.wav", "test_voice")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_voice_cloner_clone_quick(self, tmp_path):
        """Test VoiceCloner.clone_quick()."""
        from voice_soundboard.cloning.cloner import VoiceCloner
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        cloner = VoiceCloner(library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.clone_quick(audio, "Quick Test")

        assert result.success is True
        assert result.voice_id.startswith("temp_")
        assert result.profile is not None
        # Quick clone should not save to library
        assert result.voice_id not in library

    def test_voice_cloner_delete_voice(self, tmp_path):
        """Test VoiceCloner.delete_voice()."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary
        import soundfile as sf

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False, use_segment_averaging=False)
        cloner = VoiceCloner(config=config, library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        audio_path = tmp_path / "source.wav"
        sf.write(str(audio_path), audio, 16000)

        cloner.clone(audio_path, "to_delete")
        assert "to_delete" in library

        result = cloner.delete_voice("to_delete")
        assert result is True
        assert "to_delete" not in library

    def test_voice_cloner_get_voice(self, tmp_path):
        """Test VoiceCloner.get_voice()."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary
        import soundfile as sf

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False, use_segment_averaging=False)
        cloner = VoiceCloner(config=config, library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        audio_path = tmp_path / "source.wav"
        sf.write(str(audio_path), audio, 16000)

        cloner.clone(audio_path, "my_voice")

        profile = cloner.get_voice("my_voice")
        assert profile is not None
        assert profile.voice_id == "my_voice"

        # Non-existent voice
        assert cloner.get_voice("nonexistent") is None

    def test_voice_cloner_list_voices(self, tmp_path):
        """Test VoiceCloner.list_voices()."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary
        import soundfile as sf

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False, use_segment_averaging=False)
        cloner = VoiceCloner(config=config, library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        audio_path = tmp_path / "source.wav"
        sf.write(str(audio_path), audio, 16000)

        cloner.clone(audio_path, "voice1", tags=["test"])
        cloner.clone(audio_path, "voice2", tags=["demo"])

        all_voices = cloner.list_voices()
        assert len(all_voices) == 2

        test_voices = cloner.list_voices(tags=["test"])
        assert len(test_voices) == 1

    def test_voice_cloner_validate_audio(self, tmp_path):
        """Test VoiceCloner.validate_audio()."""
        from voice_soundboard.cloning.cloner import VoiceCloner
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        cloner = VoiceCloner(library=library)

        # Good audio (5 seconds)
        good_audio = np.random.randn(16000 * 5).astype(np.float32)
        result = cloner.validate_audio(good_audio)

        assert result["is_valid"] is True
        assert result["duration_seconds"] > 0

    def test_voice_cloner_validate_audio_too_short(self, tmp_path):
        """Test VoiceCloner.validate_audio() with short audio."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(min_audio_seconds=3.0)
        cloner = VoiceCloner(config=config, library=library)

        # Short audio (0.5 seconds)
        short_audio = np.random.randn(int(16000 * 0.5)).astype(np.float32)
        result = cloner.validate_audio(short_audio)

        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_voice_cloner_update_voice(self, tmp_path):
        """Test VoiceCloner.update_voice()."""
        from voice_soundboard.cloning.cloner import VoiceCloner, CloningConfig
        from voice_soundboard.cloning.library import VoiceLibrary
        import soundfile as sf

        library = VoiceLibrary(library_path=tmp_path / "voices")
        config = CloningConfig(require_consent=False, use_segment_averaging=False)
        cloner = VoiceCloner(config=config, library=library)

        audio = np.random.randn(16000 * 5).astype(np.float32)
        audio_path = tmp_path / "source.wav"
        sf.write(str(audio_path), audio, 16000)

        cloner.clone(audio_path, "update_me", name="Original Name")

        result = cloner.update_voice("update_me", name="Updated Name")

        assert result.success is True
        assert result.profile.name == "Updated Name"

    def test_voice_cloner_update_voice_not_found(self, tmp_path):
        """Test VoiceCloner.update_voice() with non-existent voice."""
        from voice_soundboard.cloning.cloner import VoiceCloner
        from voice_soundboard.cloning.library import VoiceLibrary

        library = VoiceLibrary(library_path=tmp_path / "voices")
        cloner = VoiceCloner(library=library)

        result = cloner.update_voice("nonexistent", name="New Name")

        assert result.success is False
        assert "not found" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
