"""
Context-aware prosody and emotion selection.

This module enables automatic emotion and speaking style selection
based on conversation context and content analysis.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import time


class ProsodyHint(Enum):
    """Hints for prosody adjustment."""
    NEUTRAL = "neutral"
    EMPATHETIC = "empathetic"
    EXCITED = "excited"
    CALM = "calm"
    SERIOUS = "serious"
    PLAYFUL = "playful"
    APOLOGETIC = "apologetic"
    ENCOURAGING = "encouraging"
    CONCERNED = "concerned"
    PROFESSIONAL = "professional"


@dataclass
class ContextConfig:
    """Configuration for context-aware speaker."""

    # Emotion detection
    enable_auto_emotion: bool = True
    emotion_confidence_threshold: float = 0.5

    # Context analysis
    analyze_user_sentiment: bool = True
    analyze_content_type: bool = True
    use_conversation_history: bool = True
    history_window: int = 5  # Number of turns to consider

    # Speaking style
    default_emotion: str = "neutral"
    default_preset: str = "assistant"

    # Prosody adjustments
    empathy_speed_factor: float = 0.9  # Slow down for empathy
    excitement_speed_factor: float = 1.1  # Speed up for excitement

    # Content-based rules
    question_emotion: str = "curious"
    apology_emotion: str = "sympathetic"
    greeting_emotion: str = "friendly"
    farewell_emotion: str = "warm"


@dataclass
class ConversationContext:
    """Context from the ongoing conversation."""

    # Recent messages
    messages: List[Dict[str, str]] = field(default_factory=list)

    # User state (detected from their messages)
    user_sentiment: Optional[str] = None  # positive, negative, neutral
    user_emotion: Optional[str] = None  # frustrated, happy, confused, etc.
    user_urgency: Optional[str] = None  # low, medium, high

    # Conversation metadata
    topic: Optional[str] = None
    turn_count: int = 0
    started_at: float = field(default_factory=time.time)

    # Custom context
    custom: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the context."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.turn_count += 1

    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []

    def get_last_user_message(self) -> Optional[str]:
        """Get the last message from the user."""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None


class EmotionSelector:
    """
    Selects appropriate emotions based on context and content.

    Uses a combination of:
    - Keyword analysis
    - User sentiment response
    - Content type detection
    - Conversation history
    """

    # Emotion keywords in the response
    EMOTION_KEYWORDS = {
        "happy": ["great", "wonderful", "fantastic", "excellent", "amazing", "glad", "pleased"],
        "excited": ["wow", "incredible", "awesome", "exciting", "thrilled"],
        "sympathetic": ["sorry", "understand", "difficult", "tough", "challenging", "frustrating"],
        "curious": ["interesting", "wonder", "curious", "fascinating"],
        "confident": ["definitely", "certainly", "absolutely", "sure", "guaranteed"],
        "friendly": ["hello", "hi", "hey", "welcome", "nice to meet"],
        "warm": ["goodbye", "bye", "take care", "see you", "farewell"],
        "calm": ["relax", "calm", "peace", "easy", "gentle", "slowly"],
        "serious": ["important", "critical", "urgent", "warning", "caution"],
    }

    # User sentiment to response emotion mapping
    SENTIMENT_RESPONSE = {
        "frustrated": "sympathetic",
        "confused": "patient",
        "angry": "calm",
        "sad": "sympathetic",
        "happy": "friendly",
        "excited": "excited",
        "anxious": "calm",
        "neutral": "neutral",
    }

    # Content type to emotion mapping
    CONTENT_TYPE_EMOTION = {
        "question": "curious",
        "explanation": "patient",
        "apology": "sympathetic",
        "greeting": "friendly",
        "farewell": "warm",
        "joke": "playful",
        "warning": "serious",
        "encouragement": "encouraging",
        "celebration": "excited",
    }

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def select_emotion(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
        hint: Optional[ProsodyHint] = None,
    ) -> Tuple[str, float]:
        """
        Select the best emotion for speaking the given text.

        Returns:
            Tuple of (emotion_name, confidence_score)
        """
        scores: Dict[str, float] = {}

        # 1. Analyze response keywords
        keyword_emotion, keyword_score = self._analyze_keywords(text)
        if keyword_emotion:
            scores[keyword_emotion] = scores.get(keyword_emotion, 0) + keyword_score

        # 2. Detect content type
        content_type = self._detect_content_type(text)
        if content_type and content_type in self.CONTENT_TYPE_EMOTION:
            emotion = self.CONTENT_TYPE_EMOTION[content_type]
            scores[emotion] = scores.get(emotion, 0) + 0.3

        # 3. Consider user sentiment (if context available)
        if context and self.config.analyze_user_sentiment:
            if context.user_emotion and context.user_emotion in self.SENTIMENT_RESPONSE:
                emotion = self.SENTIMENT_RESPONSE[context.user_emotion]
                scores[emotion] = scores.get(emotion, 0) + 0.4

        # 4. Apply hint if provided
        if hint:
            hint_emotion = self._hint_to_emotion(hint)
            if hint_emotion:
                scores[hint_emotion] = scores.get(hint_emotion, 0) + 0.5

        # Select highest scoring emotion
        if scores:
            best_emotion = max(scores, key=scores.get)
            confidence = min(scores[best_emotion], 1.0)

            if confidence >= self.config.emotion_confidence_threshold:
                return best_emotion, confidence

        return self.config.default_emotion, 0.0

    def _analyze_keywords(self, text: str) -> Tuple[Optional[str], float]:
        """Analyze text for emotion keywords."""
        text_lower = text.lower()
        best_emotion = None
        best_score = 0.0

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                score = min(count * 0.2, 0.6)  # Cap at 0.6
                if score > best_score:
                    best_score = score
                    best_emotion = emotion

        return best_emotion, best_score

    def _detect_content_type(self, text: str) -> Optional[str]:
        """Detect the type of content in the text."""
        text_lower = text.lower().strip()

        # Question
        if text.strip().endswith("?") or any(
            text_lower.startswith(q) for q in ["what", "how", "why", "when", "where", "who", "can", "could", "would"]
        ):
            return "question"

        # Greeting
        greetings = ["hello", "hi ", "hey ", "good morning", "good afternoon", "good evening", "welcome"]
        if any(text_lower.startswith(g) for g in greetings):
            return "greeting"

        # Farewell
        farewells = ["goodbye", "bye", "see you", "take care", "farewell", "have a great"]
        if any(f in text_lower for f in farewells):
            return "farewell"

        # Apology
        apologies = ["sorry", "apologize", "my apologies", "forgive me"]
        if any(a in text_lower for a in apologies):
            return "apology"

        # Warning
        warnings = ["warning", "caution", "be careful", "danger", "alert", "important:"]
        if any(w in text_lower for w in warnings):
            return "warning"

        # Encouragement
        encouragements = ["you can do", "keep going", "great job", "well done", "proud of you"]
        if any(e in text_lower for e in encouragements):
            return "encouragement"

        return None

    def _hint_to_emotion(self, hint: ProsodyHint) -> Optional[str]:
        """Convert prosody hint to emotion."""
        mapping = {
            ProsodyHint.NEUTRAL: "neutral",
            ProsodyHint.EMPATHETIC: "sympathetic",
            ProsodyHint.EXCITED: "excited",
            ProsodyHint.CALM: "calm",
            ProsodyHint.SERIOUS: "serious",
            ProsodyHint.PLAYFUL: "playful",
            ProsodyHint.APOLOGETIC: "sympathetic",
            ProsodyHint.ENCOURAGING: "encouraging",
            ProsodyHint.CONCERNED: "concerned",
            ProsodyHint.PROFESSIONAL: "confident",
        }
        return mapping.get(hint)

    def detect_user_sentiment(self, user_message: str) -> Tuple[str, str]:
        """
        Detect user sentiment and emotion from their message.

        Returns:
            Tuple of (sentiment, emotion)
        """
        text_lower = user_message.lower()

        # Frustration indicators
        frustration_words = ["frustrated", "annoying", "doesn't work", "broken", "stupid", "hate", "can't believe"]
        if any(w in text_lower for w in frustration_words):
            return "negative", "frustrated"

        # Confusion indicators
        confusion_words = ["confused", "don't understand", "what do you mean", "unclear", "lost", "help me understand"]
        if any(w in text_lower for w in confusion_words):
            return "negative", "confused"

        # Anger indicators
        anger_words = ["angry", "furious", "outraged", "unacceptable", "terrible", "worst"]
        if any(w in text_lower for w in anger_words):
            return "negative", "angry"

        # Happiness indicators
        happy_words = ["thanks", "thank you", "great", "awesome", "perfect", "love it", "amazing", "wonderful"]
        if any(w in text_lower for w in happy_words):
            return "positive", "happy"

        # Excitement indicators
        excitement_words = ["excited", "can't wait", "awesome", "amazing", "incredible"]
        if any(w in text_lower for w in excitement_words):
            return "positive", "excited"

        # Anxiety indicators
        anxiety_words = ["worried", "anxious", "nervous", "scared", "afraid", "concerned about"]
        if any(w in text_lower for w in anxiety_words):
            return "negative", "anxious"

        return "neutral", "neutral"


class ContextAwareSpeaker:
    """
    Speaker that automatically adjusts prosody based on context.

    Usage:
        speaker = ContextAwareSpeaker()

        # With automatic emotion
        await speaker.speak(
            "I understand how frustrating that must be.",
            context="User complained about a bug",
            auto_emotion=True
        )

        # With manual hint
        await speaker.speak(
            "Great news! We fixed the issue.",
            hint=ProsodyHint.EXCITED
        )
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        context: Optional[ConversationContext] = None,
    ):
        self.config = config or ContextConfig()
        self.context = context or ConversationContext()
        self.emotion_selector = EmotionSelector(self.config)

        # Engine reference (lazy loaded)
        self._engine = None

    def _get_engine(self):
        """Get or create TTS engine."""
        if self._engine is None:
            from voice_soundboard import VoiceEngine
            self._engine = VoiceEngine()
        return self._engine

    def update_context(self, user_message: str) -> None:
        """
        Update context with a new user message.

        Analyzes sentiment and emotion automatically.
        """
        self.context.add_message("user", user_message)

        # Detect user sentiment
        sentiment, emotion = self.emotion_selector.detect_user_sentiment(user_message)
        self.context.user_sentiment = sentiment
        self.context.user_emotion = emotion

    def speak(
        self,
        text: str,
        context: Optional[str] = None,
        hint: Optional[ProsodyHint] = None,
        auto_emotion: bool = True,
        voice: Optional[str] = None,
        preset: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> dict:
        """
        Speak text with context-aware prosody.

        Args:
            text: Text to speak
            context: Optional context string (e.g., "User expressed frustration")
            hint: Manual prosody hint
            auto_emotion: Whether to automatically select emotion
            voice: Override voice
            preset: Override preset
            speed: Override speed

        Returns:
            Dictionary with speaking result and selected emotion
        """
        selected_emotion = self.config.default_emotion
        confidence = 0.0
        speed_factor = 1.0

        # Process context string if provided
        if context:
            # Treat context as a pseudo user message for analysis
            _, user_emotion = self.emotion_selector.detect_user_sentiment(context)
            self.context.user_emotion = user_emotion

        # Select emotion
        if auto_emotion and self.config.enable_auto_emotion:
            selected_emotion, confidence = self.emotion_selector.select_emotion(
                text, self.context, hint
            )

            # Adjust speed based on emotion
            if selected_emotion in ["sympathetic", "calm", "serious"]:
                speed_factor = self.config.empathy_speed_factor
            elif selected_emotion in ["excited", "happy"]:
                speed_factor = self.config.excitement_speed_factor

        # Apply hint override
        if hint and not auto_emotion:
            hint_emotion = self.emotion_selector._hint_to_emotion(hint)
            if hint_emotion:
                selected_emotion = hint_emotion

        # Calculate final speed
        final_speed = (speed or 1.0) * speed_factor

        # Get engine and speak
        engine = self._get_engine()

        try:
            from voice_soundboard import get_emotion_params
            emotion_params = get_emotion_params(selected_emotion)
        except (ImportError, ValueError):
            emotion_params = None

        result = engine.speak(
            text,
            voice=voice,
            preset=preset or self.config.default_preset,
            speed=final_speed,
        )

        # Record in context
        self.context.add_message("assistant", text)

        return {
            "result": result,
            "emotion": selected_emotion,
            "confidence": confidence,
            "speed_factor": speed_factor,
            "final_speed": final_speed,
        }

    async def speak_async(
        self,
        text: str,
        context: Optional[str] = None,
        hint: Optional[ProsodyHint] = None,
        auto_emotion: bool = True,
        **kwargs,
    ) -> dict:
        """Async version of speak."""
        # For now, wrap sync version
        # In future, could use async streaming
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.speak(text, context, hint, auto_emotion, **kwargs)
        )

    def reset_context(self) -> None:
        """Reset the conversation context."""
        self.context = ConversationContext()
