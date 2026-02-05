"""
Voice Studio - Interactive Preset Creation and Tuning.

.. warning:: **Experimental.** This module may change or be removed between
   minor releases. See docs/API_STABILITY.md and docs/FEATURE_FLAGS.md.

Provides tools for creating and tuning voice presets with:
- Real-time audio preview
- AI-assisted natural language voice design
- Parameter adjustment with undo/redo
- Preset saving and management
"""

from .session import VoiceStudioSession
from .engine import VoiceStudioEngine
from .ai_assistant import StudioAIAssistant

__all__ = [
    "VoiceStudioSession",
    "VoiceStudioEngine",
    "StudioAIAssistant",
]
