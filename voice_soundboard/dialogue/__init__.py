"""
Multi-Speaker Dialogue Module.

Provides script parsing and multi-speaker synthesis for dialogue-based TTS.

Script Format:
    [S1:narrator] The door creaked open slowly.
    [S2:alice] Hello? Is anyone there? [gasp]
    [S3:bob] (whispering) Don't go in there...
    [S1:narrator] But she didn't listen.

Stage Directions:
    (whispering), (shouting), (sadly), etc. - parsed and applied as emotion hints

Features:
- Automatic speaker turn detection
- Voice auto-assignment based on speaker characteristics
- Stage direction parsing for emotion control
- Paralinguistic tag support (inherits from Chatterbox)
- Seamless audio concatenation with configurable pauses

Example:
    from voice_soundboard.dialogue import DialogueParser, DialogueEngine

    parser = DialogueParser()
    script = parser.parse(script_text)

    engine = DialogueEngine()
    result = engine.synthesize(script, voices={"narrator": "bm_george"})
"""

from voice_soundboard.dialogue.parser import (
    DialogueParser,
    DialogueLine,
    Speaker,
    StageDirection,
    ParsedScript,
)
from voice_soundboard.dialogue.engine import (
    DialogueEngine,
    DialogueResult,
    SpeakerTurn,
)
from voice_soundboard.dialogue.voices import (
    VoiceAssigner,
    VoiceCharacteristics,
    auto_assign_voices,
)

__all__ = [
    # Parser
    "DialogueParser",
    "DialogueLine",
    "Speaker",
    "StageDirection",
    "ParsedScript",
    # Engine
    "DialogueEngine",
    "DialogueResult",
    "SpeakerTurn",
    # Voice Assignment
    "VoiceAssigner",
    "VoiceCharacteristics",
    "auto_assign_voices",
]
