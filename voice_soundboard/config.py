"""
Configuration for Voice Soundboard.
"""

from pathlib import Path
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Voice Soundboard configuration."""

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("F:/AI/voice-soundboard/output"))
    cache_dir: Path = field(default_factory=lambda: Path("F:/AI/voice-soundboard/.cache"))

    # TTS Engine settings
    device: str = "cuda"  # "cuda" or "cpu"
    default_voice: str = "af_bella"  # Kokoro voice ID
    default_speed: float = 1.0
    sample_rate: int = 24000  # Kokoro outputs 24kHz

    # Performance
    use_gpu: bool = True
    cache_models: bool = True

    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Windows RTX 5080 safety: disable xformers
        os.environ["XFORMERS_DISABLED"] = "1"

        # Auto-detect CUDA availability via ONNX Runtime
        if self.use_gpu:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    self.device = "cuda"
                else:
                    print("CUDA not available in ONNX Runtime, using CPU")
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"


# Available Kokoro voices (from actual model)
KOKORO_VOICES = {
    # American Female
    "af_alloy": {"name": "Alloy", "gender": "female", "accent": "american", "style": "balanced"},
    "af_aoede": {"name": "Aoede", "gender": "female", "accent": "american", "style": "musical"},
    "af_bella": {"name": "Bella", "gender": "female", "accent": "american", "style": "warm"},
    "af_heart": {"name": "Heart", "gender": "female", "accent": "american", "style": "caring"},
    "af_jessica": {"name": "Jessica", "gender": "female", "accent": "american", "style": "professional"},
    "af_kore": {"name": "Kore", "gender": "female", "accent": "american", "style": "youthful"},
    "af_nicole": {"name": "Nicole", "gender": "female", "accent": "american", "style": "soft"},
    "af_nova": {"name": "Nova", "gender": "female", "accent": "american", "style": "bright"},
    "af_river": {"name": "River", "gender": "female", "accent": "american", "style": "calm"},
    "af_sarah": {"name": "Sarah", "gender": "female", "accent": "american", "style": "clear"},
    "af_sky": {"name": "Sky", "gender": "female", "accent": "american", "style": "airy"},

    # American Male
    "am_adam": {"name": "Adam", "gender": "male", "accent": "american", "style": "neutral"},
    "am_echo": {"name": "Echo", "gender": "male", "accent": "american", "style": "resonant"},
    "am_eric": {"name": "Eric", "gender": "male", "accent": "american", "style": "confident"},
    "am_fenrir": {"name": "Fenrir", "gender": "male", "accent": "american", "style": "powerful"},
    "am_liam": {"name": "Liam", "gender": "male", "accent": "american", "style": "friendly"},
    "am_michael": {"name": "Michael", "gender": "male", "accent": "american", "style": "deep"},
    "am_onyx": {"name": "Onyx", "gender": "male", "accent": "american", "style": "smooth"},
    "am_puck": {"name": "Puck", "gender": "male", "accent": "american", "style": "playful"},
    "am_santa": {"name": "Santa", "gender": "male", "accent": "american", "style": "jolly"},

    # British Female
    "bf_alice": {"name": "Alice", "gender": "female", "accent": "british", "style": "proper"},
    "bf_emma": {"name": "Emma", "gender": "female", "accent": "british", "style": "refined"},
    "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "british", "style": "warm"},
    "bf_lily": {"name": "Lily", "gender": "female", "accent": "british", "style": "gentle"},

    # British Male
    "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "british", "style": "sophisticated"},
    "bm_fable": {"name": "Fable", "gender": "male", "accent": "british", "style": "storytelling"},
    "bm_george": {"name": "George", "gender": "male", "accent": "british", "style": "authoritative"},
    "bm_lewis": {"name": "Lewis", "gender": "male", "accent": "british", "style": "friendly"},

    # Other languages
    "jf_alpha": {"name": "Alpha", "gender": "female", "accent": "japanese", "style": "neutral"},
    "zf_xiaobei": {"name": "Xiaobei", "gender": "female", "accent": "mandarin", "style": "clear"},
}

# Voice personality presets (maps to Kokoro voices + settings)
VOICE_PRESETS = {
    "assistant": {
        "voice": "af_bella",
        "speed": 1.0,
        "description": "Friendly, helpful, conversational"
    },
    "narrator": {
        "voice": "bm_george",
        "speed": 0.95,
        "description": "Calm, clear, documentary style"
    },
    "announcer": {
        "voice": "am_michael",
        "speed": 1.1,
        "description": "Bold, energetic, broadcast style"
    },
    "storyteller": {
        "voice": "bf_emma",
        "speed": 0.9,
        "description": "Expressive, varied pacing"
    },
    "whisper": {
        "voice": "af_nicole",
        "speed": 0.85,
        "description": "Soft, intimate, gentle"
    },
}
