"""
Configuration for Voice Soundboard.

All paths default to a ``voice-soundboard/`` directory next to the
project root (or wherever ``VOICE_SOUNDBOARD_DIR`` points).
Override any path via environment variables:

    VOICE_SOUNDBOARD_DIR   - base directory for output/cache/models
    VOICE_SOUNDBOARD_MODELS - model directory (overrides base/models)
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
import os

logger = logging.getLogger(__name__)


def _default_base_dir() -> Path:
    """Resolve the base directory for Voice Soundboard data."""
    env = os.environ.get("VOICE_SOUNDBOARD_DIR")
    if env:
        return Path(env)
    return Path.cwd()


def _default_output_dir() -> Path:
    return _default_base_dir() / "output"


def _default_cache_dir() -> Path:
    return _default_base_dir() / ".cache"


def _default_model_dir() -> Path:
    env = os.environ.get("VOICE_SOUNDBOARD_MODELS")
    if env:
        return Path(env)
    return _default_base_dir() / "models"


@dataclass
class Config:
    """Voice Soundboard configuration.

    All fields have sensible defaults. Override via constructor args
    or environment variables (see module docstring).

    Serializable to dict via ``config.to_dict()``.
    """

    # Paths (resolved from env vars or cwd)
    output_dir: Path = field(default_factory=_default_output_dir)
    cache_dir: Path = field(default_factory=_default_cache_dir)
    model_dir: Path = field(default_factory=_default_model_dir)

    # TTS Engine settings
    device: str = "cuda"  # "cuda" or "cpu"
    default_voice: str = "af_bella"  # Kokoro voice ID
    default_speed: float = 1.0
    sample_rate: int = 24000  # Kokoro outputs 24kHz

    # Performance
    use_gpu: bool = True
    cache_models: bool = True

    def __post_init__(self):
        """Ensure directories exist and detect device."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Windows safety: disable xformers (avoids CUDA errors on some GPUs)
        os.environ.setdefault("XFORMERS_DISABLED", "1")

        # Auto-detect CUDA availability via ONNX Runtime
        if self.use_gpu:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    self.device = "cuda"
                else:
                    logger.info("CUDA not available in ONNX Runtime, using CPU")
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

    def to_dict(self) -> dict:
        """Serialize config to a JSON-compatible dict."""
        return {
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "model_dir": str(self.model_dir),
            "device": self.device,
            "default_voice": self.default_voice,
            "default_speed": self.default_speed,
            "sample_rate": self.sample_rate,
            "use_gpu": self.use_gpu,
            "cache_models": self.cache_models,
        }


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

    # European English Female
    "ef_dora": {"name": "Dora", "gender": "female", "accent": "european", "style": "warm"},

    # European English Male
    "em_alex": {"name": "Alex", "gender": "male", "accent": "european", "style": "neutral"},
    "em_santa": {"name": "Santa", "gender": "male", "accent": "european", "style": "jolly"},

    # French Female
    "ff_siwis": {"name": "Siwis", "gender": "female", "accent": "french", "style": "elegant"},

    # Hindi Female
    "hf_alpha": {"name": "Alpha", "gender": "female", "accent": "hindi", "style": "neutral"},
    "hf_beta": {"name": "Beta", "gender": "female", "accent": "hindi", "style": "warm"},

    # Hindi Male
    "hm_omega": {"name": "Omega", "gender": "male", "accent": "hindi", "style": "deep"},
    "hm_psi": {"name": "Psi", "gender": "male", "accent": "hindi", "style": "clear"},

    # Italian Female
    "if_sara": {"name": "Sara", "gender": "female", "accent": "italian", "style": "expressive"},

    # Italian Male
    "im_nicola": {"name": "Nicola", "gender": "male", "accent": "italian", "style": "warm"},

    # Japanese Female
    "jf_alpha": {"name": "Alpha", "gender": "female", "accent": "japanese", "style": "neutral"},
    "jf_gongitsune": {"name": "Gongitsune", "gender": "female", "accent": "japanese", "style": "storytelling"},
    "jf_nezumi": {"name": "Nezumi", "gender": "female", "accent": "japanese", "style": "soft"},
    "jf_tebukuro": {"name": "Tebukuro", "gender": "female", "accent": "japanese", "style": "gentle"},

    # Japanese Male
    "jm_kumo": {"name": "Kumo", "gender": "male", "accent": "japanese", "style": "calm"},

    # Portuguese Female
    "pf_dora": {"name": "Dora", "gender": "female", "accent": "portuguese", "style": "warm"},

    # Portuguese Male
    "pm_alex": {"name": "Alex", "gender": "male", "accent": "portuguese", "style": "neutral"},
    "pm_santa": {"name": "Santa", "gender": "male", "accent": "portuguese", "style": "jolly"},

    # Mandarin Chinese Female
    "zf_xiaobei": {"name": "Xiaobei", "gender": "female", "accent": "mandarin", "style": "clear"},
    "zf_xiaoni": {"name": "Xiaoni", "gender": "female", "accent": "mandarin", "style": "gentle"},
    "zf_xiaoxiao": {"name": "Xiaoxiao", "gender": "female", "accent": "mandarin", "style": "bright"},
    "zf_xiaoyi": {"name": "Xiaoyi", "gender": "female", "accent": "mandarin", "style": "professional"},
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
