"""
Sound Effects Library.

Generates various sound effects programmatically (no external files needed).
Includes chimes, alerts, ambient sounds, and UI feedback sounds.
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import soundfile as sf

from voice_soundboard.audio import play_audio


# Standard sample rate for all effects
SAMPLE_RATE = 44100


@dataclass
class SoundEffect:
    """A generated sound effect."""
    name: str
    samples: np.ndarray
    sample_rate: int
    duration: float

    def save(self, path: Path) -> Path:
        """Save effect to WAV file."""
        sf.write(str(path), self.samples, self.sample_rate)
        return path

    def play(self) -> None:
        """Play the effect through speakers."""
        # Save to temp and play
        temp_path = Path("F:/AI/voice-soundboard/output") / f"_temp_{self.name}.wav"
        self.save(temp_path)
        play_audio(temp_path)


def _envelope(samples: np.ndarray, attack: float = 0.01, decay: float = 0.1) -> np.ndarray:
    """Apply attack/decay envelope to samples."""
    length = len(samples)
    attack_samples = int(attack * SAMPLE_RATE)
    decay_samples = int(decay * SAMPLE_RATE)

    envelope = np.ones(length)

    # Attack ramp
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay ramp
    if decay_samples > 0 and decay_samples < length:
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

    return samples * envelope


def _generate_tone(freq: float, duration: float, wave: str = "sine") -> np.ndarray:
    """Generate a basic tone."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    if wave == "sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave == "square":
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == "triangle":
        return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif wave == "sawtooth":
        return 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        return np.sin(2 * np.pi * freq * t)


# =============================================================================
# CHIMES & BELLS
# =============================================================================

def chime_notification() -> SoundEffect:
    """Pleasant notification chime - two ascending tones."""
    duration = 0.3

    # Two harmonious notes (C5 and E5)
    tone1 = _generate_tone(523.25, duration) * 0.6  # C5
    tone2 = _generate_tone(659.25, duration) * 0.4  # E5

    # Offset second tone slightly
    combined = np.zeros(int(SAMPLE_RATE * 0.5))
    combined[:len(tone1)] += _envelope(tone1, 0.01, 0.2)
    offset = int(0.15 * SAMPLE_RATE)
    combined[offset:offset + len(tone2)] += _envelope(tone2, 0.01, 0.25)

    return SoundEffect("chime_notification", combined.astype(np.float32), SAMPLE_RATE, 0.5)


def chime_success() -> SoundEffect:
    """Success/completion chime - ascending arpeggio."""
    notes = [523.25, 659.25, 783.99]  # C5, E5, G5
    duration_per = 0.12

    samples = []
    for i, freq in enumerate(notes):
        tone = _generate_tone(freq, duration_per) * (0.7 - i * 0.1)
        tone = _envelope(tone, 0.005, 0.08)
        samples.append(tone)
        if i < len(notes) - 1:
            samples.append(np.zeros(int(0.05 * SAMPLE_RATE)))  # Gap

    combined = np.concatenate(samples)
    return SoundEffect("chime_success", combined.astype(np.float32), SAMPLE_RATE, len(combined) / SAMPLE_RATE)


def chime_error() -> SoundEffect:
    """Error/failure sound - descending minor tones."""
    notes = [440, 349.23]  # A4, F4 (minor feel)
    duration_per = 0.15

    samples = []
    for freq in notes:
        tone = _generate_tone(freq, duration_per) * 0.6
        tone = _envelope(tone, 0.01, 0.1)
        samples.append(tone)

    combined = np.concatenate(samples)
    return SoundEffect("chime_error", combined.astype(np.float32), SAMPLE_RATE, len(combined) / SAMPLE_RATE)


def chime_attention() -> SoundEffect:
    """Attention-getting chime - bright ping."""
    freq = 880  # A5
    duration = 0.25

    tone = _generate_tone(freq, duration)
    # Add harmonics for brightness
    tone += _generate_tone(freq * 2, duration) * 0.3
    tone += _generate_tone(freq * 3, duration) * 0.1

    tone = _envelope(tone * 0.5, 0.001, 0.2)
    return SoundEffect("chime_attention", tone.astype(np.float32), SAMPLE_RATE, duration)


# =============================================================================
# UI SOUNDS
# =============================================================================

def click() -> SoundEffect:
    """Subtle click sound for UI interactions."""
    duration = 0.02
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Short noise burst with quick decay
    noise = np.random.randn(len(t)) * 0.3
    envelope = np.exp(-t * 200)
    click = noise * envelope

    return SoundEffect("click", click.astype(np.float32), SAMPLE_RATE, duration)


def pop() -> SoundEffect:
    """Soft pop sound."""
    duration = 0.05
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Filtered noise with envelope
    freq = 400
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 80)

    return SoundEffect("pop", (tone * 0.5).astype(np.float32), SAMPLE_RATE, duration)


def whoosh() -> SoundEffect:
    """Whoosh/swipe sound."""
    duration = 0.2
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Noise with bandpass sweep
    noise = np.random.randn(len(t))

    # Simple envelope
    envelope = np.sin(np.pi * t / duration) ** 2
    whoosh = noise * envelope * 0.3

    return SoundEffect("whoosh", whoosh.astype(np.float32), SAMPLE_RATE, duration)


# =============================================================================
# ALERTS
# =============================================================================

def alert_warning() -> SoundEffect:
    """Warning alert - two-tone beep."""
    freq1, freq2 = 800, 600
    duration = 0.15

    tone1 = _generate_tone(freq1, duration)
    tone2 = _generate_tone(freq2, duration)

    combined = np.concatenate([
        _envelope(tone1 * 0.5, 0.01, 0.05),
        _envelope(tone2 * 0.5, 0.01, 0.1)
    ])

    return SoundEffect("alert_warning", combined.astype(np.float32), SAMPLE_RATE, duration * 2)


def alert_critical() -> SoundEffect:
    """Critical alert - urgent repeated beeps."""
    freq = 1000
    beep_duration = 0.1
    gap = 0.05
    repeats = 3

    beep = _envelope(_generate_tone(freq, beep_duration) * 0.6, 0.005, 0.05)
    gap_samples = np.zeros(int(gap * SAMPLE_RATE))

    samples = []
    for i in range(repeats):
        samples.append(beep)
        if i < repeats - 1:
            samples.append(gap_samples)

    combined = np.concatenate(samples)
    return SoundEffect("alert_critical", combined.astype(np.float32), SAMPLE_RATE, len(combined) / SAMPLE_RATE)


def alert_info() -> SoundEffect:
    """Info notification - gentle single tone."""
    freq = 600
    duration = 0.2

    tone = _generate_tone(freq, duration)
    tone += _generate_tone(freq * 1.5, duration) * 0.2  # Add fifth
    tone = _envelope(tone * 0.4, 0.02, 0.15)

    return SoundEffect("alert_info", tone.astype(np.float32), SAMPLE_RATE, duration)


# =============================================================================
# AMBIENT
# =============================================================================

def ambient_rain(duration: float = 5.0) -> SoundEffect:
    """Gentle rain ambient sound."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Layer of filtered noise
    noise = np.random.randn(len(t))

    # Simple lowpass-like effect through averaging
    window = 100
    rain = np.convolve(noise, np.ones(window) / window, mode='same')

    # Add some variation
    modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    rain = rain * modulation * 0.3

    return SoundEffect("ambient_rain", rain.astype(np.float32), SAMPLE_RATE, duration)


def ambient_white_noise(duration: float = 5.0) -> SoundEffect:
    """White noise for focus/masking."""
    samples = np.random.randn(int(SAMPLE_RATE * duration)) * 0.1
    return SoundEffect("ambient_white_noise", samples.astype(np.float32), SAMPLE_RATE, duration)


def ambient_drone(duration: float = 5.0, base_freq: float = 110) -> SoundEffect:
    """Deep ambient drone."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Base tone with harmonics
    drone = np.sin(2 * np.pi * base_freq * t) * 0.4
    drone += np.sin(2 * np.pi * base_freq * 2 * t) * 0.2
    drone += np.sin(2 * np.pi * base_freq * 3 * t) * 0.1

    # Slow modulation
    mod = 1 + 0.2 * np.sin(2 * np.pi * 0.05 * t)
    drone = drone * mod

    # Fade in/out
    fade = int(0.5 * SAMPLE_RATE)
    drone[:fade] *= np.linspace(0, 1, fade)
    drone[-fade:] *= np.linspace(1, 0, fade)

    return SoundEffect("ambient_drone", drone.astype(np.float32), SAMPLE_RATE, duration)


# =============================================================================
# EFFECT REGISTRY
# =============================================================================

EFFECTS = {
    # Chimes
    "chime": chime_notification,
    "chime_notification": chime_notification,
    "chime_success": chime_success,
    "success": chime_success,
    "chime_error": chime_error,
    "error": chime_error,
    "chime_attention": chime_attention,
    "attention": chime_attention,
    "ding": chime_attention,

    # UI
    "click": click,
    "pop": pop,
    "whoosh": whoosh,

    # Alerts
    "alert_warning": alert_warning,
    "warning": alert_warning,
    "alert_critical": alert_critical,
    "critical": alert_critical,
    "alert_info": alert_info,
    "info": alert_info,

    # Ambient
    "rain": lambda: ambient_rain(5.0),
    "white_noise": lambda: ambient_white_noise(5.0),
    "drone": lambda: ambient_drone(5.0),
}


def get_effect(name: str) -> SoundEffect:
    """Get a sound effect by name."""
    if name not in EFFECTS:
        available = ", ".join(sorted(set(EFFECTS.keys())))
        raise ValueError(f"Unknown effect: {name}. Available: {available}")

    return EFFECTS[name]()


def play_effect(name: str) -> None:
    """Play a sound effect by name."""
    effect = get_effect(name)
    effect.play()


def list_effects() -> list[str]:
    """List all unique effect names."""
    # Remove aliases, keep unique
    seen = set()
    unique = []
    for name, func in EFFECTS.items():
        if func not in seen:
            unique.append(name)
            seen.add(func)
    return sorted(unique)


if __name__ == "__main__":
    print("Sound Effects Demo")
    print("-" * 40)

    output_dir = Path("F:/AI/voice-soundboard/output/effects")
    output_dir.mkdir(exist_ok=True)

    effects_to_demo = [
        "chime_notification",
        "chime_success",
        "chime_error",
        "click",
        "pop",
        "alert_warning",
        "alert_critical",
    ]

    for name in effects_to_demo:
        effect = get_effect(name)
        path = effect.save(output_dir / f"{name}.wav")
        print(f"  {name}: {effect.duration:.2f}s -> {path.name}")

    print(f"\nAll effects saved to: {output_dir}")
    print(f"\nAvailable effects: {', '.join(list_effects())}")
