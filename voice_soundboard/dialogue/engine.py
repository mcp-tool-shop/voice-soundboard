"""
Dialogue Synthesis Engine.

Synthesizes multi-speaker dialogue scripts into seamless audio.

Features:
- Multi-speaker synthesis with automatic voice assignment
- Stage direction interpretation (emotion, pacing)
- Configurable pauses between speakers
- Seamless audio concatenation
- Optional per-line audio output
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import numpy as np
import soundfile as sf

from voice_soundboard.dialogue.parser import (
    DialogueParser,
    ParsedScript,
    DialogueLine,
    Speaker,
)
from voice_soundboard.dialogue.voices import (
    VoiceAssigner,
    auto_assign_voices,
)
from voice_soundboard.engine import VoiceEngine
from voice_soundboard.config import Config


@dataclass
class SpeakerTurn:
    """Result for a single speaker's turn."""
    speaker_name: str
    text: str
    voice_id: str
    audio_path: Optional[Path] = None
    audio_samples: Optional[np.ndarray] = None
    duration_seconds: float = 0.0
    emotion: Optional[str] = None
    speed: float = 1.0
    line_number: int = 0


@dataclass
class DialogueResult:
    """Result of dialogue synthesis."""
    audio_path: Path                    # Path to concatenated audio
    duration_seconds: float             # Total duration
    turns: List[SpeakerTurn]            # Individual speaker turns
    speaker_count: int                  # Number of unique speakers
    line_count: int                     # Number of dialogue lines
    sample_rate: int = 24000            # Audio sample rate
    voice_assignments: Dict[str, str] = field(default_factory=dict)

    def get_speaker_duration(self, speaker_name: str) -> float:
        """Get total speaking time for a speaker."""
        return sum(
            turn.duration_seconds
            for turn in self.turns
            if turn.speaker_name == speaker_name
        )


class DialogueEngine:
    """
    Engine for synthesizing multi-speaker dialogue.

    Example:
        engine = DialogueEngine()
        script = '''
            [S1:narrator] The detective entered the room.
            [S2:detective] (firmly) Where were you last night?
            [S3:suspect] (nervously) I... I was at home.
        '''
        result = engine.synthesize(script, voices={"narrator": "bm_george"})
    """

    def __init__(
        self,
        voice_engine: Optional[VoiceEngine] = None,
        config: Optional[Config] = None,
        default_pause_ms: int = 400,
        narrator_pause_ms: int = 600,
        sample_rate: int = 24000,
    ):
        """
        Initialize the dialogue engine.

        Args:
            voice_engine: VoiceEngine for TTS (created if not provided)
            config: Config object
            default_pause_ms: Default pause between speakers
            narrator_pause_ms: Pause after narrator lines
            sample_rate: Audio sample rate
        """
        self.config = config or Config()
        self.voice_engine = voice_engine
        self.default_pause_ms = default_pause_ms
        self.narrator_pause_ms = narrator_pause_ms
        self.sample_rate = sample_rate
        self.parser = DialogueParser()
        self.voice_assigner = VoiceAssigner()

    def _get_engine(self) -> VoiceEngine:
        """Get or create the voice engine."""
        if self.voice_engine is None:
            self.voice_engine = VoiceEngine(self.config)
        return self.voice_engine

    def synthesize(
        self,
        script: Union[str, ParsedScript],
        voices: Optional[Dict[str, str]] = None,
        output_path: Optional[Path] = None,
        save_individual_turns: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> DialogueResult:
        """
        Synthesize a dialogue script into audio.

        Args:
            script: Script text or ParsedScript object
            voices: Voice assignments {speaker_name: voice_id}
            output_path: Path to save concatenated audio
            save_individual_turns: Save each turn as separate file
            progress_callback: Callback(current, total, speaker_name)

        Returns:
            DialogueResult with audio and metadata
        """
        # Parse script if string
        if isinstance(script, str):
            parsed = self.parser.parse(script)
        else:
            parsed = script

        # Auto-assign voices
        voice_assignments = self.voice_assigner.assign_voices(parsed, voices)

        # Synthesize each line
        engine = self._get_engine()
        turns: List[SpeakerTurn] = []
        all_samples: List[np.ndarray] = []

        for i, line in enumerate(parsed.lines):
            if progress_callback:
                progress_callback(i + 1, len(parsed.lines), line.speaker.name)

            # Get voice for this speaker
            voice_id = voice_assignments.get(line.speaker.name, "af_heart")

            # Determine speed and emotion from stage directions
            speed = line.speed
            emotion = line.get_primary_emotion()

            # Add pause before this line
            if line.pause_before_ms > 0:
                pause_samples = self._generate_silence(line.pause_before_ms)
                all_samples.append(pause_samples)

            # Synthesize the line
            try:
                samples, sr = engine.speak_raw(
                    text=line.text,
                    voice=voice_id,
                    speed=speed,
                )
            except Exception as e:
                # Skip lines that fail to synthesize
                print(f"Warning: Failed to synthesize line {i+1}: {e}")
                continue

            # Create turn result
            turn = SpeakerTurn(
                speaker_name=line.speaker.name,
                text=line.text,
                voice_id=voice_id,
                audio_samples=samples if save_individual_turns else None,
                duration_seconds=len(samples) / sr,
                emotion=emotion,
                speed=speed,
                line_number=line.line_number,
            )

            # Save individual turn if requested
            if save_individual_turns and output_path:
                turn_path = output_path.parent / f"{output_path.stem}_turn_{i+1:03d}.wav"
                sf.write(str(turn_path), samples, sr)
                turn.audio_path = turn_path

            turns.append(turn)
            all_samples.append(samples)

            # Add extra pause after narrator
            if line.speaker.name.lower() == "narrator":
                extra_pause = self.narrator_pause_ms - self.default_pause_ms
                if extra_pause > 0:
                    all_samples.append(self._generate_silence(extra_pause))

        # Concatenate all audio
        if all_samples:
            concatenated = np.concatenate(all_samples)
        else:
            concatenated = np.zeros(self.sample_rate)  # 1 second of silence

        # Determine output path
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))

        # Save concatenated audio
        sf.write(str(output_path), concatenated, self.sample_rate)

        # Calculate total duration
        total_duration = len(concatenated) / self.sample_rate

        return DialogueResult(
            audio_path=output_path,
            duration_seconds=total_duration,
            turns=turns,
            speaker_count=parsed.speaker_count(),
            line_count=len(turns),
            sample_rate=self.sample_rate,
            voice_assignments=voice_assignments,
        )

    def synthesize_streaming(
        self,
        script: Union[str, ParsedScript],
        voices: Optional[Dict[str, str]] = None,
        on_turn_complete: Optional[Callable[[SpeakerTurn], None]] = None,
    ):
        """
        Synthesize dialogue with streaming output.

        Yields audio for each turn as it's generated.

        Args:
            script: Script text or ParsedScript
            voices: Voice assignments
            on_turn_complete: Callback for each completed turn

        Yields:
            SpeakerTurn objects as they're synthesized
        """
        # Parse script if string
        if isinstance(script, str):
            parsed = self.parser.parse(script)
        else:
            parsed = script

        # Auto-assign voices
        voice_assignments = self.voice_assigner.assign_voices(parsed, voices)

        # Synthesize each line
        engine = self._get_engine()

        for i, line in enumerate(parsed.lines):
            voice_id = voice_assignments.get(line.speaker.name, "af_heart")
            speed = line.speed
            emotion = line.get_primary_emotion()

            try:
                samples, sr = engine.speak_raw(
                    text=line.text,
                    voice=voice_id,
                    speed=speed,
                )

                turn = SpeakerTurn(
                    speaker_name=line.speaker.name,
                    text=line.text,
                    voice_id=voice_id,
                    audio_samples=samples,
                    duration_seconds=len(samples) / sr,
                    emotion=emotion,
                    speed=speed,
                    line_number=line.line_number,
                )

                if on_turn_complete:
                    on_turn_complete(turn)

                yield turn

            except Exception as e:
                print(f"Warning: Failed to synthesize line {i+1}: {e}")
                continue

    def preview_assignments(
        self,
        script: Union[str, ParsedScript],
        voices: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Preview voice assignments without synthesizing.

        Args:
            script: Script text or ParsedScript
            voices: Voice overrides

        Returns:
            Dict mapping speaker names to assigned voices
        """
        if isinstance(script, str):
            parsed = self.parser.parse(script)
        else:
            parsed = script

        return self.voice_assigner.assign_voices(parsed, voices)

    def _generate_silence(self, duration_ms: int) -> np.ndarray:
        """Generate silence of specified duration."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)

    def get_script_info(
        self,
        script: Union[str, ParsedScript],
    ) -> Dict:
        """
        Get information about a script without synthesizing.

        Args:
            script: Script text or ParsedScript

        Returns:
            Dict with script statistics
        """
        if isinstance(script, str):
            parsed = self.parser.parse(script)
        else:
            parsed = script

        # Count lines per speaker
        speaker_lines = {}
        for line in parsed.lines:
            name = line.speaker.name
            speaker_lines[name] = speaker_lines.get(name, 0) + 1

        # Count words
        total_words = sum(len(line.text.split()) for line in parsed.lines)

        # Estimate duration (rough: 150 words per minute)
        estimated_duration = total_words / 150 * 60

        return {
            "speaker_count": parsed.speaker_count(),
            "line_count": parsed.line_count(),
            "speakers": list(parsed.speakers.keys()),
            "speaker_names": parsed.get_speaker_names(),
            "speaker_lines": speaker_lines,
            "total_words": total_words,
            "estimated_duration_seconds": estimated_duration,
            "title": parsed.title,
            "metadata": parsed.metadata,
        }


def synthesize_dialogue(
    script: str,
    voices: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
) -> DialogueResult:
    """
    Convenience function for dialogue synthesis.

    Args:
        script: Dialogue script text
        voices: Voice assignments
        output_path: Output audio path

    Returns:
        DialogueResult
    """
    engine = DialogueEngine()
    return engine.synthesize(script, voices, output_path)
