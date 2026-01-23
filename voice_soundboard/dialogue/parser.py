"""
Dialogue Script Parser.

Parses multi-speaker dialogue scripts into structured data for synthesis.

Script Format:
    [S1:narrator] The door creaked open slowly.
    [S2:alice] Hello? Is anyone there? [gasp]
    [S3:bob] (whispering) Don't go in there...

Supports:
- Speaker tags: [S1:name], [S2:name], etc.
- Stage directions: (whispering), (shouting), (sadly)
- Paralinguistic tags: [laugh], [sigh], [gasp], etc.
- Continuations: Lines without speaker tags continue previous speaker
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class StageDirectionType(Enum):
    """Types of stage directions that affect synthesis."""
    EMOTION = "emotion"       # (sadly), (angrily), (happily)
    VOLUME = "volume"         # (whispering), (shouting), (softly)
    PACE = "pace"             # (slowly), (quickly), (hesitantly)
    ACTION = "action"         # (pauses), (clears throat), (sighs)


@dataclass
class StageDirection:
    """A parsed stage direction from the script."""
    text: str                           # Original text, e.g., "whispering"
    direction_type: StageDirectionType = StageDirectionType.EMOTION  # Category of direction
    intensity: float = 0.5              # 0.0-1.0 intensity hint

    # Mapping hints for synthesis parameters
    speed_modifier: float = 1.0         # Multiplier for speech speed
    volume_modifier: float = 1.0        # Multiplier for volume
    emotion_hint: Optional[str] = None  # Emotion to apply

    def __post_init__(self):
        """Derive synthesis parameters from direction text."""
        text_lower = self.text.lower()

        # Volume directions
        if text_lower in ("whispering", "whispered", "whisper"):
            self.direction_type = StageDirectionType.VOLUME
            self.volume_modifier = 0.4
            self.speed_modifier = 0.85
            self.emotion_hint = "whisper"
        elif text_lower in ("shouting", "yelling", "screaming", "loud"):
            self.direction_type = StageDirectionType.VOLUME
            self.volume_modifier = 1.5
            self.intensity = 0.8
        elif text_lower in ("softly", "quietly", "gently"):
            self.direction_type = StageDirectionType.VOLUME
            self.volume_modifier = 0.6
            self.speed_modifier = 0.9

        # Pace directions
        elif text_lower in ("slowly", "deliberate", "measured"):
            self.direction_type = StageDirectionType.PACE
            self.speed_modifier = 0.7
        elif text_lower in ("quickly", "rapidly", "fast", "rushed"):
            self.direction_type = StageDirectionType.PACE
            self.speed_modifier = 1.4
        elif text_lower in ("hesitantly", "uncertain", "nervous"):
            self.direction_type = StageDirectionType.PACE
            self.speed_modifier = 0.8
            self.emotion_hint = "nervous"

        # Emotion directions
        elif text_lower in ("sadly", "sorrowfully", "mournfully"):
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = "sad"
            self.speed_modifier = 0.85
        elif text_lower in ("happily", "joyfully", "cheerfully"):
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = "happy"
            self.speed_modifier = 1.1
        elif text_lower in ("angrily", "furiously", "enraged"):
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = "angry"
            self.intensity = 0.8
        elif text_lower in ("fearfully", "scared", "terrified"):
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = "fearful"
            self.speed_modifier = 1.15
        elif text_lower in ("sarcastically", "mockingly", "dryly"):
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = "sarcastic"
        elif text_lower in ("excitedly", "enthusiastically"):
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = "excited"
            self.speed_modifier = 1.2

        # Action directions (often map to paralinguistic tags)
        elif text_lower in ("pauses", "pause", "beat"):
            self.direction_type = StageDirectionType.ACTION
        elif text_lower in ("sighs", "sighing"):
            self.direction_type = StageDirectionType.ACTION
            self.emotion_hint = "sad"
        elif text_lower in ("laughs", "laughing", "chuckling"):
            self.direction_type = StageDirectionType.ACTION
            self.emotion_hint = "happy"
        else:
            # Default to emotion type for unknown directions
            self.direction_type = StageDirectionType.EMOTION
            self.emotion_hint = text_lower


@dataclass
class Speaker:
    """A speaker in the dialogue."""
    id: str                             # Unique ID, e.g., "S1", "S2"
    name: str                           # Character name, e.g., "narrator", "alice"
    voice: Optional[str] = None         # Assigned voice ID
    default_emotion: Optional[str] = None  # Default emotion for this speaker

    # Characteristics hints for auto-assignment
    gender_hint: Optional[str] = None   # "male", "female", or None
    age_hint: Optional[str] = None      # "young", "adult", "elderly", or None
    accent_hint: Optional[str] = None   # "american", "british", etc.

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Speaker):
            return self.id == other.id
        return False


@dataclass
class DialogueLine:
    """A single line of dialogue."""
    speaker: Speaker                    # Who speaks this line
    text: str                           # The dialogue text (cleaned)
    raw_text: str                       # Original text with all tags
    stage_directions: List[StageDirection] = field(default_factory=list)
    line_number: int = 0                # Line number in original script

    # Synthesis hints derived from parsing
    emotion: Optional[str] = None       # Resolved emotion for this line
    speed: float = 1.0                  # Speed modifier
    pause_before_ms: int = 0            # Pause before this line
    pause_after_ms: int = 300           # Pause after this line (between speakers)

    def has_stage_directions(self) -> bool:
        """Check if line has any stage directions."""
        return len(self.stage_directions) > 0

    def get_primary_emotion(self) -> Optional[str]:
        """Get the primary emotion from stage directions."""
        for direction in self.stage_directions:
            if direction.emotion_hint:
                return direction.emotion_hint
        return self.emotion


@dataclass
class ParsedScript:
    """A fully parsed dialogue script."""
    lines: List[DialogueLine]           # All dialogue lines in order
    speakers: Dict[str, Speaker]        # All speakers by ID
    title: Optional[str] = None         # Optional script title
    metadata: Dict[str, str] = field(default_factory=dict)

    def get_speaker_lines(self, speaker_id: str) -> List[DialogueLine]:
        """Get all lines for a specific speaker."""
        return [line for line in self.lines if line.speaker.id == speaker_id]

    def get_speaker_names(self) -> List[str]:
        """Get list of all speaker names."""
        return [s.name for s in self.speakers.values()]

    def speaker_count(self) -> int:
        """Get number of unique speakers."""
        return len(self.speakers)

    def line_count(self) -> int:
        """Get total number of lines."""
        return len(self.lines)


class DialogueParser:
    """
    Parser for multi-speaker dialogue scripts.

    Supports flexible script formats with speaker tags, stage directions,
    and paralinguistic tags.

    Example:
        parser = DialogueParser()
        script = parser.parse('''
            [S1:narrator] The room fell silent.
            [S2:detective] (slowly) Where were you last night?
            [S3:suspect] (nervously) I... I was at home. [cough]
        ''')
    """

    # Pattern for speaker tags: [S1:name] or [speaker:name]
    SPEAKER_PATTERN = re.compile(
        r'\[(?:S(\d+)|speaker):?\s*([^\]]+)\]',
        re.IGNORECASE
    )

    # Alternative format: NAME: dialogue
    SIMPLE_SPEAKER_PATTERN = re.compile(
        r'^([A-Z][A-Z0-9_]*)\s*:\s*(.+)$',
        re.MULTILINE
    )

    # Stage directions in parentheses
    STAGE_DIRECTION_PATTERN = re.compile(
        r'\(([^)]+)\)',
        re.IGNORECASE
    )

    # Metadata pattern: #key: value (with optional leading whitespace)
    METADATA_PATTERN = re.compile(
        r'^\s*#\s*(\w+)\s*:\s*(.+)$',
        re.MULTILINE
    )

    def __init__(
        self,
        default_pause_between_speakers_ms: int = 400,
        default_pause_same_speaker_ms: int = 200,
        parse_simple_format: bool = True,
    ):
        """
        Initialize the parser.

        Args:
            default_pause_between_speakers_ms: Default pause when speaker changes
            default_pause_same_speaker_ms: Default pause for same speaker continuation
            parse_simple_format: Also parse "NAME: dialogue" format
        """
        self.default_pause_between_speakers = default_pause_between_speakers_ms
        self.default_pause_same_speaker = default_pause_same_speaker_ms
        self.parse_simple_format = parse_simple_format

    def parse(self, script_text: str) -> ParsedScript:
        """
        Parse a dialogue script into structured data.

        Args:
            script_text: The raw script text

        Returns:
            ParsedScript with all lines, speakers, and metadata
        """
        # Extract metadata
        metadata = self._extract_metadata(script_text)
        title = metadata.pop("title", None)

        # Parse lines
        lines: List[DialogueLine] = []
        speakers: Dict[str, Speaker] = {}
        current_speaker: Optional[Speaker] = None

        for line_num, raw_line in enumerate(script_text.split('\n'), 1):
            line = raw_line.strip()

            # Skip empty lines and metadata
            if not line or line.startswith('#'):
                continue

            # Try to parse speaker tag
            speaker, remaining_text = self._parse_speaker_tag(line, speakers)

            if speaker is None and self.parse_simple_format:
                # Try simple format: NAME: dialogue
                speaker, remaining_text = self._parse_simple_format(line, speakers)

            if speaker is None:
                # Continuation of previous speaker
                if current_speaker is None:
                    # No speaker yet, create default narrator
                    current_speaker = Speaker(id="S1", name="narrator")
                    speakers["S1"] = current_speaker
                speaker = current_speaker
                remaining_text = line
            else:
                current_speaker = speaker

            # Parse stage directions
            stage_directions, cleaned_text = self._parse_stage_directions(remaining_text)

            # Calculate pauses
            pause_before = 0
            if lines:
                if lines[-1].speaker.id != speaker.id:
                    pause_before = self.default_pause_between_speakers
                else:
                    pause_before = self.default_pause_same_speaker

            # Resolve emotion from stage directions
            emotion = None
            speed = 1.0
            for direction in stage_directions:
                if direction.emotion_hint:
                    emotion = direction.emotion_hint
                speed *= direction.speed_modifier

            # Create dialogue line
            dialogue_line = DialogueLine(
                speaker=speaker,
                text=cleaned_text.strip(),
                raw_text=line,
                stage_directions=stage_directions,
                line_number=line_num,
                emotion=emotion,
                speed=speed,
                pause_before_ms=pause_before,
            )

            lines.append(dialogue_line)

        return ParsedScript(
            lines=lines,
            speakers=speakers,
            title=title,
            metadata=metadata,
        )

    def _extract_metadata(self, script_text: str) -> Dict[str, str]:
        """Extract metadata from script comments."""
        metadata = {}
        for match in self.METADATA_PATTERN.finditer(script_text):
            key = match.group(1).lower()
            value = match.group(2).strip()
            metadata[key] = value
        return metadata

    def _parse_speaker_tag(
        self,
        line: str,
        speakers: Dict[str, Speaker]
    ) -> Tuple[Optional[Speaker], str]:
        """Parse [S1:name] style speaker tags."""
        match = self.SPEAKER_PATTERN.match(line)
        if not match:
            return None, line

        # Extract speaker info
        speaker_num = match.group(1)
        speaker_name = match.group(2).strip().lower()

        if speaker_num:
            speaker_id = f"S{speaker_num}"
        else:
            # Generate ID from name
            speaker_id = f"S{len(speakers) + 1}"

        # Get or create speaker
        if speaker_id not in speakers:
            speaker = Speaker(id=speaker_id, name=speaker_name)
            # Infer gender from common names
            speaker.gender_hint = self._infer_gender(speaker_name)
            speakers[speaker_id] = speaker
        else:
            speaker = speakers[speaker_id]
            # Update name if different (allows aliases)
            if speaker.name != speaker_name:
                speaker.name = speaker_name

        # Return speaker and remaining text
        remaining = line[match.end():].strip()
        return speaker, remaining

    def _parse_simple_format(
        self,
        line: str,
        speakers: Dict[str, Speaker]
    ) -> Tuple[Optional[Speaker], str]:
        """Parse NAME: dialogue format."""
        match = self.SIMPLE_SPEAKER_PATTERN.match(line)
        if not match:
            return None, line

        speaker_name = match.group(1).lower()
        dialogue = match.group(2)

        # Find or create speaker by name
        for speaker in speakers.values():
            if speaker.name == speaker_name:
                return speaker, dialogue

        # Create new speaker
        speaker_id = f"S{len(speakers) + 1}"
        speaker = Speaker(id=speaker_id, name=speaker_name)
        speaker.gender_hint = self._infer_gender(speaker_name)
        speakers[speaker_id] = speaker

        return speaker, dialogue

    def _parse_stage_directions(
        self,
        text: str
    ) -> Tuple[List[StageDirection], str]:
        """Extract stage directions from text."""
        directions = []
        cleaned_text = text

        for match in self.STAGE_DIRECTION_PATTERN.finditer(text):
            direction_text = match.group(1).strip()
            direction = StageDirection(text=direction_text)
            directions.append(direction)

        # Remove stage directions from text
        cleaned_text = self.STAGE_DIRECTION_PATTERN.sub('', text)

        return directions, cleaned_text

    def _infer_gender(self, name: str) -> Optional[str]:
        """Infer gender from common character names/roles."""
        name_lower = name.lower()

        # Common female names/roles
        female_hints = {
            "alice", "emma", "sarah", "mary", "lisa", "anna", "bella",
            "queen", "princess", "mother", "grandmother", "aunt", "sister",
            "witch", "fairy", "goddess", "lady", "madame", "mrs", "miss",
        }

        # Common male names/roles
        male_hints = {
            "bob", "john", "michael", "david", "james", "george", "adam",
            "king", "prince", "father", "grandfather", "uncle", "brother",
            "wizard", "knight", "lord", "sir", "mr", "mister",
        }

        # Neutral roles
        neutral_hints = {"narrator", "announcer", "voice", "speaker"}

        if name_lower in female_hints:
            return "female"
        elif name_lower in male_hints:
            return "male"
        elif name_lower in neutral_hints:
            return None

        # Check if name ends with common gender suffixes
        if name_lower.endswith(("a", "ie", "y", "ine", "elle")):
            return "female"

        return None


def parse_dialogue(script_text: str) -> ParsedScript:
    """
    Convenience function to parse a dialogue script.

    Args:
        script_text: Raw dialogue script

    Returns:
        ParsedScript object
    """
    parser = DialogueParser()
    return parser.parse(script_text)
