"""
Word-Level Emotion Tag Parser.

Parses inline emotion tags in text for fine-grained emotion control.

Tag Format:
    {emotion}text{/emotion}

Examples:
    I'm {happy}so excited{/happy} to see you!
    {angry}How dare you!{/angry} But {calm}I forgive you{/calm}.

Supports:
- Nested tags: {happy}Great {excited}amazing{/excited} day!{/happy}
- Multiple spans: {sad}Goodbye{/sad} and {happy}hello{/happy}
- Intensity modifiers: {happy:0.8}somewhat happy{/happy}
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class EmotionTagType(Enum):
    """Types of emotion tag boundaries."""
    OPEN = "open"
    CLOSE = "close"


@dataclass
class EmotionSpan:
    """A span of text with an associated emotion."""
    text: str                           # The text content
    emotion: str                        # Emotion name
    intensity: float = 1.0              # 0.0-1.0 intensity
    start_char: int = 0                 # Start position in original text
    end_char: int = 0                   # End position in original text
    start_word: int = 0                 # Start word index
    end_word: int = 0                   # End word index
    nested_in: Optional[str] = None     # Parent emotion if nested

    def word_count(self) -> int:
        """Count words in this span."""
        return len(self.text.split())


@dataclass
class ParsedEmotionText:
    """Result of parsing emotion-tagged text."""
    original_text: str                  # Original text with tags
    plain_text: str                     # Text with tags removed
    spans: List[EmotionSpan]            # Emotion spans
    default_emotion: str = "neutral"    # Emotion for untagged text

    def get_emotion_at_position(self, char_pos: int) -> str:
        """Get the emotion at a character position."""
        for span in self.spans:
            if span.start_char <= char_pos < span.end_char:
                return span.emotion
        return self.default_emotion

    def get_emotion_at_word(self, word_index: int) -> str:
        """Get the emotion at a word index."""
        for span in self.spans:
            if span.start_word <= word_index < span.end_word:
                return span.emotion
        return self.default_emotion

    def get_emotion_timeline(self) -> List[Tuple[float, str]]:
        """
        Get emotions as a timeline (normalized 0-1 positions).

        Returns:
            List of (position, emotion) tuples
        """
        if not self.plain_text:
            return [(0.0, self.default_emotion)]

        total_len = len(self.plain_text)
        timeline = []

        current_pos = 0
        for span in sorted(self.spans, key=lambda s: s.start_char):
            # Add gap before this span if any
            if span.start_char > current_pos:
                timeline.append((current_pos / total_len, self.default_emotion))

            timeline.append((span.start_char / total_len, span.emotion))
            current_pos = span.end_char

        # Add trailing default emotion if needed
        if current_pos < total_len:
            timeline.append((current_pos / total_len, self.default_emotion))

        return timeline

    def has_emotion_tags(self) -> bool:
        """Check if text has any emotion tags."""
        return len(self.spans) > 0

    def get_emotions_used(self) -> List[str]:
        """Get list of unique emotions used."""
        return list(set(span.emotion for span in self.spans))


class EmotionParser:
    """
    Parser for word-level emotion tags.

    Example:
        parser = EmotionParser()
        result = parser.parse("I'm {happy}so excited{/happy}!")
        print(result.spans)  # [EmotionSpan(text="so excited", emotion="happy", ...)]
    """

    # Pattern for opening tags: {emotion} or {emotion:intensity}
    OPEN_TAG_PATTERN = re.compile(
        r'\{([a-zA-Z_]+)(?::([0-9.]+))?\}',
        re.IGNORECASE
    )

    # Pattern for closing tags: {/emotion}
    CLOSE_TAG_PATTERN = re.compile(
        r'\{/([a-zA-Z_]+)\}',
        re.IGNORECASE
    )

    # Combined pattern for any tag
    ANY_TAG_PATTERN = re.compile(
        r'\{/?[a-zA-Z_]+(?::[0-9.]+)?\}',
        re.IGNORECASE
    )

    def __init__(
        self,
        default_emotion: str = "neutral",
        default_intensity: float = 1.0,
        allow_nesting: bool = True,
    ):
        """
        Initialize the parser.

        Args:
            default_emotion: Emotion for untagged text
            default_intensity: Default intensity when not specified
            allow_nesting: Allow nested emotion tags
        """
        self.default_emotion = default_emotion
        self.default_intensity = default_intensity
        self.allow_nesting = allow_nesting

    def parse(self, text: str) -> ParsedEmotionText:
        """
        Parse text with emotion tags.

        Args:
            text: Text with inline emotion tags

        Returns:
            ParsedEmotionText with spans and plain text
        """
        if not text:
            return ParsedEmotionText(
                original_text="",
                plain_text="",
                spans=[],
                default_emotion=self.default_emotion,
            )

        spans: List[EmotionSpan] = []
        tag_stack: List[Tuple[str, float, int, Optional[str]]] = []  # (emotion, intensity, start_pos, parent)
        plain_text_parts: List[str] = []
        plain_text_pos = 0

        # Track positions
        pos = 0
        while pos < len(text):
            # Check for opening tag
            open_match = self.OPEN_TAG_PATTERN.match(text, pos)
            if open_match:
                emotion = open_match.group(1).lower()
                intensity = float(open_match.group(2)) if open_match.group(2) else self.default_intensity
                intensity = max(0.0, min(1.0, intensity))

                parent = tag_stack[-1][0] if tag_stack and self.allow_nesting else None
                tag_stack.append((emotion, intensity, plain_text_pos, parent))

                pos = open_match.end()
                continue

            # Check for closing tag
            close_match = self.CLOSE_TAG_PATTERN.match(text, pos)
            if close_match:
                closing_emotion = close_match.group(1).lower()

                # Find matching opening tag
                matched = False
                for i in range(len(tag_stack) - 1, -1, -1):
                    if tag_stack[i][0] == closing_emotion:
                        emotion, intensity, start_pos, parent = tag_stack.pop(i)

                        # Calculate the text content
                        span_text = "".join(plain_text_parts[start_pos:])

                        if span_text.strip():  # Only add non-empty spans
                            span = EmotionSpan(
                                text=span_text,
                                emotion=emotion,
                                intensity=intensity,
                                start_char=start_pos,
                                end_char=plain_text_pos,
                                nested_in=parent,
                            )
                            spans.append(span)

                        matched = True
                        break

                if not matched:
                    # Unmatched closing tag - treat as literal text
                    plain_text_parts.append(close_match.group(0))
                    plain_text_pos += len(close_match.group(0))

                pos = close_match.end()
                continue

            # Regular character
            plain_text_parts.append(text[pos])
            plain_text_pos += 1
            pos += 1

        # Handle unclosed tags (just include as spans to current position)
        for emotion, intensity, start_pos, parent in tag_stack:
            span_text = "".join(plain_text_parts[start_pos:])
            if span_text.strip():
                span = EmotionSpan(
                    text=span_text,
                    emotion=emotion,
                    intensity=intensity,
                    start_char=start_pos,
                    end_char=plain_text_pos,
                    nested_in=parent,
                )
                spans.append(span)

        plain_text = "".join(plain_text_parts)

        # Calculate word indices for spans
        self._calculate_word_indices(plain_text, spans)

        return ParsedEmotionText(
            original_text=text,
            plain_text=plain_text,
            spans=spans,
            default_emotion=self.default_emotion,
        )

    def _calculate_word_indices(self, plain_text: str, spans: List[EmotionSpan]):
        """Calculate word indices for each span."""
        if not plain_text:
            return

        # Build character -> word index mapping
        words = plain_text.split()
        char_to_word: Dict[int, int] = {}

        char_pos = 0
        for word_idx, word in enumerate(words):
            # Find word start in text
            while char_pos < len(plain_text) and plain_text[char_pos].isspace():
                char_pos += 1

            # Map each character in word
            for _ in word:
                if char_pos < len(plain_text):
                    char_to_word[char_pos] = word_idx
                    char_pos += 1

        # Update spans with word indices
        for span in spans:
            span.start_word = char_to_word.get(span.start_char, 0)
            # Find end word (exclusive)
            end_char = span.end_char - 1
            while end_char >= span.start_char and plain_text[end_char].isspace():
                end_char -= 1
            span.end_word = char_to_word.get(end_char, len(words) - 1) + 1

    def remove_tags(self, text: str) -> str:
        """
        Remove all emotion tags from text.

        Args:
            text: Text with tags

        Returns:
            Plain text without tags
        """
        return self.ANY_TAG_PATTERN.sub("", text)

    def has_tags(self, text: str) -> bool:
        """Check if text contains emotion tags."""
        return bool(self.ANY_TAG_PATTERN.search(text))

    def extract_emotions(self, text: str) -> List[str]:
        """
        Extract list of emotions used in text.

        Args:
            text: Text with tags

        Returns:
            List of unique emotion names
        """
        emotions = set()
        for match in self.OPEN_TAG_PATTERN.finditer(text):
            emotions.add(match.group(1).lower())
        return list(emotions)


def parse_emotion_tags(text: str) -> ParsedEmotionText:
    """
    Convenience function to parse emotion-tagged text.

    Args:
        text: Text with inline emotion tags

    Returns:
        ParsedEmotionText object
    """
    parser = EmotionParser()
    return parser.parse(text)


def create_tagged_text(text: str, emotion: str, intensity: float = 1.0) -> str:
    """
    Wrap text in emotion tags.

    Args:
        text: Plain text
        emotion: Emotion to apply
        intensity: Intensity (0.0-1.0)

    Returns:
        Tagged text
    """
    if intensity != 1.0:
        return f"{{{emotion}:{intensity}}}{text}{{/{emotion}}}"
    return f"{{{emotion}}}{text}{{/{emotion}}}"


def merge_adjacent_spans(spans: List[EmotionSpan]) -> List[EmotionSpan]:
    """
    Merge adjacent spans with the same emotion.

    Args:
        spans: List of emotion spans

    Returns:
        Merged span list
    """
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: s.start_char)
    merged = [sorted_spans[0]]

    for span in sorted_spans[1:]:
        last = merged[-1]
        if (span.emotion == last.emotion and
            span.intensity == last.intensity and
            span.start_char == last.end_char):
            # Merge
            merged[-1] = EmotionSpan(
                text=last.text + span.text,
                emotion=last.emotion,
                intensity=last.intensity,
                start_char=last.start_char,
                end_char=span.end_char,
                start_word=last.start_word,
                end_word=span.end_word,
            )
        else:
            merged.append(span)

    return merged
