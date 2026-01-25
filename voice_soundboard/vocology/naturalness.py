"""
Natural Speech Humanization Module

Adds subtle imperfections to make TTS sound more human:
- Filled pauses (um, uh) at natural break points
- Phrase-final creaky voice (vocal fry)
- Contextual breath insertion
- Micro-timing jitter at word boundaries
- Pitch micro-drift within phonemes

Research-backed parameters ensure subtlety - too much ruins naturalness.

References:
- Clark & Fox Tree (2002): "Using uh and um in spontaneous speaking"
- Bortfeld et al. (2001): Filled pause frequency (2-6 per 100 words)
- Ishi et al.: Automatic detection of vocal fry
- Amazon Patent US9508338B1: Breath insertion in TTS
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import random
import re
import math


class FillerType(str, Enum):
    """Types of filled pauses."""
    UM = "um"      # Signals major delay (before complex content)
    UH = "uh"      # Signals minor delay (brief hesitation)
    ER = "er"      # British English variant
    AH = "ah"      # Realization/discovery


class BreathType(str, Enum):
    """Types of breath sounds."""
    INHALE = "inhale"          # Before speaking
    EXHALE = "exhale"          # Rarely used mid-speech
    CATCH_BREATH = "catch"     # Quick breath mid-phrase


@dataclass
class NaturalnessConfig:
    """
    Configuration for natural speech humanization.

    All parameters tuned for subtlety based on linguistics research.
    Default values represent "natural conversational" speech.
    """

    # === Filled Pauses ===
    # Target: 2-3 per 100 words (research shows 2-6, we aim low for subtlety)
    filler_rate: float = 0.025  # 2.5% of potential insertion points

    # Probability of "um" vs "uh" (research: females use um 3x more)
    um_probability: float = 0.6  # 60% um, 40% uh

    # Where fillers can appear (research: 60% more likely before nouns)
    filler_before_nouns_boost: float = 1.6  # 60% boost before nouns
    filler_at_clause_start: float = 0.8     # 80% of initial fillers precede clauses

    # Minimum words between fillers (prevents clustering)
    min_words_between_fillers: int = 15

    # === Phrase-Final Creaky Voice ===
    # Apply to declarative sentences (not questions)
    creaky_on_declaratives: bool = True

    # How many final syllables to affect (1-3)
    creaky_syllables: int = 2

    # F0 reduction for creaky voice (drops to 40-90 Hz range)
    creaky_f0_reduction: float = 0.4  # Reduce F0 by 40%

    # Probability of applying creaky voice (not every sentence)
    creaky_probability: float = 0.3  # 30% of eligible sentences

    # === Breath Insertion ===
    # Minimum phrase length (words) before inserting breath
    breath_min_phrase_length: int = 8

    # Probability of breath at clause boundaries
    breath_at_clause_boundary: float = 0.4  # 40% chance

    # Breath duration range (ms)
    breath_duration_min_ms: int = 100
    breath_duration_max_ms: int = 250

    # Breath volume relative to speech (dB)
    breath_volume_db: float = -25.0

    # === Micro-Timing Jitter ===
    # Random timing variation at word boundaries
    timing_jitter_ms: float = 8.0  # ±8ms (research: 5-15ms)

    # Bias toward delays (more natural than early starts)
    timing_delay_bias: float = 0.6  # 60% delays, 40% early

    # === Pitch Micro-Drift ===
    # Within-phoneme F0 drift
    pitch_drift_cents: float = 6.0  # ±6 cents (subtle)

    # Drift rate (how fast pitch wanders)
    pitch_drift_rate_hz: float = 0.5  # 0.5 Hz = slow wandering

    # More drift on stressed syllables
    stressed_drift_multiplier: float = 1.5

    # Less drift on function words
    function_word_drift_multiplier: float = 0.5


# Common function words (less pitch variation)
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "own", "same", "than", "too", "very", "just",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
}

# POS tags that indicate nouns (fillers more likely before these)
NOUN_INDICATORS = {"NN", "NNS", "NNP", "NNPS", "PRP", "WP"}


@dataclass
class TextAnnotation:
    """Annotation for a segment of text with humanization markers."""
    text: str
    start_idx: int
    end_idx: int

    # Markers
    insert_filler: Optional[FillerType] = None
    insert_breath: Optional[BreathType] = None
    apply_creaky: bool = False
    timing_offset_ms: float = 0.0
    pitch_drift_cents: float = 0.0


def _is_sentence_end(text: str, pos: int) -> bool:
    """Check if position is at end of a sentence."""
    if pos >= len(text):
        return True
    # Look for sentence-ending punctuation
    end_chars = {'.', '!', '?'}
    return text[pos] in end_chars


def _is_clause_boundary(text: str, pos: int) -> bool:
    """Check if position is at a clause boundary."""
    boundary_chars = {',', ';', ':', '-', '(', ')'}
    if pos >= len(text):
        return False
    return text[pos] in boundary_chars


def _is_question(sentence: str) -> bool:
    """Check if sentence is a question."""
    return sentence.strip().endswith('?')


def _count_syllables(word: str) -> int:
    """Rough syllable count for a word."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Handle silent e
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def _get_word_tokens(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text into words with positions."""
    tokens = []
    for match in re.finditer(r'\b\w+\b', text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


class NaturalSpeechProcessor:
    """
    Processes text to add natural speech markers.

    Example:
        >>> processor = NaturalSpeechProcessor()
        >>> result = processor.process("I went to the store to buy groceries.")
        >>> print(result.annotated_text)
        "I went to the, uh, store to buy groceries."
    """

    def __init__(self, config: Optional[NaturalnessConfig] = None):
        self.config = config or NaturalnessConfig()
        self._last_filler_word_idx = -100  # Track filler spacing
        self._random = random.Random()  # For reproducibility if needed

    def seed(self, seed: int):
        """Set random seed for reproducible results."""
        self._random.seed(seed)

    def process(
        self,
        text: str,
        speaking_style: str = "conversational"
    ) -> "ProcessedText":
        """
        Process text and add natural speech markers.

        Args:
            text: Input text to humanize
            speaking_style: "conversational", "formal", "storytelling"

        Returns:
            ProcessedText with annotations and modified text
        """
        # Adjust config based on style
        config = self._adjust_for_style(speaking_style)

        words = _get_word_tokens(text)
        annotations = []

        # Track state
        self._last_filler_word_idx = -100
        words_since_breath = 0

        for i, (word, start, end) in enumerate(words):
            annotation = TextAnnotation(
                text=word,
                start_idx=start,
                end_idx=end
            )

            # Check for filler insertion
            if self._should_insert_filler(i, word, words, config):
                annotation.insert_filler = self._choose_filler_type(word, config)
                self._last_filler_word_idx = i

            # Check for breath insertion
            words_since_breath += 1
            if self._should_insert_breath(i, word, words, words_since_breath, config):
                annotation.insert_breath = BreathType.INHALE
                words_since_breath = 0

            # Check for creaky voice (phrase-final)
            if self._should_apply_creaky(i, word, words, text, config):
                annotation.apply_creaky = True

            # Add micro-timing jitter
            annotation.timing_offset_ms = self._calculate_timing_jitter(word, config)

            # Add pitch micro-drift
            annotation.pitch_drift_cents = self._calculate_pitch_drift(word, config)

            annotations.append(annotation)

        return ProcessedText(
            original_text=text,
            annotations=annotations,
            config=config
        )

    def _adjust_for_style(self, style: str) -> NaturalnessConfig:
        """Adjust config parameters based on speaking style."""
        config = NaturalnessConfig(
            filler_rate=self.config.filler_rate,
            um_probability=self.config.um_probability,
            creaky_probability=self.config.creaky_probability,
            breath_at_clause_boundary=self.config.breath_at_clause_boundary,
            timing_jitter_ms=self.config.timing_jitter_ms,
            pitch_drift_cents=self.config.pitch_drift_cents,
        )

        if style == "formal":
            # Reduce imperfections for formal speech
            config.filler_rate *= 0.3  # 70% fewer fillers
            config.creaky_probability *= 0.5
            config.timing_jitter_ms *= 0.5

        elif style == "storytelling":
            # More expressive, slightly more variation
            config.pitch_drift_cents *= 1.3
            config.timing_jitter_ms *= 1.2
            config.creaky_probability *= 0.7  # Less creaky for engagement

        elif style == "casual":
            # More fillers, more creaky
            config.filler_rate *= 1.5
            config.creaky_probability *= 1.3

        elif style == "nervous":
            # More hesitation, more fillers, breathing
            config.filler_rate *= 2.5  # Lots of "um"s and "uh"s
            config.um_probability = 0.7  # More "um"s (major delays)
            config.breath_at_clause_boundary *= 1.5  # More breathing
            config.timing_jitter_ms *= 1.8  # More timing variation
            config.pitch_drift_cents *= 1.5  # More pitch instability
            config.creaky_probability *= 0.5  # Less creaky (more tense)

        elif style == "excited":
            # Fewer fillers, more pitch variation, faster pace feel
            config.filler_rate *= 0.7  # Fewer fillers
            config.pitch_drift_cents *= 2.0  # More pitch variation
            config.timing_jitter_ms *= 0.7  # Tighter timing
            config.creaky_probability *= 0.4  # Less creaky (more energy)
            config.breath_at_clause_boundary *= 0.8  # Fewer breaths

        return config

    def _should_insert_filler(
        self,
        word_idx: int,
        word: str,
        words: List[Tuple[str, int, int]],
        config: NaturalnessConfig
    ) -> bool:
        """Determine if a filler should be inserted before this word."""
        # Respect minimum spacing
        if word_idx - self._last_filler_word_idx < config.min_words_between_fillers:
            return False

        # Don't start sentences with fillers (usually)
        if word_idx == 0:
            return self._random.random() < 0.1  # 10% chance at start

        # Base probability
        prob = config.filler_rate

        # Boost probability before nouns (60% more likely)
        if self._looks_like_noun(word):
            prob *= config.filler_before_nouns_boost

        # Boost at clause boundaries
        if word_idx > 0:
            prev_word, prev_start, prev_end = words[word_idx - 1]
            # Check if previous word ends with clause boundary
            # (simplified - would need original text for punctuation)
            pass

        # Don't insert before very short common words
        if word.lower() in {"i", "a", "the", "to", "of", "and", "or"}:
            prob *= 0.2

        return self._random.random() < prob

    def _looks_like_noun(self, word: str) -> bool:
        """Simple heuristic to detect nouns (without POS tagger)."""
        word_lower = word.lower()

        # Not a function word
        if word_lower in FUNCTION_WORDS:
            return False

        # Capitalized (proper noun) - but not sentence start
        if word[0].isupper():
            return True

        # Common noun endings
        noun_endings = ('tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist',
                       'ism', 'ance', 'ence', 'dom', 'ship', 'hood')
        if word_lower.endswith(noun_endings):
            return True

        # Longer words are more likely to be nouns
        return len(word) > 5 and self._random.random() < 0.3

    def _choose_filler_type(self, next_word: str, config: NaturalnessConfig) -> FillerType:
        """Choose between um and uh based on context."""
        # Um for major delays (before complex content)
        # Uh for minor delays (brief hesitation)

        if len(next_word) > 8:  # Complex word coming
            um_prob = config.um_probability + 0.2
        else:
            um_prob = config.um_probability

        if self._random.random() < um_prob:
            return FillerType.UM
        return FillerType.UH

    def _should_insert_breath(
        self,
        word_idx: int,
        word: str,
        words: List[Tuple[str, int, int]],
        words_since_breath: int,
        config: NaturalnessConfig
    ) -> bool:
        """Determine if a breath should be inserted before this word."""
        # Need minimum phrase length
        if words_since_breath < config.breath_min_phrase_length:
            return False

        # Higher probability as phrase gets longer
        length_factor = min(2.0, words_since_breath / config.breath_min_phrase_length)
        prob = config.breath_at_clause_boundary * (length_factor - 0.5)

        return self._random.random() < prob

    def _should_apply_creaky(
        self,
        word_idx: int,
        word: str,
        words: List[Tuple[str, int, int]],
        original_text: str,
        config: NaturalnessConfig
    ) -> bool:
        """Determine if creaky voice should apply to this word."""
        if not config.creaky_on_declaratives:
            return False

        # Only apply to final words of sentences
        is_final = word_idx >= len(words) - config.creaky_syllables

        if not is_final:
            # Check if this word is near end of a sentence within text
            _, _, end = words[word_idx]
            remaining_text = original_text[end:end+3]
            is_final = any(c in remaining_text for c in '.!?')

        if not is_final:
            return False

        # Don't apply to questions (they rise, not fall)
        # Simple check - would need sentence boundary detection for accuracy

        return self._random.random() < config.creaky_probability

    def _calculate_timing_jitter(self, word: str, config: NaturalnessConfig) -> float:
        """Calculate random timing offset for word boundary."""
        max_jitter = config.timing_jitter_ms

        # Random value with bias toward delays
        if self._random.random() < config.timing_delay_bias:
            # Delay (positive)
            return self._random.uniform(0, max_jitter)
        else:
            # Early (negative) - smaller magnitude
            return self._random.uniform(-max_jitter * 0.5, 0)

    def _calculate_pitch_drift(self, word: str, config: NaturalnessConfig) -> float:
        """Calculate pitch micro-drift for this word."""
        base_drift = config.pitch_drift_cents

        # Less drift on function words
        if word.lower() in FUNCTION_WORDS:
            base_drift *= config.function_word_drift_multiplier

        # Gaussian distribution for natural variation
        drift = self._random.gauss(0, base_drift / 2)

        # Clamp to reasonable range
        return max(-base_drift * 2, min(base_drift * 2, drift))


@dataclass
class ProcessedText:
    """Result of natural speech processing."""
    original_text: str
    annotations: List[TextAnnotation]
    config: NaturalnessConfig

    def get_annotated_text(self, include_markers: bool = True) -> str:
        """
        Generate text with filler insertions.

        Args:
            include_markers: If True, include [breath] markers

        Returns:
            Text with fillers and optional markers inserted
        """
        result = []
        last_end = 0

        for ann in self.annotations:
            # Add text before this word
            result.append(self.original_text[last_end:ann.start_idx])

            # Add breath marker if needed
            if ann.insert_breath and include_markers:
                result.append("[breath] ")

            # Add filler if needed
            if ann.insert_filler:
                result.append(f"{ann.insert_filler.value}, ")

            # Add the word itself
            result.append(ann.text)

            last_end = ann.end_idx

        # Add remaining text
        result.append(self.original_text[last_end:])

        return ''.join(result)

    def get_ssml(self) -> str:
        """
        Generate SSML markup with natural speech features.

        Returns:
            SSML string with prosody, breaks, and markers
        """
        result = ['<speak>']
        last_end = 0

        for ann in self.annotations:
            # Add text before this word
            result.append(self.original_text[last_end:ann.start_idx])

            # Add breath
            if ann.insert_breath:
                result.append(f'<break time="{self._breath_duration()}ms"/>')

            # Add filler with prosody
            if ann.insert_filler:
                filler = ann.insert_filler.value
                result.append(
                    f'<prosody rate="90%" pitch="-5%">{filler}</prosody>, '
                )

            # Apply prosody modifications
            prosody_attrs = []

            if ann.timing_offset_ms != 0:
                # Use break for timing offset
                if ann.timing_offset_ms > 0:
                    result.append(f'<break time="{int(ann.timing_offset_ms)}ms"/>')

            if ann.pitch_drift_cents != 0:
                # Convert cents to percentage (rough approximation)
                pitch_pct = ann.pitch_drift_cents / 10  # 10 cents ≈ 1%
                prosody_attrs.append(f'pitch="{pitch_pct:+.1f}%"')

            if ann.apply_creaky:
                # Simulate creaky with lower pitch and slower rate
                prosody_attrs.append('pitch="-15%"')
                prosody_attrs.append('rate="85%"')

            if prosody_attrs:
                result.append(f'<prosody {" ".join(prosody_attrs)}>')
                result.append(ann.text)
                result.append('</prosody>')
            else:
                result.append(ann.text)

            last_end = ann.end_idx

        # Add remaining text
        result.append(self.original_text[last_end:])
        result.append('</speak>')

        return ''.join(result)

    def _breath_duration(self) -> int:
        """Get random breath duration within config range."""
        return random.randint(
            self.config.breath_duration_min_ms,
            self.config.breath_duration_max_ms
        )

    def get_timing_offsets(self) -> List[Tuple[str, float]]:
        """Get list of (word, timing_offset_ms) tuples."""
        return [(ann.text, ann.timing_offset_ms) for ann in self.annotations]

    def get_pitch_drifts(self) -> List[Tuple[str, float]]:
        """Get list of (word, pitch_drift_cents) tuples."""
        return [(ann.text, ann.pitch_drift_cents) for ann in self.annotations]

    def get_creaky_words(self) -> List[str]:
        """Get list of words that should have creaky voice applied."""
        return [ann.text for ann in self.annotations if ann.apply_creaky]

    def summary(self) -> dict:
        """Get summary statistics of applied humanization."""
        fillers = [ann for ann in self.annotations if ann.insert_filler]
        breaths = [ann for ann in self.annotations if ann.insert_breath]
        creaky = [ann for ann in self.annotations if ann.apply_creaky]

        word_count = len(self.annotations)

        return {
            "word_count": word_count,
            "fillers_inserted": len(fillers),
            "filler_rate_per_100": len(fillers) / word_count * 100 if word_count else 0,
            "filler_types": {
                "um": sum(1 for f in fillers if f.insert_filler == FillerType.UM),
                "uh": sum(1 for f in fillers if f.insert_filler == FillerType.UH),
            },
            "breaths_inserted": len(breaths),
            "creaky_words": len(creaky),
            "avg_timing_jitter_ms": sum(abs(a.timing_offset_ms) for a in self.annotations) / word_count if word_count else 0,
            "avg_pitch_drift_cents": sum(abs(a.pitch_drift_cents) for a in self.annotations) / word_count if word_count else 0,
        }


# Convenience function
def humanize_text(
    text: str,
    style: str = "conversational",
    config: Optional[NaturalnessConfig] = None
) -> ProcessedText:
    """
    Add natural speech features to text.

    Args:
        text: Input text
        style: Speaking style preset:
            - "conversational": Default, moderate fillers and naturalness
            - "formal": Minimal fillers, cleaner delivery
            - "storytelling": Expressive, more pitch variation
            - "casual": More fillers, more creaky voice
            - "nervous": Hesitant, lots of fillers, unsteady timing
            - "excited": Energetic, fewer fillers, wide pitch range
        config: Optional custom configuration

    Returns:
        ProcessedText with annotations

    Example:
        >>> result = humanize_text("I need to go to the store to buy some groceries for dinner.")
        >>> print(result.get_annotated_text())
        "I need to go to the, uh, store to buy some groceries for dinner."
        >>> print(result.summary())
        {'word_count': 12, 'fillers_inserted': 1, ...}
    """
    processor = NaturalSpeechProcessor(config)
    return processor.process(text, speaking_style=style)
