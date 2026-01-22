"""
SSML Parser - Speech Synthesis Markup Language support.

Converts SSML tags to Kokoro-compatible text with:
- <break> - Pauses of various lengths
- <emphasis> - Stressed words (via punctuation hints)
- <prosody> - Speed/pitch adjustments
- <say-as> - Interpretation hints (date, time, cardinal, etc.)
- <phoneme> - Direct phoneme control
- <sub> - Pronunciation substitutions

Example:
    ssml = '''
    <speak>
        Hello <break time="500ms"/> world!
        <emphasis level="strong">This is important.</emphasis>
        The date is <say-as interpret-as="date">2024-01-15</say-as>.
    </speak>
    '''
    text, params = parse_ssml(ssml)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# SECURITY: Use defusedxml to prevent XXE (XML External Entity) attacks
# Standard xml.etree.ElementTree is vulnerable to billion laughs, external entity
# expansion, and DTD retrieval attacks
import defusedxml.ElementTree as ET


@dataclass
class SSMLParams:
    """Parameters extracted from SSML."""
    speed: float = 1.0
    pitch: float = 1.0  # Not directly supported, but tracked
    volume: float = 1.0  # Not directly supported, but tracked
    voice: Optional[str] = None
    phonemes: list[str] = field(default_factory=list)


# Pause durations mapped to punctuation/silence
BREAK_MARKERS = {
    "none": "",
    "x-weak": ",",
    "weak": ",",
    "medium": "...",
    "strong": "...",
    "x-strong": "... ...",
}


def _parse_time(time_str: str) -> float:
    """Parse time string like '500ms' or '1.5s' to seconds."""
    time_str = time_str.strip().lower()
    if time_str.endswith("ms"):
        return float(time_str[:-2]) / 1000
    elif time_str.endswith("s"):
        return float(time_str[:-1])
    else:
        try:
            return float(time_str)
        except ValueError:
            return 0.5  # Default pause


def _time_to_pause(seconds: float) -> str:
    """Convert time in seconds to pause markers."""
    if seconds <= 0:
        return ""
    elif seconds < 0.25:
        return ","
    elif seconds < 0.5:
        return "..."
    elif seconds < 1.0:
        return "... ..."
    else:
        # For long pauses, use multiple ellipses
        count = int(seconds / 0.5)
        return " ".join(["..."] * min(count, 5))


def _format_date(date_str: str) -> str:
    """Format date string for speech."""
    # Handle ISO format: 2024-01-15
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if match:
        year, month, day = match.groups()
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_name = months[int(month) - 1]
        day_int = int(day)
        # Add ordinal suffix
        if 10 <= day_int % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_int % 10, "th")
        return f"{month_name} {day_int}{suffix}, {year}"
    return date_str


def _format_time(time_str: str) -> str:
    """Format time string for speech."""
    # Handle 24h format: 14:30
    match = re.match(r"(\d{1,2}):(\d{2})", time_str)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour == 0:
            hour_str = "12"
            ampm = "AM"
        elif hour < 12:
            hour_str = str(hour)
            ampm = "AM"
        elif hour == 12:
            hour_str = "12"
            ampm = "PM"
        else:
            hour_str = str(hour - 12)
            ampm = "PM"

        if minute == 0:
            return f"{hour_str} {ampm}"
        elif minute < 10:
            return f"{hour_str} oh {minute} {ampm}"
        else:
            return f"{hour_str} {minute} {ampm}"
    return time_str


def _format_cardinal(num_str: str) -> str:
    """Format number as cardinal."""
    try:
        num = int(num_str.replace(",", ""))
        # Simple number to words for common cases
        if num < 0:
            return "negative " + _format_cardinal(str(-num))
        if num == 0:
            return "zero"
        if num < 20:
            words = [
                "", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
            ]
            return words[num]
        if num < 100:
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            return tens[num // 10] + ("-" + _format_cardinal(str(num % 10)) if num % 10 else "")
        # For larger numbers, just return the string (TTS handles it)
        return num_str
    except ValueError:
        return num_str


def _format_ordinal(num_str: str) -> str:
    """Format number as ordinal."""
    try:
        num = int(num_str)
        if 10 <= num % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
        return f"{num}{suffix}"
    except ValueError:
        return num_str


def _format_characters(text: str) -> str:
    """Spell out characters."""
    return " ".join(text.upper())


def _format_telephone(phone: str) -> str:
    """Format phone number for speech."""
    # Remove non-digits
    digits = re.sub(r"\D", "", phone)
    # Speak each digit
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    return " ".join(digit_words.get(d, d) for d in digits)


def _process_say_as(text: str, interpret_as: str) -> str:
    """Process say-as interpretation."""
    interpret_as = interpret_as.lower()

    if interpret_as == "date":
        return _format_date(text)
    elif interpret_as == "time":
        return _format_time(text)
    elif interpret_as == "cardinal":
        return _format_cardinal(text)
    elif interpret_as == "ordinal":
        return _format_ordinal(text)
    elif interpret_as == "characters" or interpret_as == "spell-out":
        return _format_characters(text)
    elif interpret_as == "telephone":
        return _format_telephone(text)
    else:
        return text


def _process_element(elem: ET.Element, params: SSMLParams) -> str:
    """Process an SSML element and return text."""
    result = []

    # Handle element's own text
    if elem.text:
        result.append(elem.text)

    # Process child elements
    for child in elem:
        tag = child.tag.lower()

        if tag == "break":
            # Handle pause
            strength = child.get("strength", "medium")
            time_attr = child.get("time")

            if time_attr:
                pause = _time_to_pause(_parse_time(time_attr))
            else:
                pause = BREAK_MARKERS.get(strength, "...")

            result.append(f" {pause} ")

        elif tag == "emphasis":
            # Handle emphasis - wrap in markers that affect prosody
            level = child.get("level", "moderate")
            inner_text = _process_element(child, params)

            if level == "strong":
                # Add exclamation-like emphasis
                result.append(f"*{inner_text}*")
            elif level == "reduced":
                result.append(inner_text)
            else:
                result.append(inner_text)

        elif tag == "prosody":
            # Handle prosody changes
            rate = child.get("rate", "medium")
            pitch = child.get("pitch", "medium")

            # Adjust speed based on rate
            rate_map = {
                "x-slow": 0.5,
                "slow": 0.75,
                "medium": 1.0,
                "fast": 1.25,
                "x-fast": 1.5,
            }

            if rate in rate_map:
                params.speed = rate_map[rate]
            elif rate.endswith("%"):
                try:
                    params.speed = float(rate[:-1]) / 100
                except ValueError:
                    pass

            result.append(_process_element(child, params))

        elif tag == "say-as":
            # Handle interpretation hints
            interpret_as = child.get("interpret-as", "")
            inner_text = "".join(child.itertext())
            result.append(_process_say_as(inner_text, interpret_as))

        elif tag == "sub":
            # Handle substitution
            alias = child.get("alias", "")
            if alias:
                result.append(alias)
            else:
                result.append(_process_element(child, params))

        elif tag == "phoneme":
            # Handle phonemes - pass through for now
            # Kokoro supports phonemes via is_phonemes=True
            ph = child.get("ph", "")
            if ph:
                params.phonemes.append(ph)
                result.append(ph)
            else:
                result.append(_process_element(child, params))

        elif tag == "voice":
            # Handle voice change
            name = child.get("name")
            if name:
                params.voice = name
            result.append(_process_element(child, params))

        elif tag == "p" or tag == "s":
            # Paragraph or sentence - add pause after
            result.append(_process_element(child, params))
            result.append("... ")

        elif tag == "speak":
            # Root element - just process children
            result.append(_process_element(child, params))

        else:
            # Unknown tag - process as text
            result.append(_process_element(child, params))

        # Handle tail text (text after the closing tag)
        if child.tail:
            result.append(child.tail)

    return "".join(result)


def parse_ssml(ssml: str) -> tuple[str, SSMLParams]:
    """
    Parse SSML and return plain text with extracted parameters.

    Args:
        ssml: SSML string (with or without <speak> wrapper)

    Returns:
        Tuple of (processed_text, SSMLParams)

    Example:
        text, params = parse_ssml('<speak>Hello <break time="1s"/> world!</speak>')
        # text = "Hello ... ... world!"
        # params.speed = 1.0
    """
    params = SSMLParams()

    # Wrap in <speak> if not present
    ssml = ssml.strip()
    if not ssml.lower().startswith("<speak"):
        ssml = f"<speak>{ssml}</speak>"

    try:
        root = ET.fromstring(ssml)
    except ET.ParseError as e:
        # If parsing fails, return original text stripped of tags
        text = re.sub(r"<[^>]+>", "", ssml)
        return text.strip(), params

    # Process the tree
    text = _process_element(root, params)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Clean up multiple consecutive pauses
    text = re.sub(r"(\.\.\.\s*)+", "... ", text)

    return text, params


def ssml_to_text(ssml: str) -> str:
    """
    Simple conversion from SSML to plain text.

    Args:
        ssml: SSML string

    Returns:
        Plain text with pause markers
    """
    text, _ = parse_ssml(ssml)
    return text


# Convenience functions for building SSML
def pause(time: str = "500ms") -> str:
    """Generate a pause tag."""
    return f'<break time="{time}"/>'


def emphasis(text: str, level: str = "moderate") -> str:
    """Generate an emphasis tag."""
    return f'<emphasis level="{level}">{text}</emphasis>'


def say_as(text: str, interpret_as: str) -> str:
    """Generate a say-as tag."""
    return f'<say-as interpret-as="{interpret_as}">{text}</say-as>'


def prosody(text: str, rate: str = "medium", pitch: str = "medium") -> str:
    """Generate a prosody tag."""
    return f'<prosody rate="{rate}" pitch="{pitch}">{text}</prosody>'


if __name__ == "__main__":
    # Test examples
    examples = [
        # Basic pause
        '<speak>Hello <break time="500ms"/> world!</speak>',

        # Emphasis
        '<speak>This is <emphasis level="strong">very important</emphasis>.</speak>',

        # Say-as
        '<speak>The date is <say-as interpret-as="date">2024-01-15</say-as>.</speak>',
        '<speak>Call me at <say-as interpret-as="telephone">555-123-4567</say-as>.</speak>',
        '<speak>The answer is <say-as interpret-as="cardinal">42</say-as>.</speak>',

        # Prosody
        '<speak><prosody rate="slow">Speaking slowly now.</prosody></speak>',

        # Substitution
        '<speak>The <sub alias="World Wide Web">WWW</sub> is great.</speak>',

        # Combined
        '''<speak>
            Welcome! <break time="1s"/>
            Today is <say-as interpret-as="date">2024-01-15</say-as>.
            <emphasis level="strong">Please listen carefully.</emphasis>
            <prosody rate="slow">This part is slower.</prosody>
        </speak>''',
    ]

    print("SSML Parser Tests")
    print("=" * 60)

    for ssml in examples:
        text, params = parse_ssml(ssml)
        print(f"\nInput: {ssml[:60]}...")
        print(f"Output: {text}")
        if params.speed != 1.0:
            print(f"Speed: {params.speed}")
