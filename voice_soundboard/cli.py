"""
Command-line interface for Voice Soundboard.

Usage:
    voice-soundboard speak "Hello world!"
    voice-soundboard speak "I'm excited!" --emotion excited
    voice-soundboard speak "Good day!" --voice bm_george
    voice-soundboard speak "Breaking news!" --preset announcer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_speak(args: argparse.Namespace) -> None:
    """Handle the 'speak' subcommand."""
    from voice_soundboard import VoiceEngine

    engine = VoiceEngine()

    kwargs: dict = {}
    if args.voice:
        kwargs["voice"] = args.voice
    if args.preset:
        kwargs["preset"] = args.preset
    if args.emotion:
        kwargs["style"] = args.emotion
    if args.speed is not None:
        kwargs["speed"] = args.speed
    if args.output:
        kwargs["save_as"] = args.output

    result = engine.speak(args.text, **kwargs)
    print(result.audio_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="voice-soundboard",
        description="Text-to-speech from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # speak subcommand
    speak_parser = subparsers.add_parser(
        "speak",
        help="Generate speech from text.",
    )
    speak_parser.add_argument(
        "text",
        help="The text to speak.",
    )
    speak_parser.add_argument(
        "--voice",
        help="Voice ID (e.g. af_bella, bm_george). Default: af_bella.",
    )
    speak_parser.add_argument(
        "--preset",
        choices=["assistant", "narrator", "announcer", "storyteller", "whisper"],
        help="Voice preset.",
    )
    speak_parser.add_argument(
        "--emotion",
        help="Emotion or style hint (e.g. excited, warmly).",
    )
    speak_parser.add_argument(
        "--speed",
        type=float,
        help="Speech speed multiplier (0.5-2.0). Default: 1.0.",
    )
    speak_parser.add_argument(
        "-o", "--output",
        help="Output filename (without extension). Default: auto-generated.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "speak":
        cmd_speak(args)


if __name__ == "__main__":
    main()
