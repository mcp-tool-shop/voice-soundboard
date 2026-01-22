"""
MCP Server for Voice Soundboard.

Exposes TTS and sound effects capabilities to AI agents via MCP.
"""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from voice_soundboard.engine import VoiceEngine
from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS
from voice_soundboard.audio import play_audio, stop_playback, get_audio_duration
from voice_soundboard.interpreter import interpret_style, apply_style_to_params
from voice_soundboard.effects import get_effect, list_effects, EFFECTS
from voice_soundboard.ssml import parse_ssml
from voice_soundboard.emotions import (
    get_emotion_params, get_emotion_voice_params,
    apply_emotion_to_text, list_emotions, EMOTIONS
)
from voice_soundboard.streaming import stream_realtime, RealtimeStreamResult


# Global engine instance (lazy loaded)
_engine: VoiceEngine | None = None


def get_engine() -> VoiceEngine:
    """Get or create the voice engine singleton."""
    global _engine
    if _engine is None:
        _engine = VoiceEngine()
    return _engine


# Create MCP server
server = Server("voice-soundboard")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available voice tools."""
    return [
        Tool(
            name="speak",
            description=(
                "Generate natural speech from text. Returns path to audio file. "
                "Use 'style' for natural language hints like 'warmly', 'excitedly', "
                "'like a narrator'. Use 'voice' for specific voice IDs, 'preset' for "
                "predefined personalities (assistant, narrator, announcer, storyteller, whisper)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak"
                    },
                    "style": {
                        "type": "string",
                        "description": "Natural language style hint: 'warmly', 'excitedly', 'like a narrator', 'in a british accent'"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Specific voice ID (e.g., 'af_bella', 'bm_george'). Overrides style."
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset: assistant, narrator, announcer, storyteller, whisper"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speech speed multiplier (0.5-2.0, default 1.0)"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play audio immediately after generation (default: false)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="list_voices",
            description="List all available voices with their characteristics (gender, accent, style)",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_gender": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "Filter by gender"
                    },
                    "filter_accent": {
                        "type": "string",
                        "enum": ["american", "british", "japanese", "mandarin"],
                        "description": "Filter by accent"
                    }
                }
            }
        ),
        Tool(
            name="list_presets",
            description="List available voice presets with descriptions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="play_audio",
            description="Play an audio file through speakers",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to audio file"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="stop_audio",
            description="Stop any currently playing audio",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="sound_effect",
            description=(
                "Play a sound effect. Available effects: chime, success, error, attention, "
                "click, pop, whoosh, warning, critical, info, rain, white_noise, drone"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "effect": {
                        "type": "string",
                        "description": "Effect name: chime, success, error, attention, click, pop, whoosh, warning, critical, info, rain, white_noise, drone"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional path to save the effect (plays by default)"
                    }
                },
                "required": ["effect"]
            }
        ),
        Tool(
            name="list_effects",
            description="List all available sound effects",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="speak_long",
            description=(
                "Stream speech for long text (paragraphs, articles). "
                "More efficient for text longer than a few sentences. "
                "Saves to file and optionally plays."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Long text to speak"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speed multiplier"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play after generation (default: false)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="speak_ssml",
            description=(
                "Speak text with SSML markup for fine control over pauses, emphasis, "
                "and pronunciation. Supports: <break time='500ms'/>, "
                "<emphasis level='strong'>text</emphasis>, "
                "<say-as interpret-as='date'>2024-01-15</say-as>, "
                "<prosody rate='slow'>text</prosody>, <sub alias='replacement'>abbr</sub>"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ssml": {
                        "type": "string",
                        "description": "SSML-formatted text (with or without <speak> wrapper)"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    },
                    "play": {
                        "type": "boolean",
                        "description": "Play after generation (default: false)"
                    }
                },
                "required": ["ssml"]
            }
        ),
        Tool(
            name="speak_realtime",
            description=(
                "Stream speech with real-time playback. Audio plays immediately as it generates, "
                "no waiting for the full file. Best for interactive responses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID"
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(VOICE_PRESETS.keys()),
                        "description": "Voice preset"
                    },
                    "emotion": {
                        "type": "string",
                        "enum": list(EMOTIONS.keys()),
                        "description": "Emotion to apply (happy, sad, excited, calm, angry, etc.)"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "description": "Speed multiplier"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="list_emotions",
            description="List all available emotions for speech synthesis with their characteristics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "speak":
        return await handle_speak(arguments)
    elif name == "list_voices":
        return await handle_list_voices(arguments)
    elif name == "list_presets":
        return await handle_list_presets(arguments)
    elif name == "play_audio":
        return await handle_play_audio(arguments)
    elif name == "stop_audio":
        return await handle_stop_audio(arguments)
    elif name == "sound_effect":
        return await handle_sound_effect(arguments)
    elif name == "list_effects":
        return await handle_list_effects(arguments)
    elif name == "speak_long":
        return await handle_speak_long(arguments)
    elif name == "speak_ssml":
        return await handle_speak_ssml(arguments)
    elif name == "speak_realtime":
        return await handle_speak_realtime(arguments)
    elif name == "list_emotions":
        return await handle_list_emotions(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_speak(args: dict[str, Any]) -> list[TextContent]:
    """Generate speech from text."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    style = args.get("style")
    voice = args.get("voice")
    preset = args.get("preset")
    speed = args.get("speed")
    should_play = args.get("play", False)

    # Apply style interpretation
    if style:
        voice, speed, preset = apply_style_to_params(style, voice, speed, preset)

    try:
        engine = get_engine()
        result = engine.speak(
            text=text,
            voice=voice,
            preset=preset,
            speed=speed,
        )

        # Play if requested
        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Generated speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speed: {result.realtime_factor:.1f}x realtime"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating speech: {e}")]


async def handle_list_voices(args: dict[str, Any]) -> list[TextContent]:
    """List available voices with optional filtering."""
    filter_gender = args.get("filter_gender")
    filter_accent = args.get("filter_accent")

    voices = []
    for voice_id, info in sorted(KOKORO_VOICES.items()):
        # Apply filters
        if filter_gender and info.get("gender") != filter_gender:
            continue
        if filter_accent and info.get("accent") != filter_accent:
            continue

        voices.append(
            f"  {voice_id}: {info['name']} ({info['gender']}, {info['accent']}, {info['style']})"
        )

    if not voices:
        return [TextContent(type="text", text="No voices match the filters")]

    response = f"Available voices ({len(voices)}):\n" + "\n".join(voices)
    return [TextContent(type="text", text=response)]


async def handle_list_presets(args: dict[str, Any]) -> list[TextContent]:
    """List voice presets."""
    lines = ["Voice presets:"]
    for name, config in VOICE_PRESETS.items():
        lines.append(
            f"  {name}: {config['description']} (voice: {config['voice']}, speed: {config.get('speed', 1.0)})"
        )
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_play_audio(args: dict[str, Any]) -> list[TextContent]:
    """Play an audio file."""
    path = args.get("path")
    if not path:
        return [TextContent(type="text", text="Error: 'path' is required")]

    path = Path(path)
    if not path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {path}")]

    try:
        duration = get_audio_duration(path)
        await asyncio.to_thread(play_audio, path)
        return [TextContent(type="text", text=f"Played: {path.name} ({duration:.2f}s)")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error playing audio: {e}")]


async def handle_stop_audio(args: dict[str, Any]) -> list[TextContent]:
    """Stop audio playback."""
    try:
        stop_playback()
        return [TextContent(type="text", text="Audio playback stopped")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error stopping audio: {e}")]


async def handle_sound_effect(args: dict[str, Any]) -> list[TextContent]:
    """Play or save a sound effect."""
    effect_name = args.get("effect", "")
    if not effect_name:
        return [TextContent(type="text", text="Error: 'effect' is required")]

    save_path = args.get("save_path")

    try:
        effect = get_effect(effect_name)

        if save_path:
            path = Path(save_path)
            effect.save(path)
            return [TextContent(type="text", text=f"Saved effect '{effect_name}' to: {path}")]
        else:
            # Play the effect
            await asyncio.to_thread(effect.play)
            return [TextContent(type="text", text=f"Played effect: {effect_name} ({effect.duration:.2f}s)")]

    except ValueError as e:
        return [TextContent(type="text", text=str(e))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error with sound effect: {e}")]


async def handle_list_effects(args: dict[str, Any]) -> list[TextContent]:
    """List all available sound effects."""
    effects = list_effects()

    categories = {
        "Chimes": ["chime", "chime_success", "chime_error", "chime_attention"],
        "UI": ["click", "pop", "whoosh"],
        "Alerts": ["alert_warning", "alert_critical", "alert_info"],
        "Ambient": ["rain", "white_noise", "drone"],
    }

    lines = ["Available sound effects:"]
    for category, items in categories.items():
        lines.append(f"\n  {category}:")
        for item in items:
            if item in EFFECTS:
                lines.append(f"    - {item}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_speak_long(args: dict[str, Any]) -> list[TextContent]:
    """Stream speech for long text."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    voice = args.get("voice")
    preset = args.get("preset")
    speed = args.get("speed", 1.0)
    should_play = args.get("play", False)

    try:
        from voice_soundboard.streaming import StreamingEngine
        import hashlib

        engine = StreamingEngine()

        # Generate output filename
        text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        output_path = Path("F:/AI/voice-soundboard/output") / f"stream_{text_hash}.wav"

        result = await engine.stream_to_file(
            text=text,
            output_path=output_path,
            voice=voice,
            preset=preset,
            speed=speed,
        )

        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Streamed speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Duration: {result.total_duration:.2f}s\n"
            f"  Chunks: {result.total_chunks}\n"
            f"  Gen time: {result.generation_time:.2f}s"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error streaming speech: {e}")]


async def handle_speak_ssml(args: dict[str, Any]) -> list[TextContent]:
    """Speak text with SSML markup."""
    ssml = args.get("ssml", "")
    if not ssml:
        return [TextContent(type="text", text="Error: 'ssml' is required")]

    voice = args.get("voice")
    preset = args.get("preset")
    should_play = args.get("play", False)

    try:
        # Parse SSML to text and extract parameters
        text, ssml_params = parse_ssml(ssml)

        # Use SSML-extracted parameters if not overridden
        if ssml_params.voice and not voice:
            voice = ssml_params.voice
        speed = ssml_params.speed

        engine = get_engine()
        result = engine.speak(
            text=text,
            voice=voice,
            preset=preset,
            speed=speed,
        )

        if should_play:
            await asyncio.to_thread(play_audio, result.audio_path)

        response = (
            f"Generated SSML speech:\n"
            f"  File: {result.audio_path}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Duration: {result.duration_seconds:.2f}s\n"
            f"  Speed: {speed}x\n"
            f"  Processed text: {text[:80]}{'...' if len(text) > 80 else ''}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error with SSML speech: {e}")]


async def handle_speak_realtime(args: dict[str, Any]) -> list[TextContent]:
    """Stream speech with real-time playback."""
    text = args.get("text", "")
    if not text:
        return [TextContent(type="text", text="Error: 'text' is required")]

    voice = args.get("voice")
    preset = args.get("preset")
    emotion = args.get("emotion")
    speed = args.get("speed")

    try:
        # Apply emotion if specified
        if emotion:
            emotion_params = get_emotion_voice_params(emotion, voice, speed)
            voice = emotion_params["voice"]
            speed = emotion_params["speed"]
            # Optionally modify text for emotional emphasis
            text = apply_emotion_to_text(text, emotion)

        # Stream with real-time playback
        result = await stream_realtime(
            text=text,
            voice=voice,
            preset=preset,
            speed=speed or 1.0,
        )

        emotion_str = f"\n  Emotion: {emotion}" if emotion else ""
        response = (
            f"Real-time speech completed:\n"
            f"  Duration: {result.total_duration:.2f}s\n"
            f"  Chunks: {result.total_chunks}\n"
            f"  Voice: {result.voice_used}\n"
            f"  Gen time: {result.generation_time:.2f}s{emotion_str}"
        )
        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error with realtime speech: {e}")]


async def handle_list_emotions(args: dict[str, Any]) -> list[TextContent]:
    """List available emotions."""
    lines = ["Available emotions:"]

    # Group by category
    categories = {
        "Positive": ["happy", "excited", "joyful"],
        "Calm": ["calm", "peaceful", "neutral"],
        "Negative": ["sad", "melancholy", "angry", "frustrated"],
        "High-energy": ["fearful", "surprised", "urgent"],
        "Professional": ["confident", "serious", "professional"],
        "Storytelling": ["mysterious", "dramatic", "whimsical"],
    }

    for category, emotions in categories.items():
        lines.append(f"\n  {category}:")
        for emotion in emotions:
            if emotion in EMOTIONS:
                params = EMOTIONS[emotion]
                voice = params.voice_preference or "default"
                lines.append(f"    {emotion}: speed={params.speed:.2f}, voice={voice}")

    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
