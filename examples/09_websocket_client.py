#!/usr/bin/env python3
"""
WebSocket Client Example

Demonstrates real-time communication with the Voice Soundboard WebSocket server:
- Connecting with authentication
- Sending speak requests
- Receiving audio data
- Streaming responses
- Multi-speaker dialogue via WebSocket
"""

import asyncio
import json
import base64
from pathlib import Path

try:
    import websockets
except ImportError:
    print("WebSocket client requires: pip install websockets")
    exit(1)


class VoiceSoundboardClient:
    """WebSocket client for Voice Soundboard server."""

    def __init__(self, uri: str = "ws://localhost:8765", api_key: str = None):
        self.uri = uri
        self.api_key = api_key
        self.ws = None

    async def connect(self):
        """Connect to the WebSocket server."""
        uri = self.uri
        if self.api_key:
            uri += f"?key={self.api_key}"

        self.ws = await websockets.connect(uri)
        print(f"Connected to {self.uri}")

    async def disconnect(self):
        """Disconnect from the server."""
        if self.ws:
            await self.ws.close()
            print("Disconnected")

    async def send(self, action: str, **kwargs) -> dict:
        """Send a request and get the response."""
        message = {"action": action, **kwargs}
        await self.ws.send(json.dumps(message))
        response = await self.ws.recv()
        return json.loads(response)

    async def speak(self, text: str, voice: str = None, emotion: str = None,
                    preset: str = None, style: str = None, speed: float = None,
                    play: bool = False, return_audio: bool = False) -> dict:
        """Generate speech."""
        params = {"text": text}
        if voice:
            params["voice"] = voice
        if emotion:
            params["emotion"] = emotion
        if preset:
            params["preset"] = preset
        if style:
            params["style"] = style
        if speed:
            params["speed"] = speed
        if play:
            params["play"] = play
        if return_audio:
            params["return_audio"] = return_audio

        return await self.send("speak", **params)

    async def stream(self, text: str, voice: str = None) -> dict:
        """Stream speech generation."""
        params = {"text": text}
        if voice:
            params["voice"] = voice

        return await self.send("speak_stream", **params)

    async def effect(self, effect_name: str, play: bool = True) -> dict:
        """Play a sound effect."""
        return await self.send("effect", effect=effect_name, play=play)

    async def list_voices(self) -> dict:
        """Get available voices."""
        return await self.send("list_voices")

    async def list_effects(self) -> dict:
        """Get available effects."""
        return await self.send("list_effects")


async def demo_basic():
    """Basic WebSocket communication demo."""
    print("\n" + "=" * 50)
    print("1. Basic Speech Generation")
    print("=" * 50)

    client = VoiceSoundboardClient()
    await client.connect()

    try:
        # Simple speech
        result = await client.speak(
            "Hello from the WebSocket client!",
            play=True
        )
        print(f"Result: {result}")

        # With voice selection
        result = await client.speak(
            "This is a British voice.",
            voice="bm_george",
            play=True
        )
        print(f"Result: {result}")

        # With emotion
        result = await client.speak(
            "I'm so excited about this!",
            emotion="excited",
            play=True
        )
        print(f"Result: {result}")

    finally:
        await client.disconnect()


async def demo_with_audio_return():
    """Demo receiving audio data via WebSocket."""
    print("\n" + "=" * 50)
    print("2. Receiving Audio Data")
    print("=" * 50)

    client = VoiceSoundboardClient()
    await client.connect()

    try:
        # Request audio data in response
        result = await client.speak(
            "This audio will be returned as base64.",
            return_audio=True
        )

        if "audio" in result:
            # Decode and save the audio
            audio_data = base64.b64decode(result["audio"])
            output_path = Path("websocket_audio.wav")
            output_path.write_bytes(audio_data)
            print(f"Saved audio to: {output_path}")
            print(f"Audio size: {len(audio_data)} bytes")
        else:
            print(f"Response: {result}")

    finally:
        await client.disconnect()


async def demo_streaming():
    """Demo streaming speech generation."""
    print("\n" + "=" * 50)
    print("3. Streaming Speech")
    print("=" * 50)

    client = VoiceSoundboardClient()
    await client.connect()

    try:
        # Stream a long text
        result = await client.stream(
            """Voice Soundboard supports real-time streaming for low-latency
            applications. This is especially useful for interactive voice
            assistants and chatbots where responsiveness matters."""
        )
        print(f"Streaming result: {result}")

    finally:
        await client.disconnect()


async def demo_effects():
    """Demo sound effects via WebSocket."""
    print("\n" + "=" * 50)
    print("4. Sound Effects")
    print("=" * 50)

    client = VoiceSoundboardClient()
    await client.connect()

    try:
        # List available effects
        effects = await client.list_effects()
        print(f"Available effects: {effects.get('effects', [])}")

        # Play some effects
        for effect_name in ["chime", "success", "attention"]:
            print(f"Playing: {effect_name}")
            await client.effect(effect_name, play=True)
            await asyncio.sleep(0.5)

    finally:
        await client.disconnect()


async def demo_authenticated():
    """Demo with API key authentication."""
    print("\n" + "=" * 50)
    print("5. Authenticated Connection")
    print("=" * 50)

    # Start server with: VOICE_API_KEY=secret123 python -m voice_soundboard.websocket_server
    client = VoiceSoundboardClient(api_key="secret123")

    try:
        await client.connect()
        result = await client.speak("Authenticated successfully!", play=True)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Connection failed (is server running with API key?): {e}")
    finally:
        await client.disconnect()


async def demo_dialogue():
    """Demo multi-speaker dialogue via WebSocket."""
    print("\n" + "=" * 50)
    print("6. Multi-Speaker Dialogue")
    print("=" * 50)

    client = VoiceSoundboardClient()
    await client.connect()

    try:
        script = """
[S1:alice] Hey, have you tried the new Voice Soundboard?
[S2:bob] No, what is it?
[S1:alice] It's an AI voice synthesis platform with 54 voices!
[S2:bob] That sounds amazing!
"""
        result = await client.send(
            "dialogue",
            script=script,
            voices={"alice": "af_bella", "bob": "am_michael"},
            play=True
        )
        print(f"Dialogue result: {result}")

    finally:
        await client.disconnect()


async def interactive_client():
    """Interactive WebSocket client."""
    print("\n" + "=" * 50)
    print("7. Interactive Client")
    print("=" * 50)
    print("Commands: speak <text>, effect <name>, voices, effects, quit")

    client = VoiceSoundboardClient()
    await client.connect()

    try:
        while True:
            try:
                command = input("\n> ").strip()
            except EOFError:
                break

            if not command:
                continue

            if command == "quit":
                break
            elif command == "voices":
                result = await client.list_voices()
                voices = result.get("voices", {})
                print(f"Available voices ({len(voices)}):")
                for vid in list(voices.keys())[:10]:
                    print(f"  - {vid}")
            elif command == "effects":
                result = await client.list_effects()
                print(f"Effects: {result.get('effects', [])}")
            elif command.startswith("speak "):
                text = command[6:]
                result = await client.speak(text, play=True)
                print(f"Generated: {result.get('audio_path', 'unknown')}")
            elif command.startswith("effect "):
                effect = command[7:]
                await client.effect(effect, play=True)
                print(f"Played: {effect}")
            else:
                print("Unknown command. Try: speak <text>, effect <name>, voices, effects, quit")

    finally:
        await client.disconnect()


async def main():
    """Run all demos."""
    print("Voice Soundboard - WebSocket Client Demo")
    print("=" * 50)
    print("\nMake sure the WebSocket server is running:")
    print("  python -m voice_soundboard.websocket_server")
    print("\nPress Enter to start demos, or 'i' for interactive mode...")

    try:
        choice = input().strip().lower()
    except EOFError:
        choice = ""

    if choice == "i":
        await interactive_client()
    else:
        await demo_basic()
        await asyncio.sleep(1)

        await demo_with_audio_return()
        await asyncio.sleep(1)

        await demo_streaming()
        await asyncio.sleep(1)

        await demo_effects()
        await asyncio.sleep(1)

        await demo_dialogue()

        print("\n" + "=" * 50)
        print("All demos complete!")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
