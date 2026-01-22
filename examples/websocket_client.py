"""
Example WebSocket client for Voice Soundboard.

Usage:
    1. Start the server: python -m voice_soundboard.websocket_server
    2. Run this client: python examples/websocket_client.py

The client demonstrates all available WebSocket actions.
"""

import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


async def send_and_receive(ws, action: str, params: dict = None):
    """Send a message and print the response."""
    message = {"action": action}
    if params:
        message.update(params)

    print(f"\n>>> Sending: {action}")
    await ws.send(json.dumps(message))

    response = await ws.recv()
    data = json.loads(response)

    if data.get("success"):
        print(f"<<< Success: {json.dumps(data.get('data', {}), indent=2)}")
    else:
        print(f"<<< Error: {data.get('error')}")

    return data


async def demo_basic_speech(ws):
    """Demo basic speech generation."""
    print("\n" + "=" * 50)
    print("Demo: Basic Speech Generation")
    print("=" * 50)

    await send_and_receive(ws, "speak", {
        "text": "Hello! This is a test of the WebSocket API.",
        "voice": "af_bella",
        "play": True,
    })


async def demo_emotions(ws):
    """Demo emotion-based speech."""
    print("\n" + "=" * 50)
    print("Demo: Emotional Speech")
    print("=" * 50)

    # List emotions first
    await send_and_receive(ws, "list_emotions")

    # Speak with emotion
    await send_and_receive(ws, "speak", {
        "text": "This is absolutely fantastic news!",
        "emotion": "excited",
        "play": True,
    })


async def demo_realtime(ws):
    """Demo realtime streaming."""
    print("\n" + "=" * 50)
    print("Demo: Realtime Streaming (plays as it generates)")
    print("=" * 50)

    await send_and_receive(ws, "speak_realtime", {
        "text": "This audio is streaming in real-time. You hear it as it generates, without waiting for the full file.",
        "preset": "narrator",
    })


async def demo_ssml(ws):
    """Demo SSML speech."""
    print("\n" + "=" * 50)
    print("Demo: SSML Speech")
    print("=" * 50)

    await send_and_receive(ws, "speak_ssml", {
        "ssml": """
            <speak>
                Welcome! <break time="500ms"/>
                Today is <say-as interpret-as="date">2026-01-22</say-as>.
                <emphasis level="strong">This is important.</emphasis>
            </speak>
        """,
        "play": True,
    })


async def demo_effects(ws):
    """Demo sound effects."""
    print("\n" + "=" * 50)
    print("Demo: Sound Effects")
    print("=" * 50)

    # List effects
    await send_and_receive(ws, "list_effects")

    # Play effect
    await send_and_receive(ws, "effect", {"effect": "chime"})
    await asyncio.sleep(1)
    await send_and_receive(ws, "effect", {"effect": "success"})


async def demo_streaming_chunks(ws):
    """Demo streaming with chunk callbacks."""
    print("\n" + "=" * 50)
    print("Demo: Streaming Chunks (receive audio data)")
    print("=" * 50)

    message = {
        "action": "speak_stream",
        "text": "This demonstrates streaming where you receive audio chunks as they generate.",
        "voice": "bm_george",
    }

    print(f"\n>>> Sending: speak_stream")
    await ws.send(json.dumps(message))

    # Receive multiple messages (start, chunks, end)
    while True:
        response = await ws.recv()
        data = json.loads(response)
        action = data.get("action", "")

        if action == "speak_stream_start":
            print(f"<<< Stream started")
        elif action == "speak_stream_chunk":
            print(f"<<< Chunk {data.get('chunk_index')}: {data.get('samples')} samples")
        elif action == "speak_stream_end":
            print(f"<<< Stream ended: {data.get('data', {}).get('total_chunks')} chunks, {data.get('data', {}).get('total_duration'):.2f}s")
            break
        else:
            print(f"<<< {action}: {data}")
            if not data.get("success", True):
                break


async def demo_voices_and_presets(ws):
    """Demo listing voices and presets."""
    print("\n" + "=" * 50)
    print("Demo: Voices and Presets")
    print("=" * 50)

    await send_and_receive(ws, "list_voices", {"gender": "female"})
    await send_and_receive(ws, "list_presets")


async def interactive_mode(ws):
    """Interactive mode - type messages to send."""
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("=" * 50)
    print("Type JSON messages to send to the server.")
    print("Examples:")
    print('  {"action": "speak", "text": "Hello", "play": true}')
    print('  {"action": "effect", "effect": "chime"}')
    print('  {"action": "status"}')
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() == "quit":
                break

            await ws.send(user_input)
            response = await ws.recv()
            print(f"<<< {response}\n")

        except json.JSONDecodeError:
            print("Invalid JSON")
        except KeyboardInterrupt:
            break


async def main():
    """Main client function."""
    uri = "ws://localhost:8765"

    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as ws:
            # Receive welcome message
            welcome = await ws.recv()
            print(f"Connected: {json.loads(welcome).get('data', {}).get('message')}")

            # Get server status
            await send_and_receive(ws, "status")

            # Run demos
            if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
                await interactive_mode(ws)
            else:
                await demo_voices_and_presets(ws)
                await demo_basic_speech(ws)
                await demo_emotions(ws)
                await demo_ssml(ws)
                await demo_effects(ws)
                # await demo_realtime(ws)  # Uncomment to test realtime
                # await demo_streaming_chunks(ws)  # Uncomment to test streaming

                print("\n" + "=" * 50)
                print("All demos complete!")
                print("=" * 50)

    except ConnectionRefusedError:
        print(f"Could not connect to {uri}")
        print("Make sure the server is running:")
        print("  python -m voice_soundboard.websocket_server")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
