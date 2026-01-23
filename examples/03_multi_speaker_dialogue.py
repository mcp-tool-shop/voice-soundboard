#!/usr/bin/env python3
"""
Multi-Speaker Dialogue Example

Demonstrates conversation synthesis:
- Multiple speakers with distinct voices
- Stage directions (whispering, angrily, etc.)
- Auto voice assignment
- Turn pause control
"""

import asyncio
from voice_soundboard import DialogueEngine, play_audio


async def main():
    engine = DialogueEngine()
    print("Voice Soundboard - Multi-Speaker Dialogue Demo")
    print("=" * 50)

    # 1. Simple two-person dialogue
    print("\n1. Two-person dialogue:")
    script1 = """
[S1:alice] Hey, did you hear the news?
[S2:bob] No, what happened?
[S1:alice] They announced the new project at work!
[S2:bob] That's exciting! When does it start?
"""
    result = await engine.speak_dialogue(
        script1,
        voices={"alice": "af_bella", "bob": "am_michael"}
    )
    print(f"   Generated: {result.audio_path}")
    play_audio(result.audio_path)

    # 2. With stage directions
    print("\n2. Dialogue with stage directions:")
    script2 = """
[S1:narrator] The room was silent.
[S2:detective] (whispering) Did you hear that?
[S3:witness] (nervously) I... I didn't hear anything.
[S2:detective] (firmly) I think you're lying.
[S3:witness] (panicking) No! I swear!
"""
    result = await engine.speak_dialogue(
        script2,
        voices={
            "narrator": "bm_george",
            "detective": "am_michael",
            "witness": "af_nicole"
        }
    )
    play_audio(result.audio_path)

    # 3. Auto voice assignment
    print("\n3. Auto voice assignment (no voices specified):")
    script3 = """
[S1:narrator] Once upon a time...
[S2:princess] Oh, what a lovely day!
[S3:dragon] (menacingly) Not for long!
[S2:princess] (gasping) A dragon!
"""
    result = await engine.speak_dialogue(script3)  # Voices auto-assigned
    print(f"   Auto-assigned voices: {result.voice_assignments}")
    play_audio(result.audio_path)

    # 4. Custom turn pauses
    print("\n4. With longer pauses between speakers:")
    script4 = """
[S1:person1] I have something important to tell you.
[S2:person2] What is it?
[S1:person1] I'm leaving.
"""
    result = await engine.speak_dialogue(
        script4,
        turn_pause_ms=800,  # 800ms between speakers
        voices={"person1": "af_bella", "person2": "bm_george"}
    )
    play_audio(result.audio_path)

    print("\nDemo complete!")
    print("\nStage directions: (whispering), (shouting), (angrily), (nervously), (sarcastically), (laughing)")


if __name__ == "__main__":
    asyncio.run(main())
