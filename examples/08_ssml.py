#!/usr/bin/env python3
"""
SSML (Speech Synthesis Markup Language) Example

Demonstrates SSML parsing and synthesis:
- Pauses with <break>
- Speed control with <prosody>
- Emphasis with <emphasis>
- Special formatting with <say-as>
- Pronunciation with <sub>
"""

from voice_soundboard import VoiceEngine, parse_ssml, play_audio


def main():
    engine = VoiceEngine()
    print("Voice Soundboard - SSML Demo")
    print("=" * 35)

    # 1. Basic SSML with breaks
    print("\n1. Pauses with <break>:")
    ssml1 = """
    <speak>
        Hello. <break time="500ms"/> How are you today?
        <break time="1s"/>
        I hope you're doing well.
    </speak>
    """
    text, params = parse_ssml(ssml1)
    print(f"   Parsed text: {text.strip()}")
    result = engine.speak(text)
    play_audio(result.audio_path)

    # 2. Prosody (speed control)
    print("\n2. Speed control with <prosody>:")
    ssml2 = """
    <speak>
        Normal speed text.
        <prosody rate="slow">This is spoken slowly.</prosody>
        <prosody rate="fast">This is spoken quickly!</prosody>
        Back to normal.
    </speak>
    """
    text, params = parse_ssml(ssml2)
    result = engine.speak(text, speed=params.speed)
    play_audio(result.audio_path)

    # 3. Emphasis
    print("\n3. Emphasis levels:")
    ssml3 = """
    <speak>
        This is <emphasis level="moderate">important</emphasis> information.
        But this is <emphasis level="strong">CRITICAL</emphasis>!
    </speak>
    """
    text, params = parse_ssml(ssml3)
    result = engine.speak(text)
    play_audio(result.audio_path)

    # 4. Say-as (special formatting)
    print("\n4. Special formatting with <say-as>:")
    ssml4 = """
    <speak>
        The date is <say-as interpret-as="date">2024-01-15</say-as>.
        Call me at <say-as interpret-as="telephone">555-1234</say-as>.
        The answer is <say-as interpret-as="cardinal">42</say-as>.
    </speak>
    """
    text, params = parse_ssml(ssml4)
    print(f"   Parsed: {text.strip()}")
    result = engine.speak(text)
    play_audio(result.audio_path)

    # 5. Substitution
    print("\n5. Pronunciation substitution with <sub>:")
    ssml5 = """
    <speak>
        I love <sub alias="typescript">TS</sub> and <sub alias="javascript">JS</sub>.
        The <sub alias="World Wide Web Consortium">W3C</sub> creates web standards.
    </speak>
    """
    text, params = parse_ssml(ssml5)
    print(f"   Parsed: {text.strip()}")
    result = engine.speak(text)
    play_audio(result.audio_path)

    # 6. Complex example
    print("\n6. Complex SSML:")
    ssml6 = """
    <speak>
        <prosody rate="medium">
            Welcome to the presentation.
            <break time="300ms"/>
        </prosody>

        <emphasis level="strong">Today's topic:</emphasis>
        Voice Synthesis.
        <break time="500ms"/>

        <prosody rate="slow">
            Let me explain <emphasis level="moderate">carefully</emphasis>.
        </prosody>

        <break time="1s"/>

        That concludes our session.
        Thank you!
    </speak>
    """
    text, params = parse_ssml(ssml6)
    result = engine.speak(text, speed=params.speed)
    play_audio(result.audio_path)

    print("\nDemo complete!")
    print("\nSupported SSML tags:")
    print("  <speak> - Root element")
    print("  <break time='Xms|Xs'> - Pause")
    print("  <prosody rate='slow|medium|fast'> - Speed")
    print("  <emphasis level='moderate|strong'> - Stress")
    print("  <say-as interpret-as='date|time|cardinal|telephone'> - Format")
    print("  <sub alias='text'> - Pronunciation substitution")


if __name__ == "__main__":
    main()
