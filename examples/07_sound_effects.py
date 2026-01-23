#!/usr/bin/env python3
"""
Sound Effects Example

Demonstrates built-in sound effects:
- Playing effects
- Saving effects to files
- Available effect types
"""

from voice_soundboard import list_effects, play_effect, get_effect


def main():
    print("Voice Soundboard - Sound Effects Demo")
    print("=" * 42)

    # 1. List available effects
    print("\n1. Available effects:")
    effects = list_effects()
    for effect in effects:
        print(f"   - {effect}")

    # 2. Play each effect
    print("\n2. Playing effects:")
    demo_effects = ["chime", "success", "error", "attention", "click"]
    for effect_name in demo_effects:
        print(f"   Playing: {effect_name}")
        play_effect(effect_name)
        import time
        time.sleep(0.5)  # Small pause between effects

    # 3. Notification sounds
    print("\n3. Notification sounds:")
    notifications = ["info", "warning", "critical"]
    for notif in notifications:
        print(f"   {notif.upper()}:")
        play_effect(notif)
        import time
        time.sleep(0.8)

    # 4. UI sounds
    print("\n4. UI interaction sounds:")
    ui_sounds = ["click", "pop", "whoosh"]
    for sound in ui_sounds:
        print(f"   {sound}:")
        play_effect(sound)
        import time
        time.sleep(0.3)

    # 5. Ambient/background sounds
    print("\n5. Ambient sounds (short preview):")
    ambient = ["rain", "white_noise", "drone"]
    for sound in ambient:
        print(f"   {sound} (1 second preview):")
        # Get the effect and play a short preview
        effect = get_effect(sound)
        # Note: In a real app, you might want to loop these
        play_effect(sound)
        import time
        time.sleep(1.0)

    # 6. Save effect to file
    print("\n6. Saving effect to file:")
    effect = get_effect("chime")
    output_path = "notification_chime.wav"
    effect.save(output_path)
    print(f"   Saved 'chime' to: {output_path}")

    print("\nDemo complete!")
    print(f"\nTotal effects available: {len(effects)}")


if __name__ == "__main__":
    main()
