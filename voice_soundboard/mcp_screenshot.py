"""
MCP Screenshot Tool - Capture screenshots and screen recordings.

Provides tools for:
- Taking screenshots (full screen, window, or region)
- Opening Windows Snipping Tool
- Starting/stopping screen recordings with FFmpeg
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime
import json
import tempfile


# Output directory for captures
CAPTURES_DIR = Path(__file__).parent.parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)


def take_screenshot(
    region: str = "full",
    output_path: str = None,
    window_title: str = None
) -> dict:
    """
    Take a screenshot using PowerShell.

    Args:
        region: "full" for full screen, "window" to capture a specific window
        output_path: Optional path to save the screenshot (auto-generated if not provided)
        window_title: Window title to capture (for region="window")

    Returns:
        Dict with path to the saved screenshot
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(CAPTURES_DIR / f"screenshot_{timestamp}.png")

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if region == "full":
        # Full screen capture using PowerShell
        ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$screen = [System.Windows.Forms.Screen]::PrimaryScreen
$bitmap = New-Object System.Drawing.Bitmap($screen.Bounds.Width, $screen.Bounds.Height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($screen.Bounds.Location, [System.Drawing.Point]::Empty, $screen.Bounds.Size)
$bitmap.Save("{output_path}")
$graphics.Dispose()
$bitmap.Dispose()
'''
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and Path(output_path).exists():
            return {
                "success": True,
                "path": output_path,
                "message": f"Screenshot saved to {output_path}"
            }
        else:
            return {
                "success": False,
                "error": result.stderr or "Failed to capture screenshot"
            }

    elif region == "window" and window_title:
        # Capture specific window
        ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$window = Get-Process | Where-Object {{$_.MainWindowTitle -like "*{window_title}*"}} | Select-Object -First 1
if ($window) {{
    $handle = $window.MainWindowHandle

    Add-Type @"
    using System;
    using System.Runtime.InteropServices;
    public class Win32 {{
        [DllImport("user32.dll")]
        public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [StructLayout(LayoutKind.Sequential)]
        public struct RECT {{
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }}
    }}
"@

    $rect = New-Object Win32+RECT
    [Win32]::GetWindowRect($handle, [ref]$rect)

    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top

    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, [System.Drawing.Size]::new($width, $height))
    $bitmap.Save("{output_path}")
    $graphics.Dispose()
    $bitmap.Dispose()
    Write-Output "Success"
}} else {{
    Write-Error "Window not found: {window_title}"
}}
'''
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True
        )

        if "Success" in result.stdout and Path(output_path).exists():
            return {
                "success": True,
                "path": output_path,
                "message": f"Window screenshot saved to {output_path}"
            }
        else:
            return {
                "success": False,
                "error": result.stderr or "Failed to capture window"
            }

    return {"success": False, "error": "Invalid parameters"}


def open_snipping_tool(mode: str = "rectangular") -> dict:
    """
    Open Windows Snipping Tool in the specified mode.

    Args:
        mode: "rectangular", "freeform", "window", or "fullscreen"

    Returns:
        Dict indicating success
    """
    # Map mode to Snipping Tool command
    mode_flags = {
        "rectangular": "/clip",
        "freeform": "/clip",
        "window": "/clip",
        "fullscreen": "/clip"
    }

    try:
        # Try new Snipping Tool (Windows 11)
        subprocess.Popen(["SnippingTool.exe", mode_flags.get(mode, "/clip")])
        return {
            "success": True,
            "message": f"Snipping Tool opened in {mode} mode. Use Win+Shift+S for quick access."
        }
    except FileNotFoundError:
        try:
            # Fallback to legacy snipping tool
            subprocess.Popen(["snippingtool.exe"])
            return {
                "success": True,
                "message": "Legacy Snipping Tool opened"
            }
        except FileNotFoundError:
            # Try Win+Shift+S shortcut simulation
            ps_script = '''
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.SendKeys]::SendWait("#+s")
'''
            subprocess.run(["powershell", "-Command", ps_script])
            return {
                "success": True,
                "message": "Triggered Win+Shift+S for screen snip"
            }


def start_screen_recording(
    output_path: str = None,
    region: str = "full",
    fps: int = 30,
    duration: int = None
) -> dict:
    """
    Start screen recording using FFmpeg.

    Args:
        output_path: Path to save the recording (auto-generated if not provided)
        region: "full" for full screen (region capture not yet supported)
        fps: Frames per second (default 30)
        duration: Max duration in seconds (None for unlimited, stop with stop_recording)

    Returns:
        Dict with recording info
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(CAPTURES_DIR / f"recording_{timestamp}.mp4")

    # Check if FFmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "success": False,
            "error": "FFmpeg not found. Install FFmpeg and add to PATH."
        }

    # Build FFmpeg command for Windows screen capture
    cmd = [
        "ffmpeg",
        "-f", "gdigrab",
        "-framerate", str(fps),
        "-i", "desktop",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23"
    ]

    if duration:
        cmd.extend(["-t", str(duration)])

    cmd.append(output_path)

    # Start recording in background
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Save PID for stopping later
        pid_file = CAPTURES_DIR / "recording.pid"
        pid_file.write_text(str(process.pid))

        return {
            "success": True,
            "pid": process.pid,
            "output_path": output_path,
            "message": f"Recording started. PID: {process.pid}. Call stop_screen_recording() to stop."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def stop_screen_recording() -> dict:
    """
    Stop the current screen recording.

    Returns:
        Dict indicating success
    """
    pid_file = CAPTURES_DIR / "recording.pid"

    if not pid_file.exists():
        return {
            "success": False,
            "error": "No active recording found"
        }

    try:
        pid = int(pid_file.read_text())

        # Send 'q' to FFmpeg to gracefully stop
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)

        pid_file.unlink()

        return {
            "success": True,
            "message": f"Recording stopped (PID: {pid})"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def list_captures() -> dict:
    """
    List all captured screenshots and recordings.

    Returns:
        Dict with list of capture files
    """
    captures = []

    for f in CAPTURES_DIR.glob("*"):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".mp4", ".gif"]:
            captures.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                "type": "screenshot" if f.suffix.lower() in [".png", ".jpg", ".jpeg"] else "recording"
            })

    captures.sort(key=lambda x: x["created"], reverse=True)

    return {
        "captures_dir": str(CAPTURES_DIR),
        "count": len(captures),
        "captures": captures[:20]  # Return last 20
    }


# MCP Tool definitions for registration
MCP_TOOLS = [
    {
        "name": "screenshot",
        "description": "Take a screenshot of the screen or a specific window",
        "parameters": {
            "region": {"type": "string", "enum": ["full", "window"], "default": "full"},
            "window_title": {"type": "string", "description": "Window title to capture (for region=window)"},
            "output_path": {"type": "string", "description": "Optional path to save screenshot"}
        },
        "handler": take_screenshot
    },
    {
        "name": "snipping_tool",
        "description": "Open Windows Snipping Tool for manual capture",
        "parameters": {
            "mode": {"type": "string", "enum": ["rectangular", "freeform", "window", "fullscreen"], "default": "rectangular"}
        },
        "handler": open_snipping_tool
    },
    {
        "name": "start_recording",
        "description": "Start screen recording (requires FFmpeg)",
        "parameters": {
            "output_path": {"type": "string", "description": "Path to save recording"},
            "fps": {"type": "integer", "default": 30},
            "duration": {"type": "integer", "description": "Max duration in seconds"}
        },
        "handler": start_screen_recording
    },
    {
        "name": "stop_recording",
        "description": "Stop the current screen recording",
        "parameters": {},
        "handler": stop_screen_recording
    },
    {
        "name": "list_captures",
        "description": "List all captured screenshots and recordings",
        "parameters": {},
        "handler": list_captures
    }
]


if __name__ == "__main__":
    # Test the tools
    print("Testing screenshot tool...")
    result = take_screenshot()
    print(json.dumps(result, indent=2))

    print("\nListing captures...")
    result = list_captures()
    print(json.dumps(result, indent=2))
