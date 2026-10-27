"""
launch_public.py
Starts the Streamlit app and exposes it publicly — NO account required.

Uses localhost.run (free, no signup) via SSH for the tunnel.
Optionally falls back to ngrok if you provide an authtoken.

Usage:
    python launch_public.py                          # localhost.run (free, no login)
    python launch_public.py --ngrok YOUR_TOKEN       # ngrok (stable URL, free account)
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# ── Ensure venv python / streamlit are used ──
VENV_DIR        = Path(__file__).parent / ".venv" / "Scripts"
STREAMLIT_EXE   = VENV_DIR / "streamlit.exe"
PYTHON_EXE      = VENV_DIR / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)

def _start_streamlit(port: int) -> subprocess.Popen:
    print(f"[*] Starting Streamlit on port {port} …")
    proc = subprocess.Popen(
        [str(PYTHON_EXE), "-m", "streamlit", "run", "app.py",
         "--server.port", str(port),
         "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    return proc


def _tunnel_localhostrun(port: int) -> subprocess.Popen:
    """Open a free public tunnel via localhost.run (SSH, no account needed)."""
    print("[*] Opening free public tunnel via localhost.run …")
    proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-R", f"80:localhost:{port}",
         "nokey@localhost.run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    public_url = None
    for line in proc.stdout:
        line = line.strip()
        if "tunneled with tls termination" in line and "https://" in line:
            public_url = line.split("https://")[-1].strip()
            public_url = "https://" + public_url
            break
    return proc, public_url


def _tunnel_ngrok(port: int, token: str) -> None:
    """Open ngrok tunnel (requires free account + authtoken)."""
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = token
    tunnel = ngrok.connect(port, "http")
    return None, tunnel.public_url.replace("http://", "https://")


def main():
    parser = argparse.ArgumentParser(description="Launch VoiceGuard AI publicly")
    parser.add_argument("--ngrok", default=None, metavar="TOKEN",
                        help="Use ngrok with your authtoken instead of localhost.run")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    port = args.port
    st_proc = _start_streamlit(port)

    if args.ngrok:
        tunnel_proc, public_url = _tunnel_ngrok(port, args.ngrok)
    else:
        tunnel_proc, public_url = _tunnel_localhostrun(port)

    print("\n" + "=" * 62)
    print("  🛡️  VoiceGuard AI — Voice Cloning Detection")
    print("=" * 62)
    if public_url:
        print(f"\n  🌐  Public URL  :  {public_url}")
    print(f"  🏠  Local URL   :  http://localhost:{port}")
    print("\n  Share the Public URL with anyone worldwide!")
    print("  (The link is active while this script is running)\n")
    print("  Press Ctrl+C to stop.\n")
    print("=" * 62)

    try:
        if tunnel_proc:
            tunnel_proc.wait()
        else:
            st_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down …")
        if tunnel_proc:
            tunnel_proc.terminate()
        st_proc.terminate()


if __name__ == "__main__":
    main()
