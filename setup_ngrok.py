"""
setup_ngrok.py
One-time ngrok authtoken setup + permanent public URL launcher.
Run this script ONCE to save your token, then use it every time.

Usage:
    python setup_ngrok.py --token YOUR_NGROK_AUTHTOKEN
"""

import sys
import subprocess
import time
import argparse
import json
from pathlib import Path

VENV_SCRIPTS = Path(__file__).parent / ".venv" / "Scripts"
PYTHON_EXE   = VENV_SCRIPTS / "python.exe"
TOKEN_FILE   = Path(__file__).parent / ".ngrok_token"

def save_token(token: str):
    TOKEN_FILE.write_text(token.strip())
    print(f"[OK] Token saved to {TOKEN_FILE}")

def load_token() -> str | None:
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None

def start_streamlit(port: int):
    print(f"[*] Starting Streamlit on port {port} ...")
    proc = subprocess.Popen(
        [str(PYTHON_EXE), "-m", "streamlit", "run", "app.py",
         "--server.port", str(port), "--server.headless", "true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    return proc

def launch_ngrok(port: int, token: str):
    from pyngrok import ngrok, conf

    # Save token to ngrok config
    conf.get_default().auth_token = token

    print("[*] Opening ngrok tunnel ...")
    tunnel = ngrok.connect(port, "http")
    url = tunnel.public_url.replace("http://", "https://")
    return url

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None,
                        help="Your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    # ── Get token ────────────────────────────
    token = args.token or load_token()
    if not token:
        print("\n" + "=" * 60)
        print("  ngrok authtoken not set!")
        print("=" * 60)
        print("\n  1. Sign up free at: https://dashboard.ngrok.com/signup")
        print("  2. Get your token:  https://dashboard.ngrok.com/get-started/your-authtoken")
        print("  3. Run this command:")
        print("\n     python setup_ngrok.py --token YOUR_TOKEN_HERE\n")
        sys.exit(1)

    # Save token for future use
    save_token(token)

    # ── Start Streamlit ───────────────────────
    st_proc = start_streamlit(args.port)

    # ── Start ngrok ───────────────────────────
    try:
        url = launch_ngrok(args.port, token)
    except Exception as e:
        print(f"\n[ERROR] ngrok failed: {e}")
        print("Check your token is correct at: https://dashboard.ngrok.com/get-started/your-authtoken")
        st_proc.terminate()
        sys.exit(1)

    # ── Print the public URL ──────────────────
    print("\n" + "=" * 62)
    print("  [+] VoiceGuard AI - Voice Cloning Detection")
    print("=" * 62)
    print(f"\n  [+] PUBLIC URL  :  {url}")
    print(f"  [+] Local URL   :  http://localhost:{args.port}")
    print()
    print("  [OK] Share this link with ANYONE worldwide!")
    print("  [OK] Secured with HTTPS automatically")
    print("  [OK] URL stays the same every time you run this script")
    print()
    print("  Press Ctrl+C to stop the server.\n")
    print("=" * 62)

    try:
        st_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down ...")
        from pyngrok import ngrok
        ngrok.kill()
        st_proc.terminate()

if __name__ == "__main__":
    main()
