import subprocess
import sys

files = [
    ["python", "screener_v4_dual.py", "sesi1"],
    ["python", "scanner_bsjp_v3_dual.py", "--mode", "session1"],
    ["python", "scanner_bsjp_v3_dualsession.py", "--mode", "session1"],
    ["python", "scanner_dual_session_v5.py", "session1"],
    ["python", "scanner_bsjp_v4_dual_session.py", "--session", "session1"],
    ["python", "grok_screener_v2.2.py", "--session", "siang"],
]

for cmd in files:
    script = cmd[1]
    print(f"\n{'='*50}")
    print(f"▶ Menjalankan: {' '.join(cmd)}")
    print(f"{'='*50}")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ {script} selesai.")
        else:
            print(f"⚠️  {script} exit code {result.returncode}")
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {script} — skip.")
    except KeyboardInterrupt:
        print("\n🛑 Dihentikan user.")
        sys.exit(0)