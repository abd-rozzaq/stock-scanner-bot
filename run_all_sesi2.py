import subprocess

files = [
    "screener_v4_dual.py",
    "grok_screener_v2.2.py --session sore",
    "scanner_bsjp_v3_dual.py --mode afternoon",
    "scanner_bsjp_v3_dualsession.py --mode session2",
    "scanner_dual_session_v5.py session2",
    "scanner_bsjp_v4_dual_session.py --session session2"
]

processes = [subprocess.Popen(["python", f]) for f in files]

# Menunggu semua proses selesai
for p in processes:
    p.wait()