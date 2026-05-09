import subprocess

files = [
    "screener_v4_dual.py",
    "scanner_bsjp_v3_dual.py --mode midday",
    "scanner_bsjp_v3_dualsession.py --mode session1",
    "scanner_dual_session_v5.py session1",
    "scanner_bsjp_v4_dual_session.py --session session1",
    "grok_screener_v2.2.py --session siang"
]

processes = [subprocess.Popen(["python", f]) for f in files]

# Menunggu semua proses selesai
for p in processes:
    p.wait()