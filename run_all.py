import subprocess

files = [
    "scanner_gemini.py",
    "scanner_claude.py",
    "scanner_chatgpt.py",
    "scanner_grok.py"
]

processes = [subprocess.Popen(["python", f]) for f in files]

# Menunggu semua proses selesai
for p in processes:
    p.wait()