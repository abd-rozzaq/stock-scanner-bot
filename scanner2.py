# ================================================================
# scanner_bsjp_v2.py - ENHANCED BSJP STOCK SCANNER
# Version : 2.0.0
# Author  : Merlin AI Assistant
# Date    : 2026-02-12
# Desc    : Scanner saham IDX untuk strategi BSJP
#           (Beli Sore Jual Pagi) dengan noise reduction,
#           scoring system, blacklist, dan integrasi Telegram.
#
# MAIN CRITERIA (semua harus terpenuhi):
#   1. Close >= High * 0.98
#   2. Volume (proxy Frequency) > 1000
#   3. Volume > Volume MA 5
#   4. Price Change % >= 3
#
# ADDITIONAL NOISE FILTERS (scoring, min 3 dari 4):
#   5. RSI(14) < 80         (anti-overbought)
#   6. Value >= 1 Miliar     (minimum value transaksi)
#   7. Close > MA 20         (uptrend confirmation)
#   8. Volume >= 2x Vol MA20 (volume spike)
#
# SUMBER DATA : data/data.csv
# BLACKLIST   : data/blacklist.csv (opsional)
# CONFIG      : config.py (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
# ================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import logging
import time

# Coba import requests untuk Telegram
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING SETUP
# ======================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"scanner_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BSJP_V2")

# ======================================================
# 1. CONSTANTS
# ======================================================
VERSION = "2.0.0"

# Main Criteria Thresholds
CLOSE_HIGH_RATIO = 0.98          # Close >= High * 0.98
MIN_FREQUENCY = 1000             # Volume (proxy frequency) > 1000
MIN_PRICE_CHANGE_PCT = 3         # Price Change % >= 3

# Noise Filter Thresholds
RSI_MAX = 80                     # RSI(14) harus < 80
MIN_VALUE_IDR = 1_000_000_000    # Minimum value transaksi 1 Miliar
VOL_SPIKE_MULTIPLIER = 2         # Volume >= 2x Volume MA20

# Scoring
MIN_EXTRA_SCORE = 3              # Minimum 3 dari 4 filter tambahan

# Risk Management
TP_PERCENT = 0.08                # Target Profit 8%
CL_PERCENT = 0.05               # Cut Loss 5%

# Technical Indicator Periods
RSI_PERIOD = 14
MA20_PERIOD = 20
MA5_PERIOD = 5

# Telegram
MAX_MESSAGE_LENGTH = 4096        # Batas karakter Telegram per pesan

# Data fetch
YFINANCE_PERIOD = "2mo"          # 2 bulan data untuk MA20 yang akurat
YFINANCE_INTERVAL = "1d"

# ======================================================
# 2. TELEGRAM CONFIG
# ======================================================
TELEGRAM_OK = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if REQUESTS_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        TELEGRAM_OK = True
        logger.info(f"Telegram Config Loaded. Chat ID: {TELEGRAM_CHAT_ID}")
    else:
        logger.warning("Telegram config ditemukan tapi requests tidak tersedia atau token/chat_id kosong.")
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram notifikasi dinonaktifkan.")
except Exception as e:
    logger.warning(f"Error loading config.py: {e}")

# ======================================================
# 3. TELEGRAM HELPER
# ======================================================
def send_telegram_message(message: str) -> bool:
    """
    Mengirim pesan ke Telegram.
    Otomatis split jika pesan melebihi MAX_MESSAGE_LENGTH.
    Returns True jika berhasil, False jika gagal.
    """
    if not TELEGRAM_OK:
        logger.info("Telegram tidak aktif, pesan tidak dikirim.")
        return False

    # Split pesan jika terlalu panjang
    messages_to_send = split_telegram_message(message)

    all_success = True
    for i, msg_chunk in enumerate(messages_to_send):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg_chunk,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=15)

            if response.status_code == 200:
                logger.info(f"Pesan Telegram terkirim ({i+1}/{len(messages_to_send)}).")
            else:
                logger.error(
                    f"Gagal kirim Telegram (Status {response.status_code}): {response.text}"
                )
                all_success = False
        except requests.exceptions.Timeout:
            logger.error("Timeout saat mengirim pesan Telegram.")
            all_success = False
        except requests.exceptions.ConnectionError:
            logger.error("Tidak dapat terhubung ke server Telegram.")
            all_success = False
        except Exception as e:
            logger.error(f"Error koneksi Telegram: {e}")
            all_success = False

        # Delay antar pesan agar tidak rate-limited
        if len(messages_to_send) > 1 and i < len(messages_to_send) - 1:
            time.sleep(1)

    return all_success


def split_telegram_message(message: str) -> List[str]:
    """
    Memecah pesan yang melebihi MAX_MESSAGE_LENGTH menjadi beberapa bagian.
    Split berdasarkan newline agar tidak memotong di tengah baris.
    """
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]

    chunks = []
    current_chunk = ""
    lines = message.split("\n")

    for line in lines:
        # Jika satu baris saja sudah melebihi limit, potong paksa
        if len(line) > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Potong baris panjang
            for j in range(0, len(line), MAX_MESSAGE_LENGTH):
                chunks.append(line[j:j + MAX_MESSAGE_LENGTH])
            continue

        # Cek apakah menambahkan baris ini akan melebihi limit
        test_chunk = current_chunk + line + "\n"
        if len(test_chunk) > MAX_MESSAGE_LENGTH:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk = test_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [message[:MAX_MESSAGE_LENGTH]]

# ======================================================
# 4. DATA HELPERS
# ======================================================
def load_tickers_from_csv() -> List[str]:
    """
    Membaca daftar ticker saham dari file CSV.
    Mencari di data/data.csv, fallback ke data.csv di root.
    """
    file_path = os.path.join("data", "data.csv")

    if not os.path.exists(file_path):
        if os.path.exists("data.csv"):
            file_path = "data.csv"
        else:
            logger.error(f"File data.csv tidak ditemukan di 'data/' maupun root directory.")
            return []

    try:
        df = pd.read_csv(file_path)
        possible_cols = ['Ticker', 'ticker', 'Kode', 'kode', 'Code', 'code', 'Symbol', 'symbol']
        found_col = next((c for c in possible_cols if c in df.columns), df.columns[0])

        logger.info(f"Membaca '{file_path}' (Kolom: {found_col})")
        tickers = df[found_col].dropna().astype(str).tolist()
        cleaned = list(set([t.strip().upper() for t in tickers if len(t.strip()) >= 4]))
        logger.info(f"Total ticker unik: {len(cleaned)}")
        return sorted(cleaned)
    except pd.errors.EmptyDataError:
        logger.error(f"File '{file_path}' kosong.")
        return []
    except Exception as e:
        logger.error(f"Error membaca '{file_path}': {e}")
        return []


def load_blacklist() -> List[str]:
    """
    Membaca daftar blacklist saham dari data/blacklist.csv.
    File ini opsional — jika tidak ada, return list kosong.
    Blacklist berisi saham yang baru suspend, UMA, atau unusual activity.
    """
    file_path = os.path.join("data", "blacklist.csv")

    if not os.path.exists(file_path):
        logger.info("File blacklist.csv tidak ditemukan — tidak ada blacklist aktif.")
        return []

    try:
        df = pd.read_csv(file_path)
        possible_cols = ['Ticker', 'ticker', 'Kode', 'kode', 'Code', 'code', 'Symbol', 'symbol']
        found_col = next((c for c in possible_cols if c in df.columns), df.columns[0])

        tickers = df[found_col].dropna().astype(str).tolist()
        cleaned = list(set([t.strip().upper() for t in tickers if len(t.strip()) >= 4]))
        logger.info(f"Blacklist aktif: {len(cleaned)} saham ({', '.join(cleaned[:10])}{'...' if len(cleaned) > 10 else ''})")
        return cleaned
    except Exception as e:
        logger.warning(f"Error membaca blacklist: {e}")
        return []


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    Mengambil data historis saham dari Yahoo Finance.
    Menambahkan suffix .JK untuk saham IDX.
    Returns DataFrame dengan kolom: Open, High, Low, Close, Volume
    """
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(
            symbol,
            period=YFINANCE_PERIOD,
            interval=YFINANCE_INTERVAL,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns dari yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"{ticker}: Kolom data tidak lengkap.")
            return pd.DataFrame()

        result = df[required_cols].copy()

        # Hapus baris dengan NaN
        result.dropna(inplace=True)

        return result
    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# ======================================================
# 5. TECHNICAL INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """
    Menghitung RSI (Relative Strength Index).
    Handles edge cases: data tidak cukup, division by zero.
    Returns RSI value (0-100) atau -1 jika gagal.
    """
    try:
        if len(series) < period + 1:
            return -1.0

        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        # Menggunakan Exponential Moving Average (Wilder's smoothing)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        current_avg_gain = avg_gain.iloc[-1]
        current_avg_loss = avg_loss.iloc[-1]

        # Handle division by zero
        if current_avg_loss == 0:
            if current_avg_gain == 0:
                return 50.0  # Tidak ada pergerakan, RSI netral
            else:
                return 100.0  # Semua gain, tidak ada loss

        rs = current_avg_gain / current_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return round(rsi, 1)
    except Exception as e:
        logger.debug(f"Error menghitung RSI: {e}")
        return -1.0

# ======================================================
# 6. SCREENER LOGIC (ENHANCED)
# ======================================================
def analyze_stock(ticker: str) -> Optional[Dict]:
    """
    Menganalisis satu saham berdasarkan kriteria BSJP V2.

    MAIN CRITERIA (semua wajib terpenuhi):
      cond1: Close >= High * 0.98
      cond2: Volume > 1000
      cond3: Volume > Volume MA 5
      cond4: Price Change % >= 3

    ADDITIONAL FILTERS (skor, min 3/4):
      cond5: RSI(14) < 80
      cond6: Value >= 1 Miliar
      cond7: Close > MA 20
      cond8: Volume >= 2x Volume MA 20

    Returns dict dengan semua metrik jika lolos, None jika tidak.
    """
    df = fetch_stock_data(ticker)

    # Butuh minimal 21 data point untuk MA20 yang akurat
    if df.empty or len(df) < (MA20_PERIOD + 1):
        return None

    current = df.iloc[-1]

    # Validasi data — pastikan harga valid
    if current['Close'] <= 0 or current['High'] <= 0 or current['Volume'] <= 0:
        return None

    prev_close = df['Close'].iloc[-2]
    if prev_close <= 0:
        return None

    # ============================================
    # MAIN CRITERIA — Semua harus terpenuhi
    # ============================================

    # Cond 1: Close dekat High (diperketat dari 0.97 ke 0.98)
    cond1 = current['Close'] >= (current['High'] * CLOSE_HIGH_RATIO)

    # Cond 2: Volume (proxy Frequency) > 1000
    current_vol = float(current['Volume'])
    cond2 = current_vol > MIN_FREQUENCY

    # Cond 3: Volume > Volume MA 5
    vol_ma5 = df['Volume'].rolling(window=MA5_PERIOD).mean().iloc[-1]
    if pd.isna(vol_ma5):
        return None
    cond3 = current_vol > vol_ma5

    # Cond 4: Price Change % >= 3
    price_change_pct = ((current['Close'] - prev_close) / prev_close) * 100
    cond4 = price_change_pct >= MIN_PRICE_CHANGE_PCT

    # Jika kriteria utama tidak lolos, langsung return None
    if not all([cond1, cond2, cond3, cond4]):
        return None

    # ============================================
    # ADDITIONAL NOISE FILTERS — Scoring
    # ============================================

    # Cond 5: RSI(14) < 80 (Anti-Overbought)
    rsi_value = calculate_rsi(df['Close'], RSI_PERIOD)
    if rsi_value < 0:
        # RSI gagal dihitung, anggap tidak lolos
        cond5 = False
    else:
        cond5 = rsi_value < RSI_MAX

    # Cond 6: Value transaksi >= 1 Miliar IDR
    value_today = current['Close'] * current['Volume']
    cond6 = value_today >= MIN_VALUE_IDR

    # Cond 7: Close > MA 20 (Uptrend Confirmation)
    ma20 = df['Close'].rolling(window=MA20_PERIOD).mean().iloc[-1]
    if pd.isna(ma20):
        cond7 = False
    else:
        cond7 = current['Close'] > ma20

    # Cond 8: Volume Spike — Volume >= 2x Volume MA 20
    vol_ma20 = df['Volume'].rolling(window=MA20_PERIOD).mean().iloc[-1]
    if pd.isna(vol_ma20) or vol_ma20 == 0:
        cond8 = False
    else:
        cond8 = current_vol >= (VOL_SPIKE_MULTIPLIER * vol_ma20)

    # Hitung skor tambahan
    extra_filters = [cond5, cond6, cond7, cond8]
    extra_score = sum(extra_filters)

    # Minimum skor 3 dari 4 filter tambahan
    if extra_score < MIN_EXTRA_SCORE:
        return None

    # ============================================
    # RISK MANAGEMENT CALCULATIONS
    # ============================================
    entry_price = float(current['Close'])
    tp_price = entry_price * (1 + TP_PERCENT)
    cl_price = entry_price * (1 - CL_PERCENT)

    # ============================================
    # BUILD RESULT
    # ============================================
    result = {
        "ticker": ticker,
        "close": int(current['Close']),
        "high": int(current['High']),
        "low": int(current['Low']),
        "open": int(current['Open']),
        "change_pct": round(float(price_change_pct), 2),
        "volume": int(current_vol),
        "vol_ma5": int(vol_ma5),
        "vol_ma20": int(vol_ma20) if not pd.isna(vol_ma20) else 0,
        "rsi": rsi_value if rsi_value >= 0 else 0.0,
        "value_idr": float(value_today),
        "value_b": round(float(value_today) / 1e9, 2),
        "ma20": round(float(ma20), 0) if not pd.isna(ma20) else 0,
        "above_ma20": cond7,
        "vol_spike": cond8,
        "rsi_ok": cond5,
        "value_ok": cond6,
        "extra_score": extra_score,
        "entry": int(entry_price),
        "tp": int(tp_price),
        "cl": int(cl_price),
        "status": "MATCH"
    }

    return result

# ======================================================
# 7. OUTPUT FORMATTERS
# ======================================================
def format_terminal_table(results: List[Dict]) -> str:
    """
    Membuat tabel terformat untuk output di terminal.
    Menampilkan semua metrik termasuk RSI, Value, MA20, Vol Spike, Score, TP, CL.
    """
    if not results:
        return "Tidak ada hasil."

    # Header
    header = (
        f"{'No':>3} | {'Ticker':<6} | {'Close':>7} | {'Chg%':>6} | "
        f"{'RSI':>5} | {'Val(B)':>7} | {'MA20':>2} | {'VSpk':>4} | "
        f"{'Score':>5} | {'Entry':>7} | {'TP(8%)':>7} | {'CL(5%)':>7}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for i, r in enumerate(results, 1):
        ma20_icon = "Y" if r['above_ma20'] else "N"
        vspk_icon = "Y" if r['vol_spike'] else "N"
        star = " *" if r['extra_score'] == 4 else ""

        line = (
            f"{i:>3} | {r['ticker']:<6} | {r['close']:>7,} | "
            f"{r['change_pct']:>+5.1f}% | {r['rsi']:>5.1f} | "
            f"{r['value_b']:>6.2f}B | "
            f"  {ma20_icon:<1} | "
            f"  {vspk_icon:<1} | "
            f" {r['extra_score']}/4{star} | "
            f"{r['entry']:>7,} | {r['tp']:>7,} | {r['cl']:>7,}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def format_telegram_message(results: List[Dict], scan_time: str,
                            total_scanned: int, total_skipped: int,
                            blacklisted_count: int) -> str:
    """
    Membuat pesan Telegram dengan format HTML.
    Termasuk header, detail per saham, dan summary statistics.
    """
    now_str = scan_time

    # === HEADER ===
    msg_lines = [
        f"<b>BSJP Scanner V{VERSION}</b>",
        f"<i>{now_str}</i>",
        "",
        f"Scanned: {total_scanned} | Blacklisted: {blacklisted_count} | Skipped: {total_skipped}",
        f"<b>Match: {len(results)} saham</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not results:
        msg_lines.append("")
        msg_lines.append("<i>Tidak ada saham yang lolos semua filter.</i>")
        return "\n".join(msg_lines)

    # === DETAIL PER SAHAM ===
    for r in results:
        star = " ⭐" if r['extra_score'] == 4 else ""
        ma20_icon = "✅" if r['above_ma20'] else "❌"
        vspk_icon = "✅" if r['vol_spike'] else "❌"

        detail_line = (
            f"<b>{r['ticker']}</b> | "
            f"{r['change_pct']:+.2f}% | "
            f"RSI: {r['rsi']:.0f} | "
            f"Score: {r['extra_score']}/4{star} | "
            f"Val: {r['value_b']:.2f}B"
        )
        risk_line = (
            f"   Entry: {r['entry']:,} → "
            f"TP: {r['tp']:,} | CL: {r['cl']:,}"
        )
        filter_line = (
            f"   MA20: {ma20_icon} | VolSpike: {vspk_icon}"
        )

        msg_lines.append(detail_line)
        msg_lines.append(risk_line)
        msg_lines.append(filter_line)
        msg_lines.append("")

    # === SUMMARY STATISTICS ===
    avg_rsi = sum(r['rsi'] for r in results) / len(results)
    avg_change = sum(r['change_pct'] for r in results) / len(results)
    total_value = sum(r['value_b'] for r in results)
    perfect_score_count = sum(1 for r in results if r['extra_score'] == 4)

    msg_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    msg_lines.append("<b>SUMMARY</b>")
    msg_lines.append(f"Avg RSI: {avg_rsi:.1f} | Avg Chg: {avg_change:+.2f}%")
    msg_lines.append(f"Total Value: {total_value:.2f}B IDR")
    if perfect_score_count > 0:
        msg_lines.append(f"⭐ Perfect Score (4/4): {perfect_score_count} saham")
    msg_lines.append("")
    msg_lines.append(f"<i>TP: +{TP_PERCENT*100:.0f}% | CL: -{CL_PERCENT*100:.0f}%</i>")

    return "\n".join(msg_lines)


def print_summary_stats(results: List[Dict], total_scanned: int,
                        total_skipped: int, blacklisted_count: int):
    """
    Menampilkan ringkasan statistik di terminal.
    """
    print("\n" + "=" * 60)
    print("RINGKASAN STATISTIK")
    print("=" * 60)
    print(f"  Total Saham di-scan    : {total_scanned}")
    print(f"  Blacklisted (skip)     : {blacklisted_count}")
    print(f"  Gagal fetch / skip     : {total_skipped}")
    print(f"  Total MATCH            : {len(results)}")
    print("-" * 40)

    if results:
        avg_rsi = sum(r['rsi'] for r in results) / len(results)
        avg_change = sum(r['change_pct'] for r in results) / len(results)
        max_change = max(r['change_pct'] for r in results)
        min_change = min(r['change_pct'] for r in results)
        total_value = sum(r['value_b'] for r in results)
        perfect_count = sum(1 for r in results if r['extra_score'] == 4)
        score3_count = sum(1 for r in results if r['extra_score'] == 3)

        print(f"  Rata-rata RSI          : {avg_rsi:.1f}")
        print(f"  Rata-rata Change %     : {avg_change:+.2f}%")
        print(f"  Change tertinggi       : {max_change:+.2f}%")
        print(f"  Change terendah        : {min_change:+.2f}%")
        print(f"  Total Value (Miliar)   : {total_value:.2f}B IDR")
        print(f"  Score 4/4 (Perfect)    : {perfect_count} saham")
        print(f"  Score 3/4              : {score3_count} saham")
    else:
        print("  Tidak ada saham yang lolos filter.")

    print("=" * 60)

# ======================================================
# 8. MAIN EXECUTION
# ======================================================
def run_scanner():
    """
    Fungsi utama: menjalankan scanner BSJP V2.
    Flow:
      1. Load tickers dari CSV
      2. Load blacklist (opsional)
      3. Scan setiap saham
      4. Format dan tampilkan hasil di terminal
      5. Kirim notifikasi ke Telegram
    """
    scan_start = dt.datetime.now()
    scan_time_str = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 60)
    print(f"  BSJP SCANNER V{VERSION}")
    print(f"  {scan_time_str}")
    print("=" * 60)
    logger.info(f"Scanner BSJP V{VERSION} dimulai.")
    print()

    # --- 1. Load Tickers ---
    tickers = load_tickers_from_csv()
    if not tickers:
        logger.error("Tidak ada saham untuk di-scan. Pastikan data/data.csv tersedia.")
        return

    total_scanned = len(tickers)
    logger.info(f"Total saham untuk di-scan: {total_scanned}")

    # --- 2. Load Blacklist ---
    blacklist = load_blacklist()
    blacklisted_count = 0

    # Filter out blacklisted tickers
    if blacklist:
        original_count = len(tickers)
        tickers = [t for t in tickers if t not in blacklist]
        blacklisted_count = original_count - len(tickers)
        if blacklisted_count > 0:
            logger.info(f"{blacklisted_count} saham dilewati karena blacklist.")

    # --- 3. Scan Setiap Saham ---
    results = []
    skipped = 0

    print(f"\nMemulai scanning {len(tickers)} saham...\n")

    for i, ticker in enumerate(tickers):
        # Progress indicator
        progress_pct = ((i + 1) / len(tickers)) * 100
        print(f"\r  [{progress_pct:5.1f}%] Scanning {i+1}/{len(tickers)}: {ticker:<6}", end="", flush=True)

        try:
            res = analyze_stock(ticker)
            if res:
                logger.info(
                    f"HIT: {ticker} | Chg: {res['change_pct']:+.2f}% | "
                    f"RSI: {res['rsi']:.1f} | Score: {res['extra_score']}/4 | "
                    f"Value: {res['value_b']:.2f}B"
                )
                print(f"\n  ✅ HIT: {ticker} (+{res['change_pct']:.2f}%, RSI:{res['rsi']:.0f}, Score:{res['extra_score']}/4)")
                results.append(res)
        except KeyboardInterrupt:
            print("\n\n⚠️ Scanner dihentikan oleh user.")
            logger.warning("Scanner dihentikan oleh user (KeyboardInterrupt).")
            break
        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            skipped += 1
            continue

    # Clear progress line
    print("\r" + " " * 60 + "\r", end="")

    # --- 4. Sort Results ---
    # Urutkan: Score tertinggi dulu, lalu Change % tertinggi
    results.sort(key=lambda x: (-x['extra_score'], -x['change_pct']))

    # --- 5. Terminal Output ---
    print(f"\n{'='*60}")
    print(f"  SCAN SELESAI — Ditemukan: {len(results)} saham")
    print(f"{'='*60}")

    if results:
        table_output = format_terminal_table(results)
        print(f"\n{table_output}")

    print_summary_stats(results, total_scanned, skipped, blacklisted_count)

    # --- 6. Telegram Notification ---
    if TELEGRAM_OK:
        logger.info("Mengirim hasil ke Telegram...")
        telegram_msg = format_telegram_message(
            results=results,
            scan_time=scan_time_str,
            total_scanned=total_scanned,
            total_skipped=skipped,
            blacklisted_count=blacklisted_count
        )
        success = send_telegram_message(telegram_msg)
        if success:
            print("\n✅ Hasil terkirim ke Telegram.")
        else:
            print("\n❌ Gagal mengirim ke Telegram.")
    else:
        print("\nℹ️  Telegram tidak dikonfigurasi. Hasil hanya ditampilkan di terminal.")

    # --- 7. Timing ---
    elapsed = (dt.datetime.now() - scan_start).total_seconds()
    logger.info(f"Scanner selesai dalam {elapsed:.1f} detik.")
    print(f"\n⏱️  Waktu eksekusi: {elapsed:.1f} detik")
    print()


# ======================================================
# 9. ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_scanner()