# ================================================================
# scanner_bsjp_v2_combined.py - BSJP V2 + 3 SCREENER COMBINED
# Version : 2.1.0 (Combined Edition)
# Author  : Grok + Team (Merlin AI Assistant base + 3 screener integration)
# Date    : 2026-04-27
# Desc    : Scanner saham IDX BSJP (Beli Sore Jual Pagi) yang sudah DIGABUNGKAN
#           dengan 3 set Screening Rules dari screenshot yang kamu attach.
#           TOTAL 4 FILTER UTAMA GABUNGAN (semua harus terpenuhi + scoring).
#
# 4 FILTER GABUNGAN (kombinasi BSJP + 3 Screener):
#   1. PRICE ACTION & TREND (Close near High + Change% + >MA20 + >MA50)
#   2. VOLUME & LIQUIDITY SPIKE (Vol > MA5 + Vol > 2x MA20 + Vol > 1000)
#   3. VALUE STRENGTH (Value >= 1M & Value > Value MA20) ← proxy Bandar Value
#   4. TECHNICAL FILTER (RSI(14) < 80 + anti-overbought)
#
# Catatan penting:
# - Bandar Value (dari Stockbit/RTI) TIDAK tersedia di yfinance → kita gunakan
#   total Value (Close × Volume) sebagai proxy yang paling mendekati.
# - Semua rule dari 3 screenshot sudah diintegrasikan (Volume 2x MA20,
#   Value >=100jt/1M, Value > Value MA20, Price > MA20/MA50, dll).
# - Scoring tetap dipertahankan (min 3 dari 4 additional filters).
#
# SUMBER DATA : data/data.csv
# BLACKLIST   : data/blacklist.csv
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
    f"scanner_combined_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

logger = logging.getLogger("BSJP_V2_COMBINED")

# ======================================================
# 1. CONSTANTS - 4 FILTER GABUNGAN
# ======================================================
VERSION = "2.1.0"

# Main Criteria Thresholds (Filter 1 & 2)
CLOSE_HIGH_RATIO = 0.98
MIN_FREQUENCY = 1000
MIN_PRICE_CHANGE_PCT = 3

# Noise Filter Thresholds (Filter 3 & 4)
RSI_MAX = 80
MIN_VALUE_IDR = 1_000_000_000          # 1 Miliar (dari screener 2) - bisa diubah ke 100_000_000
VOL_SPIKE_MULTIPLIER = 2
VALUE_MA20_MULTIPLIER = 1.0            # Value > 1x Value MA20 (screener 3)

# MA Periods
RSI_PERIOD = 14
MA20_PERIOD = 20
MA5_PERIOD = 5
MA50_PERIOD = 50                       # BARU dari screener 3

# Scoring
MIN_EXTRA_SCORE = 3                    # Minimum 3 dari 4 filter tambahan

# Risk Management
TP_PERCENT = 0.08
CL_PERCENT = 0.05

# Telegram
MAX_MESSAGE_LENGTH = 4096

# Data fetch
YFINANCE_PERIOD = "3mo"                # Diperpanjang untuk MA50 yang akurat
YFINANCE_INTERVAL = "1d"

# ======================================================
# 2. TELEGRAM CONFIG (sama seperti sebelumnya)
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
# 3. TELEGRAM HELPER (tidak diubah)
# ======================================================
def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        logger.info("Telegram tidak aktif, pesan tidak dikirim.")
        return False

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
                logger.error(f"Gagal kirim Telegram (Status {response.status_code}): {response.text}")
                all_success = False
        except Exception as e:
            logger.error(f"Error koneksi Telegram: {e}")
            all_success = False

        if len(messages_to_send) > 1 and i < len(messages_to_send) - 1:
            time.sleep(1)

    return all_success


def split_telegram_message(message: str) -> List[str]:
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]

    chunks = []
    current_chunk = ""
    lines = message.split("\n")

    for line in lines:
        if len(line) > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for j in range(0, len(line), MAX_MESSAGE_LENGTH):
                chunks.append(line[j:j + MAX_MESSAGE_LENGTH])
            continue

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
# 4. DATA HELPERS (tidak diubah)
# ======================================================
def load_tickers_from_csv() -> List[str]:
    file_path = os.path.join("data", "data.csv")
    if not os.path.exists(file_path):
        if os.path.exists("data.csv"):
            file_path = "data.csv"
        else:
            logger.error(f"File data.csv tidak ditemukan.")
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
    except Exception as e:
        logger.error(f"Error membaca '{file_path}': {e}")
        return []


def load_blacklist() -> List[str]:
    file_path = os.path.join("data", "blacklist.csv")
    if not os.path.exists(file_path):
        logger.info("File blacklist.csv tidak ditemukan.")
        return []

    try:
        df = pd.read_csv(file_path)
        possible_cols = ['Ticker', 'ticker', 'Kode', 'kode', 'Code', 'code', 'Symbol', 'symbol']
        found_col = next((c for c in possible_cols if c in df.columns), df.columns[0])

        tickers = df[found_col].dropna().astype(str).tolist()
        cleaned = list(set([t.strip().upper() for t in tickers if len(t.strip()) >= 4]))
        logger.info(f"Blacklist aktif: {len(cleaned)} saham")
        return cleaned
    except Exception as e:
        logger.warning(f"Error membaca blacklist: {e}")
        return []


def fetch_stock_data(ticker: str) -> pd.DataFrame:
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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        result = df[required_cols].copy()
        result.dropna(inplace=True)
        return result
    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# ======================================================
# 5. TECHNICAL INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    try:
        if len(series) < period + 1:
            return -1.0

        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        current_avg_gain = avg_gain.iloc[-1]
        current_avg_loss = avg_loss.iloc[-1]

        if current_avg_loss == 0:
            return 100.0 if current_avg_gain > 0 else 50.0

        rs = current_avg_gain / current_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return round(rsi, 1)
    except Exception as e:
        logger.debug(f"Error menghitung RSI: {e}")
        return -1.0

# ======================================================
# 6. SCREENER LOGIC - 4 FILTER GABUNGAN
# ======================================================
def analyze_stock(ticker: str) -> Optional[Dict]:
    df = fetch_stock_data(ticker)
    if df.empty or len(df) < MA50_PERIOD + 1:   # Butuh data cukup untuk MA50
        return None

    current = df.iloc[-1]
    if current['Close'] <= 0 or current['High'] <= 0 or current['Volume'] <= 0:
        return None

    prev_close = df['Close'].iloc[-2]
    if prev_close <= 0:
        return None

    current_vol = float(current['Volume'])
    value_today = current['Close'] * current_vol

    # ============================================
    # FILTER 1: PRICE ACTION & TREND (dari BSJP + screener 3)
    # ============================================
    cond1 = current['Close'] >= (current['High'] * CLOSE_HIGH_RATIO)
    price_change_pct = ((current['Close'] - prev_close) / prev_close) * 100
    cond4 = price_change_pct >= MIN_PRICE_CHANGE_PCT

    # MA20 & MA50 (screener 3)
    ma20 = df['Close'].rolling(window=MA20_PERIOD).mean().iloc[-1]
    ma50 = df['Close'].rolling(window=MA50_PERIOD).mean().iloc[-1]
    cond_ma20 = not pd.isna(ma20) and current['Close'] > ma20
    cond_ma50 = not pd.isna(ma50) and current['Close'] > ma50
    cond_trend = cond_ma20 and cond_ma50

    # ============================================
    # FILTER 2: VOLUME & LIQUIDITY SPIKE (dari semua screener)
    # ============================================
    cond2 = current_vol > MIN_FREQUENCY
    vol_ma5 = df['Volume'].rolling(window=MA5_PERIOD).mean().iloc[-1]
    cond3 = not pd.isna(vol_ma5) and current_vol > vol_ma5
    vol_ma20 = df['Volume'].rolling(window=MA20_PERIOD).mean().iloc[-1]
    cond_vol_spike = not pd.isna(vol_ma20) and vol_ma20 > 0 and current_vol >= (VOL_SPIKE_MULTIPLIER * vol_ma20)

    # ============================================
    # FILTER 3: VALUE STRENGTH (dari screener 1, 2, 3 - proxy Bandar Value)
    # ============================================
    value_series = df['Close'] * df['Volume']
    value_ma20 = value_series.rolling(window=MA20_PERIOD).mean().iloc[-1]
    cond_value_ma20 = not pd.isna(value_ma20) and value_today > (VALUE_MA20_MULTIPLIER * value_ma20)
    cond_value = value_today >= MIN_VALUE_IDR and cond_value_ma20

    # ============================================
    # FILTER 4: TECHNICAL FILTER (RSI dari BSJP)
    # ============================================
    rsi_value = calculate_rsi(df['Close'], RSI_PERIOD)
    cond_rsi = rsi_value >= 0 and rsi_value < RSI_MAX

    # MAIN CRITERIA (semua harus terpenuhi)
    if not all([cond1, cond2, cond3, cond4, cond_vol_spike, cond_trend, cond_value, cond_rsi]):
        return None

    # ============================================
    # SCORING (untuk additional confirmation)
    # ============================================
    extra_filters = [cond_rsi, cond_value, cond_trend, cond_vol_spike]
    extra_score = sum(extra_filters)

    if extra_score < MIN_EXTRA_SCORE:
        return None

    # Risk Management
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
        "vol_ma5": int(vol_ma5) if not pd.isna(vol_ma5) else 0,
        "vol_ma20": int(vol_ma20) if not pd.isna(vol_ma20) else 0,
        "rsi": rsi_value if rsi_value >= 0 else 0.0,
        "value_idr": float(value_today),
        "value_b": round(float(value_today) / 1e9, 2),
        "value_ma20_b": round(float(value_ma20) / 1e9, 2) if not pd.isna(value_ma20) else 0,
        "ma20": round(float(ma20), 0) if not pd.isna(ma20) else 0,
        "ma50": round(float(ma50), 0) if not pd.isna(ma50) else 0,
        "above_ma20": cond_ma20,
        "above_ma50": cond_ma50,
        "vol_spike": cond_vol_spike,
        "value_strength": cond_value,
        "extra_score": extra_score,
        "entry": int(entry_price),
        "tp": int(tp_price),
        "cl": int(cl_price),
        "status": "MATCH"
    }

    return result

# ======================================================
# 7. OUTPUT FORMATTERS (di-update dengan kolom baru)
# ======================================================
def format_terminal_table(results: List[Dict]) -> str:
    if not results:
        return "Tidak ada hasil."

    header = (
        f"{'No':>3} | {'Ticker':<6} | {'Close':>7} | {'Chg%':>6} | "
        f"{'RSI':>5} | {'Val(B)':>7} | {'MA20':>4} | {'MA50':>4} | "
        f"{'VSpk':>4} | {'Score':>5} | {'Entry':>7} | {'TP':>7} | {'CL':>7}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for i, r in enumerate(results, 1):
        ma20_icon = "Y" if r.get('above_ma20', False) else "N"
        ma50_icon = "Y" if r.get('above_ma50', False) else "N"
        vspk_icon = "Y" if r.get('vol_spike', False) else "N"
        star = " *" if r['extra_score'] == 4 else ""

        line = (
            f"{i:>3} | {r['ticker']:<6} | {r['close']:>7,} | "
            f"{r['change_pct']:>+5.1f}% | {r['rsi']:>5.1f} | "
            f"{r['value_b']:>6.2f}B | "
            f"{ma20_icon:>1}/{ma50_icon:<1} | "
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
    now_str = scan_time

    msg_lines = [
        f"<b>GROK SCREENER V{VERSION}</b>",
        f"<i>{now_str}</i>",
        "",
        f"Scanned: {total_scanned} | Blacklisted: {blacklisted_count} | Skipped: {total_skipped}",
        f"<b>Match: {len(results)} saham (4 Filter Gabungan)</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not results:
        msg_lines.append("<i>Tidak ada saham yang lolos 4 filter gabungan.</i>")
        return "\n".join(msg_lines)

    for r in results:
        star = " ⭐" if r['extra_score'] == 4 else ""
        ma_icon = "✅" if r.get('above_ma20', False) and r.get('above_ma50', False) else "❌"
        vspk_icon = "✅" if r.get('vol_spike', False) else "❌"

        detail_line = (
            f"<b>{r['ticker']}</b> | "
            f"{r['change_pct']:+.2f}% | "
            f"RSI: {r['rsi']:.0f} | "
            f"Score: {r['extra_score']}/4{star} | "
            f"Val: {r['value_b']:.2f}B"
        )
        risk_line = f"   Entry: {r['entry']:,} → TP: {r['tp']:,} | CL: {r['cl']:,}"
        filter_line = f"   MA20/MA50: {ma_icon} | VolSpike: {vspk_icon} | Value > MA20: ✅"

        msg_lines.append(detail_line)
        msg_lines.append(risk_line)
        msg_lines.append(filter_line)
        msg_lines.append("")

    # Summary
    avg_rsi = sum(r['rsi'] for r in results) / len(results)
    avg_change = sum(r['change_pct'] for r in results) / len(results)
    total_value = sum(r['value_b'] for r in results)
    perfect_score_count = sum(1 for r in results if r['extra_score'] == 4)

    msg_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    msg_lines.append("<b>SUMMARY 4 FILTER GABUNGAN</b>")
    msg_lines.append(f"Avg RSI: {avg_rsi:.1f} | Avg Chg: {avg_change:+.2f}%")
    msg_lines.append(f"Total Value: {total_value:.2f}B IDR")
    if perfect_score_count > 0:
        msg_lines.append(f"⭐ Perfect Score (4/4): {perfect_score_count}")
    msg_lines.append(f"<i>TP: +{TP_PERCENT*100:.0f}% | CL: -{CL_PERCENT*100:.0f}%</i>")

    return "\n".join(msg_lines)


def print_summary_stats(results: List[Dict], total_scanned: int,
                        total_skipped: int, blacklisted_count: int):
    print("\n" + "=" * 70)
    print("RINGKASAN STATISTIK - 4 FILTER GABUNGAN")
    print("=" * 70)
    print(f"  Total Saham di-scan    : {total_scanned}")
    print(f"  Blacklisted            : {blacklisted_count}")
    print(f"  Skipped                : {total_skipped}")
    print(f"  TOTAL MATCH (4 Filter) : {len(results)}")
    print("-" * 40)

    if results:
        avg_rsi = sum(r['rsi'] for r in results) / len(results)
        avg_change = sum(r['change_pct'] for r in results) / len(results)
        total_value = sum(r['value_b'] for r in results)
        perfect_count = sum(1 for r in results if r['extra_score'] == 4)

        print(f"  Rata-rata RSI          : {avg_rsi:.1f}")
        print(f"  Rata-rata Change %     : {avg_change:+.2f}%")
        print(f"  Total Value            : {total_value:.2f}B IDR")
        print(f"  Perfect Score (4/4)    : {perfect_count} saham")
    else:
        print("  Tidak ada saham yang lolos 4 filter gabungan.")

    print("=" * 70)

# ======================================================
# 8. MAIN EXECUTION (sama seperti sebelumnya)
# ======================================================
def run_scanner():
    scan_start = dt.datetime.now()
    scan_time_str = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 70)
    print(f"  BSJP SCANNER V{VERSION} - 4 FILTER GABUNGAN")
    print(f"  {scan_time_str}")
    print("=" * 70)
    logger.info(f"Scanner BSJP Combined V{VERSION} dimulai.")

    tickers = load_tickers_from_csv()
    if not tickers:
        logger.error("Tidak ada saham untuk di-scan.")
        return

    total_scanned = len(tickers)
    blacklist = load_blacklist()
    blacklisted_count = 0
    if blacklist:
        original_count = len(tickers)
        tickers = [t for t in tickers if t not in blacklist]
        blacklisted_count = original_count - len(tickers)

    results = []
    skipped = 0

    print(f"\nMemulai scanning {len(tickers)} saham dengan 4 filter gabungan...\n")

    for i, ticker in enumerate(tickers):
        progress_pct = ((i + 1) / len(tickers)) * 100
        print(f"\r  [{progress_pct:5.1f}%] Scanning {i+1}/{len(tickers)}: {ticker:<6}", end="", flush=True)

        try:
            res = analyze_stock(ticker)
            if res:
                logger.info(f"HIT: {ticker} | Chg: {res['change_pct']:+.2f}% | RSI: {res['rsi']:.1f} | Score: {res['extra_score']}/4 | Value: {res['value_b']:.2f}B")
                print(f"\n  ✅ HIT: {ticker} (+{res['change_pct']:.2f}%, RSI:{res['rsi']:.0f}, Score:{res['extra_score']}/4)")
                results.append(res)
        except KeyboardInterrupt:
            print("\n\n⚠️ Scanner dihentikan oleh user.")
            break
        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            skipped += 1
            continue

    print("\r" + " " * 80 + "\r", end="")

    results.sort(key=lambda x: (-x['extra_score'], -x['change_pct']))

    print(f"\n{'='*70}")
    print(f"  SCAN SELESAI — Ditemukan: {len(results)} saham (4 Filter Gabungan)")
    print(f"{'='*70}")

    if results:
        table_output = format_terminal_table(results)
        print(f"\n{table_output}")

    print_summary_stats(results, total_scanned, skipped, blacklisted_count)

    if TELEGRAM_OK:
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
        print("\nℹ️  Telegram tidak dikonfigurasi.")

    elapsed = (dt.datetime.now() - scan_start).total_seconds()
    logger.info(f"Scanner selesai dalam {elapsed:.1f} detik.")
    print(f"\n⏱️  Waktu eksekusi: {elapsed:.1f} detik")


# ======================================================
# 9. ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_scanner()