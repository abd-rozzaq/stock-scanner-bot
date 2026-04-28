# ================================================================
# screener.py - COMBINED MULTI-SCREENER IDX STOCK SCANNER
# Version : 3.0.0
# Author  : Merlin AI Assistant
# Date    : 2026-04-27
# Desc    : Scanner saham IDX gabungan 4 set filter:
#
# ── FILTER SET ORIGINAL (BSJP V2) ──────────────────────────────
#   MAIN CRITERIA (semua wajib):
#     1. Close >= High * 0.98
#     2. Volume > 1000
#     3. Volume > Volume MA 5
#     4. Price Change % >= 3
#
#   NOISE FILTERS (scoring, min 3 dari 4):
#     5. RSI(14) < 80
#     6. Value >= 1 Miliar
#     7. Close > MA 20
#     8. Volume >= 2x Vol MA20
#
# ── FILTER SET 1 — BANDAR SCREENER ────────────────────────────
#     B1. Bandar Value > 1 × Bandar Value MA 20
#     B2. Value MA 20 > 1.000.000.000
#     B3. Previous Bandar Value <= 1 × Bandar Value  (akumulasi baru)
#     B4. Bandar Value MA 10 > 1 × Bandar Value MA 20
#
# ── FILTER SET 2 — TREND SCREENER ─────────────────────────────
#     T1. Price > 1 × Price MA 20
#     T2. Price > 1 × Price MA 50
#     T3. Volume >= 2 × Volume MA 20
#     T4. Value > 1 × Value MA 20
#
# ── FILTER SET 3 — LIKUIDITAS SCREENER ────────────────────────
#     L1. Volume > 2 × Volume MA 20
#     L2. Value >= 100.000.000 (100 Juta)
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

logger = logging.getLogger("COMBINED_SCREENER")

# ======================================================
# 1. CONSTANTS
# ======================================================
VERSION = "3.0.0"

# ── Original BSJP Thresholds ──
CLOSE_HIGH_RATIO        = 0.98
MIN_FREQUENCY           = 1000
MIN_PRICE_CHANGE_PCT    = 3
RSI_MAX                 = 80
MIN_VALUE_IDR           = 1_000_000_000    # 1 Miliar
VOL_SPIKE_MULTIPLIER    = 2
MIN_EXTRA_SCORE         = 3

# ── Bandar Screener Thresholds ──
BANDAR_VALUE_MA_PERIOD  = 20               # periode MA untuk Bandar Value
BANDAR_VALUE_MA10_PER   = 10
BANDAR_VALUE_MA20_MIN   = 1_000_000_000    # Value MA20 > 1 Miliar

# ── Trend Screener Thresholds ──
MA50_PERIOD             = 50

# ── Likuiditas Screener Thresholds ──
MIN_VALUE_LIKUIDITAS    = 100_000_000      # 100 Juta

# ── Risk Management ──
TP_PERCENT              = 0.08
CL_PERCENT              = 0.05

# ── Technical Periods ──
RSI_PERIOD              = 14
MA20_PERIOD             = 20
MA5_PERIOD              = 5

# ── Telegram ──
MAX_MESSAGE_LENGTH      = 4096

# ── Data Fetch ──
YFINANCE_PERIOD         = "3mo"            # 3 bulan untuk MA50
YFINANCE_INTERVAL       = "1d"

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
        except requests.exceptions.Timeout:
            logger.error("Timeout saat mengirim pesan Telegram.")
            all_success = False
        except requests.exceptions.ConnectionError:
            logger.error("Tidak dapat terhubung ke server Telegram.")
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
# 4. DATA HELPERS
# ======================================================
def load_tickers_from_csv() -> List[str]:
    file_path = os.path.join("data", "data.csv")
    if not os.path.exists(file_path):
        if os.path.exists("data.csv"):
            file_path = "data.csv"
        else:
            logger.error("File data.csv tidak ditemukan di 'data/' maupun root directory.")
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
        logger.info(f"Blacklist aktif: {len(cleaned)} saham")
        return cleaned
    except Exception as e:
        logger.warning(f"Error membaca blacklist: {e}")
        return []


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    Mengambil data historis saham dari Yahoo Finance.
    Period diperpanjang ke 3mo agar MA50 bisa dihitung.
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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"{ticker}: Kolom data tidak lengkap.")
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
        return round(100.0 - (100.0 / (1.0 + rs)), 1)
    except Exception as e:
        logger.debug(f"Error menghitung RSI: {e}")
        return -1.0


def compute_bandar_value(df: pd.DataFrame) -> pd.Series:
    """
    Proxy Bandar Value = Close * Volume (nilai transaksi harian).
    Dalam konteks IDX, "Bandar Value" merepresentasikan net value flow.
    Jika kamu punya sumber data Bandar Value asli (misal RTI/Stockbit),
    ganti fungsi ini dengan data tersebut.
    """
    return df['Close'] * df['Volume']


# ======================================================
# 6. COMBINED SCREENER LOGIC
# ======================================================
def analyze_stock(ticker: str) -> Optional[Dict]:
    """
    Menganalisis satu saham berdasarkan GABUNGAN 4 set filter:

    ── SET ORIGINAL (BSJP V2) ──────────────────────────────────
      MAIN (semua wajib):
        cond1: Close >= High * 0.98
        cond2: Volume > 1000
        cond3: Volume > Volume MA 5
        cond4: Price Change % >= 3

      NOISE FILTERS (scoring, min 3/4):
        cond5: RSI(14) < 80
        cond6: Value >= 1 Miliar
        cond7: Close > MA 20
        cond8: Volume >= 2x Volume MA 20

    ── SET 1 — BANDAR SCREENER ──────────────────────────────────
        B1: Bandar Value > 1 × Bandar Value MA 20
        B2: Value MA 20 > 1 Miliar
        B3: Previous Bandar Value <= 1 × Bandar Value  (akumulasi baru)
        B4: Bandar Value MA 10 > 1 × Bandar Value MA 20

    ── SET 2 — TREND SCREENER ───────────────────────────────────
        T1: Price > 1 × Price MA 20
        T2: Price > 1 × Price MA 50
        T3: Volume >= 2 × Volume MA 20
        T4: Value > 1 × Value MA 20

    ── SET 3 — LIKUIDITAS SCREENER ──────────────────────────────
        L1: Volume > 2 × Volume MA 20
        L2: Value >= 100 Juta

    Returns dict dengan semua metrik jika lolos SEMUA wajib +
    gabungan scoring, None jika tidak lolos.
    """
    df = fetch_stock_data(ticker)

    # Butuh minimal 51 data point untuk MA50
    if df.empty or len(df) < (MA50_PERIOD + 1):
        return None

    current  = df.iloc[-1]
    prev     = df.iloc[-2]

    if current['Close'] <= 0 or current['High'] <= 0 or current['Volume'] <= 0:
        return None
    if prev['Close'] <= 0:
        return None

    close       = float(current['Close'])
    high        = float(current['High'])
    volume      = float(current['Volume'])
    prev_close  = float(prev['Close'])

    # ── Derived Series ────────────────────────────────────────
    value_series        = df['Close'] * df['Volume']          # Value harian (IDR)
    bandar_series       = compute_bandar_value(df)            # proxy Bandar Value

    value_today         = close * volume
    bandar_today        = float(bandar_series.iloc[-1])
    bandar_prev         = float(bandar_series.iloc[-2])

    vol_ma5             = df['Volume'].rolling(MA5_PERIOD).mean().iloc[-1]
    vol_ma20            = df['Volume'].rolling(MA20_PERIOD).mean().iloc[-1]
    ma20                = df['Close'].rolling(MA20_PERIOD).mean().iloc[-1]
    ma50                = df['Close'].rolling(MA50_PERIOD).mean().iloc[-1]
    value_ma20          = value_series.rolling(MA20_PERIOD).mean().iloc[-1]
    bandar_ma20         = bandar_series.rolling(BANDAR_VALUE_MA_PERIOD).mean().iloc[-1]
    bandar_ma10         = bandar_series.rolling(BANDAR_VALUE_MA10_PER).mean().iloc[-1]

    price_change_pct    = ((close - prev_close) / prev_close) * 100

    # Validasi MA tidak NaN
    if any(pd.isna(x) for x in [vol_ma5, vol_ma20, ma20, ma50, value_ma20, bandar_ma20, bandar_ma10]):
        return None

    # ==========================================================
    # ORIGINAL BSJP — MAIN CRITERIA (semua wajib)
    # ==========================================================
    cond1 = close >= (high * CLOSE_HIGH_RATIO)              # Close dekat High
    cond2 = volume > MIN_FREQUENCY                          # Volume > 1000
    cond3 = volume > float(vol_ma5)                         # Volume > Vol MA5
    cond4 = price_change_pct >= MIN_PRICE_CHANGE_PCT        # Change >= 3%

    if not all([cond1, cond2, cond3, cond4]):
        return None

    # ==========================================================
    # ORIGINAL BSJP — NOISE FILTERS (scoring, min 3/4)
    # ==========================================================
    rsi_value = calculate_rsi(df['Close'], RSI_PERIOD)
    cond5 = (rsi_value >= 0) and (rsi_value < RSI_MAX)                        # RSI < 80
    cond6 = value_today >= MIN_VALUE_IDR                                        # Value >= 1 Miliar
    cond7 = close > float(ma20)                                                 # Close > MA20
    cond8 = volume >= (VOL_SPIKE_MULTIPLIER * float(vol_ma20))                  # Volume >= 2× MA20

    extra_score = sum([cond5, cond6, cond7, cond8])
    if extra_score < MIN_EXTRA_SCORE:
        return None

    # ==========================================================
    # SET 1 — BANDAR SCREENER
    # ==========================================================
    #   B1: Bandar Value > 1 × Bandar Value MA 20
    b1 = bandar_today > float(bandar_ma20)

    #   B2: Value MA 20 > 1 Miliar
    b2 = float(value_ma20) > BANDAR_VALUE_MA20_MIN

    #   B3: Previous Bandar Value <= 1 × Bandar Value (akumulasi baru)
    #       artinya hari ini Bandar Value naik (baru masuk)
    b3 = bandar_prev <= bandar_today

    #   B4: Bandar Value MA 10 > 1 × Bandar Value MA 20
    b4 = float(bandar_ma10) > float(bandar_ma20)

    bandar_pass = all([b1, b2, b3, b4])

    # ==========================================================
    # SET 2 — TREND SCREENER
    # ==========================================================
    #   T1: Price > 1 × Price MA 20
    t1 = close > float(ma20)

    #   T2: Price > 1 × Price MA 50
    t2 = close > float(ma50)

    #   T3: Volume >= 2 × Volume MA 20
    t3 = volume >= (2 * float(vol_ma20))

    #   T4: Value > 1 × Value MA 20
    t4 = value_today > float(value_ma20)

    trend_pass = all([t1, t2, t3, t4])

    # ==========================================================
    # SET 3 — LIKUIDITAS SCREENER
    # ==========================================================
    #   L1: Volume > 2 × Volume MA 20
    l1 = volume > (2 * float(vol_ma20))

    #   L2: Value >= 100 Juta
    l2 = value_today >= MIN_VALUE_LIKUIDITAS

    likuiditas_pass = all([l1, l2])

    # ==========================================================
    # GABUNGAN — saham harus lolos SEMUA screener tambahan
    # ==========================================================
    if not all([bandar_pass, trend_pass, likuiditas_pass]):
        return None

    # ==========================================================
    # RISK MANAGEMENT
    # ==========================================================
    entry_price = close
    tp_price    = entry_price * (1 + TP_PERCENT)
    cl_price    = entry_price * (1 - CL_PERCENT)

    # ==========================================================
    # BUILD RESULT
    # ==========================================================
    result = {
        "ticker":           ticker,
        "close":            int(close),
        "high":             int(high),
        "low":              int(current['Low']),
        "open":             int(current['Open']),
        "change_pct":       round(float(price_change_pct), 2),
        "volume":           int(volume),
        "vol_ma5":          int(vol_ma5),
        "vol_ma20":         int(vol_ma20),

        # Original BSJP
        "rsi":              rsi_value if rsi_value >= 0 else 0.0,
        "value_idr":        float(value_today),
        "value_b":          round(float(value_today) / 1e9, 3),
        "ma20":             round(float(ma20), 0),
        "above_ma20":       cond7,
        "vol_spike":        cond8,
        "rsi_ok":           cond5,
        "value_ok":         cond6,
        "extra_score":      extra_score,

        # Bandar Screener
        "bandar_value_b":   round(bandar_today / 1e9, 3),
        "bandar_ma20_b":    round(float(bandar_ma20) / 1e9, 3),
        "bandar_ma10_b":    round(float(bandar_ma10) / 1e9, 3),
        "bandar_pass":      bandar_pass,
        "b1_val_gt_ma20":   b1,
        "b2_val_ma20_ok":   b2,
        "b3_akumulasi":     b3,
        "b4_ma10_gt_ma20":  b4,

        # Trend Screener
        "ma50":             round(float(ma50), 0),
        "above_ma50":       t2,
        "value_ma20_b":     round(float(value_ma20) / 1e9, 3),
        "trend_pass":       trend_pass,

        # Likuiditas Screener
        "likuiditas_pass":  likuiditas_pass,

        # Risk Management
        "entry":            int(entry_price),
        "tp":               int(tp_price),
        "cl":               int(cl_price),
        "status":           "MATCH"
    }

    return result


# ======================================================
# 7. OUTPUT FORMATTERS
# ======================================================
def format_terminal_table(results: List[Dict]) -> str:
    if not results:
        return "Tidak ada hasil."

    header = (
        f"{'No':>3} | {'Ticker':<6} | {'Close':>7} | {'Chg%':>6} | "
        f"{'RSI':>5} | {'Val(B)':>7} | {'MA20':>2} | {'MA50':>2} | "
        f"{'VSpk':>4} | {'Bandar':>6} | {'Score':>5} | "
        f"{'Entry':>7} | {'TP(8%)':>7} | {'CL(5%)':>7}"
    )
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for i, r in enumerate(results, 1):
        ma20_icon   = "Y" if r['above_ma20']   else "N"
        ma50_icon   = "Y" if r['above_ma50']   else "N"
        vspk_icon   = "Y" if r['vol_spike']     else "N"
        bandar_icon = "✓" if r['bandar_pass']  else "✗"
        star = " *" if r['extra_score'] == 4 else ""

        line = (
            f"{i:>3} | {r['ticker']:<6} | {r['close']:>7,} | "
            f"{r['change_pct']:>+5.1f}% | {r['rsi']:>5.1f} | "
            f"{r['value_b']:>6.3f}B | "
            f"  {ma20_icon:<1} | "
            f"  {ma50_icon:<1} | "
            f"  {vspk_icon:<1} | "
            f"     {bandar_icon:<1} | "
            f" {r['extra_score']}/4{star} | "
            f"{r['entry']:>7,} | {r['tp']:>7,} | {r['cl']:>7,}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def format_telegram_message(results: List[Dict], scan_time: str,
                            total_scanned: int, total_skipped: int,
                            blacklisted_count: int) -> str:
    msg_lines = [
        f"<b>Claude Screener V{VERSION}</b>",
        f"<i>{scan_time}</i>",
        "",
        f"Scanned: {total_scanned} | Blacklisted: {blacklisted_count} | Skipped: {total_skipped}",
        f"<b>Match: {len(results)} saham</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not results:
        msg_lines.append("")
        msg_lines.append("<i>Tidak ada saham yang lolos semua filter.</i>")
        return "\n".join(msg_lines)

    for r in results:
        star        = " ⭐" if r['extra_score'] == 4 else ""
        ma20_icon   = "✅" if r['above_ma20']  else "❌"
        ma50_icon   = "✅" if r['above_ma50']  else "❌"
        vspk_icon   = "✅" if r['vol_spike']   else "❌"
        bandar_icon = "✅" if r['bandar_pass'] else "❌"

        detail_line = (
            f"<b>{r['ticker']}</b> | "
            f"{r['change_pct']:+.2f}% | "
            f"RSI: {r['rsi']:.0f} | "
            f"Score: {r['extra_score']}/4{star} | "
            f"Val: {r['value_b']:.3f}B"
        )
        risk_line = (
            f"   Entry: {r['entry']:,} → "
            f"TP: {r['tp']:,} | CL: {r['cl']:,}"
        )
        filter_line = (
            f"   MA20: {ma20_icon} MA50: {ma50_icon} "
            f"VolSpike: {vspk_icon} Bandar: {bandar_icon}"
        )
        bandar_detail = (
            f"   BandarVal: {r['bandar_value_b']:.3f}B "
            f"(MA20: {r['bandar_ma20_b']:.3f}B | MA10: {r['bandar_ma10_b']:.3f}B)"
        )

        msg_lines.extend([detail_line, risk_line, filter_line, bandar_detail, ""])

    avg_rsi     = sum(r['rsi'] for r in results) / len(results)
    avg_change  = sum(r['change_pct'] for r in results) / len(results)
    total_value = sum(r['value_b'] for r in results)
    perfect     = sum(1 for r in results if r['extra_score'] == 4)

    msg_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    msg_lines.append("<b>SUMMARY</b>")
    msg_lines.append(f"Avg RSI: {avg_rsi:.1f} | Avg Chg: {avg_change:+.2f}%")
    msg_lines.append(f"Total Value: {total_value:.3f}B IDR")
    if perfect > 0:
        msg_lines.append(f"⭐ Perfect Score (4/4): {perfect} saham")
    msg_lines.append("")
    msg_lines.append(f"<i>TP: +{TP_PERCENT*100:.0f}% | CL: -{CL_PERCENT*100:.0f}%</i>")

    return "\n".join(msg_lines)


def print_summary_stats(results: List[Dict], total_scanned: int,
                        total_skipped: int, blacklisted_count: int):
    print("\n" + "=" * 70)
    print("RINGKASAN STATISTIK — COMBINED SCREENER")
    print("=" * 70)
    print(f"  Total Saham di-scan    : {total_scanned}")
    print(f"  Blacklisted (skip)     : {blacklisted_count}")
    print(f"  Gagal fetch / skip     : {total_skipped}")
    print(f"  Total MATCH            : {len(results)}")
    print("-" * 50)

    if results:
        avg_rsi         = sum(r['rsi'] for r in results) / len(results)
        avg_change      = sum(r['change_pct'] for r in results) / len(results)
        max_change      = max(r['change_pct'] for r in results)
        min_change      = min(r['change_pct'] for r in results)
        total_value     = sum(r['value_b'] for r in results)
        perfect_count   = sum(1 for r in results if r['extra_score'] == 4)
        score3_count    = sum(1 for r in results if r['extra_score'] == 3)

        print(f"  Rata-rata RSI          : {avg_rsi:.1f}")
        print(f"  Rata-rata Change %     : {avg_change:+.2f}%")
        print(f"  Change tertinggi       : {max_change:+.2f}%")
        print(f"  Change terendah        : {min_change:+.2f}%")
        print(f"  Total Value (Miliar)   : {total_value:.3f}B IDR")
        print(f"  Score 4/4 (Perfect)    : {perfect_count} saham")
        print(f"  Score 3/4              : {score3_count} saham")

        print(f"\n  Filter Breakdown (dari {len(results)} match):")
        print(f"    Bandar Pass          : {sum(1 for r in results if r['bandar_pass'])}")
        print(f"    Trend Pass           : {sum(1 for r in results if r['trend_pass'])}")
        print(f"    Likuiditas Pass      : {sum(1 for r in results if r['likuiditas_pass'])}")
        print(f"    Above MA20           : {sum(1 for r in results if r['above_ma20'])}")
        print(f"    Above MA50           : {sum(1 for r in results if r['above_ma50'])}")
    else:
        print("  Tidak ada saham yang lolos filter.")

    print("=" * 70)


# ======================================================
# 8. MAIN EXECUTION
# ======================================================
def run_scanner():
    """
    Fungsi utama: Combined Multi-Screener IDX.
    Flow:
      1. Load tickers dari CSV
      2. Load blacklist (opsional)
      3. Scan setiap saham (Original + Bandar + Trend + Likuiditas)
      4. Format dan tampilkan hasil di terminal
      5. Kirim notifikasi ke Telegram
    """
    scan_start      = dt.datetime.now()
    scan_time_str   = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 70)
    print(f"  CLaude SCREENER V{VERSION}")
    print(f"  Filter: BSJP Original + Bandar + Trend + Likuiditas")
    print(f"  {scan_time_str}")
    print("=" * 70)
    logger.info(f"Claude Screener V{VERSION} dimulai.")
    print()

    # 1. Load Tickers
    tickers = load_tickers_from_csv()
    if not tickers:
        logger.error("Tidak ada saham untuk di-scan. Pastikan data/data.csv tersedia.")
        return

    total_scanned = len(tickers)
    logger.info(f"Total saham untuk di-scan: {total_scanned}")

    # 2. Load Blacklist
    blacklist           = load_blacklist()
    blacklisted_count   = 0

    if blacklist:
        original_count      = len(tickers)
        tickers             = [t for t in tickers if t not in blacklist]
        blacklisted_count   = original_count - len(tickers)
        if blacklisted_count > 0:
            logger.info(f"{blacklisted_count} saham dilewati karena blacklist.")

    # 3. Scan
    results = []
    skipped = 0

    print(f"\nMemulai scanning {len(tickers)} saham...\n")
    print("  [Filter aktif: BSJP Original + Bandar + Trend + Likuiditas]\n")

    for i, ticker in enumerate(tickers):
        progress_pct = ((i + 1) / len(tickers)) * 100
        print(f"\r  [{progress_pct:5.1f}%] Scanning {i+1}/{len(tickers)}: {ticker:<6}", end="", flush=True)

        try:
            res = analyze_stock(ticker)
            if res:
                logger.info(
                    f"HIT: {ticker} | Chg: {res['change_pct']:+.2f}% | "
                    f"RSI: {res['rsi']:.1f} | Score: {res['extra_score']}/4 | "
                    f"Value: {res['value_b']:.3f}B | Bandar: {res['bandar_pass']}"
                )
                print(
                    f"\n  ✅ HIT: {ticker} "
                    f"(+{res['change_pct']:.2f}%, RSI:{res['rsi']:.0f}, "
                    f"Score:{res['extra_score']}/4, Bandar:{'✓' if res['bandar_pass'] else '✗'})"
                )
                results.append(res)
        except KeyboardInterrupt:
            print("\n\n⚠️ Scanner dihentikan oleh user.")
            logger.warning("Scanner dihentikan oleh user (KeyboardInterrupt).")
            break
        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            skipped += 1
            continue

    print("\r" + " " * 70 + "\r", end="")

    # 4. Sort: extra_score DESC, change_pct DESC
    results.sort(key=lambda x: (-x['extra_score'], -x['change_pct']))

    # 5. Terminal Output
    print(f"\n{'='*70}")
    print(f"  SCAN SELESAI — Ditemukan: {len(results)} saham")
    print(f"{'='*70}")

    if results:
        table_output = format_terminal_table(results)
        print(f"\n{table_output}")

    print_summary_stats(results, total_scanned, skipped, blacklisted_count)

    # 6. Telegram
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

    # 7. Timing
    elapsed = (dt.datetime.now() - scan_start).total_seconds()
    logger.info(f"Scanner selesai dalam {elapsed:.1f} detik.")
    print(f"\n⏱️  Waktu eksekusi: {elapsed:.1f} detik")
    print()


# ======================================================
# 9. ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_scanner()