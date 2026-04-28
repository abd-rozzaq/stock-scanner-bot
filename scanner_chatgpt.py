# ================================================================
# screener.py - IDX STOCK SCANNER WITH 4 COMBINED FILTER GROUPS
# Version : 3.0.0
# Desc    : Gabungan screener BSJP existing + 3 screener tambahan
#           dari aturan Screening Rules.
#
# FILTER 1 - Existing BSJP V2
#   Main criteria, semua wajib terpenuhi:
#     1. Close >= High * 0.98
#     2. Volume > 1000
#     3. Volume > Volume MA 5
#     4. Price Change % >= 3
#   Additional noise filters, minimal 3 dari 4:
#     5. RSI(14) < 80
#     6. Value >= 1 Miliar
#     7. Close > Price MA 20
#     8. Volume >= 2 x Volume MA 20
#
# FILTER 2 - Bandar Screener
#     1. Bandar Value > 1 x Bandar Value MA 20
#     2. Value MA 20 > 1 Miliar
#     3. Previous Bandar Value <= 1 x Bandar Value
#     4. Bandar Value MA 10 > 1 x Bandar Value MA 20
#
# FILTER 3 - Trend, Volume, Value Screener
#     1. Price > 1 x Price MA 20
#     2. Price > 1 x Price MA 50
#     3. Volume >= 2 x Volume MA 20
#     4. Value > 1 x Value MA 20
#
# FILTER 4 - Basic Liquidity Screener
#     1. Volume > 2 x Volume MA 20
#     2. Value >= 100 Juta
#
# SUMBER DATA HARGA : yfinance
# TICKER            : data/data.csv atau data.csv
# BANDAR VALUE      : data/bandar_value.csv, wajib jika Filter 2 dipakai
# BLACKLIST         : data/blacklist.csv, opsional
# CONFIG TELEGRAM   : config.py, opsional
# ================================================================

import datetime as dt
import html
import logging
import os
import re
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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

logger = logging.getLogger("SCREENER_4_FILTER")

# ======================================================
# 1. CONSTANTS
# ======================================================
VERSION = "3.0.0"

# Filter mode
# True  = semua 4 filter harus lolos.
# False = Filter 2 Bandar dilewati jika file data/bandar_value.csv tidak tersedia.
REQUIRE_BANDAR_FILTER = True

# Existing BSJP thresholds
CLOSE_HIGH_RATIO = 0.98
MIN_FREQUENCY = 1000
MIN_PRICE_CHANGE_PCT = 3
RSI_MAX = 80
MIN_VALUE_IDR = 1_000_000_000
MIN_VALUE_BASIC_IDR = 100_000_000
VOL_SPIKE_MULTIPLIER = 2
MIN_EXTRA_SCORE = 3

# Risk management
TP_PERCENT = 0.08
CL_PERCENT = 0.05

# Indicator periods
RSI_PERIOD = 14
MA5_PERIOD = 5
MA10_PERIOD = 10
MA20_PERIOD = 20
MA50_PERIOD = 50

# Telegram
MAX_MESSAGE_LENGTH = 4096

# Data fetch. 6mo dipakai agar Price MA 50 cukup aman.
YFINANCE_PERIOD = "6mo"
YFINANCE_INTERVAL = "1d"

# Local optional data
BANDAR_VALUE_FILE = os.path.join("data", "bandar_value.csv")

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
        logger.warning("Telegram config ditemukan, tetapi requests tidak tersedia atau token/chat_id kosong.")
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram notifikasi dinonaktifkan.")
except Exception as e:
    logger.warning(f"Error loading config.py: {e}")

# ======================================================
# 3. UTILITY HELPERS
# ======================================================
def clean_ticker(ticker: str) -> str:
    """Normalisasi ticker IDX."""
    return str(ticker).strip().upper().replace(".JK", "")


def parse_number(value) -> float:
    """
    Parsing angka dari format umum Indonesia atau internasional.
    Contoh: 1.000.000.000, 1,000,000,000, Rp1.000.000.000.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if not s:
        return np.nan

    s = re.sub(r"[^0-9,.-]", "", s)
    if not s or s in {"-", ".", ","}:
        return np.nan

    # Jika ada titik dan koma, tentukan separator desimal dari posisi terakhir.
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            # Format Indonesia: 1.234.567,89
            s = s.replace(".", "").replace(",", ".")
        else:
            # Format internasional: 1,234,567.89
            s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = "".join(parts)
    elif "," in s:
        parts = s.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = "".join(parts)
        else:
            s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return np.nan


def safe_divide(numerator: float, denominator: float) -> float:
    """Pembagian aman untuk rasio."""
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)

# ======================================================
# 4. TELEGRAM HELPER
# ======================================================
def send_telegram_message(message: str) -> bool:
    """Mengirim pesan ke Telegram dan otomatis split jika terlalu panjang."""
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
                logger.info(f"Pesan Telegram terkirim ({i + 1}/{len(messages_to_send)}).")
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
    """Memecah pesan Telegram berdasarkan batas karakter."""
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]

    chunks = []
    current_chunk = ""

    for line in message.split("\n"):
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
# 5. DATA HELPERS
# ======================================================
def load_tickers_from_csv() -> List[str]:
    """Membaca daftar ticker dari data/data.csv atau data.csv."""
    file_path = os.path.join("data", "data.csv")

    if not os.path.exists(file_path):
        if os.path.exists("data.csv"):
            file_path = "data.csv"
        else:
            logger.error("File data.csv tidak ditemukan di folder data/ atau root directory.")
            return []

    try:
        df = pd.read_csv(file_path)
        possible_cols = ["Ticker", "ticker", "Kode", "kode", "Code", "code", "Symbol", "symbol"]
        found_col = next((c for c in possible_cols if c in df.columns), df.columns[0])

        tickers = df[found_col].dropna().astype(str).tolist()
        cleaned = sorted(set(clean_ticker(t) for t in tickers if len(str(t).strip()) >= 4))
        logger.info(f"Membaca '{file_path}' kolom '{found_col}'. Total ticker unik: {len(cleaned)}")
        return cleaned
    except pd.errors.EmptyDataError:
        logger.error(f"File '{file_path}' kosong.")
        return []
    except Exception as e:
        logger.error(f"Error membaca '{file_path}': {e}")
        return []


def load_blacklist() -> List[str]:
    """Membaca daftar blacklist saham dari data/blacklist.csv. File ini opsional."""
    file_path = os.path.join("data", "blacklist.csv")

    if not os.path.exists(file_path):
        logger.info("File blacklist.csv tidak ditemukan. Tidak ada blacklist aktif.")
        return []

    try:
        df = pd.read_csv(file_path)
        possible_cols = ["Ticker", "ticker", "Kode", "kode", "Code", "code", "Symbol", "symbol"]
        found_col = next((c for c in possible_cols if c in df.columns), df.columns[0])

        tickers = df[found_col].dropna().astype(str).tolist()
        cleaned = sorted(set(clean_ticker(t) for t in tickers if len(str(t).strip()) >= 4))
        logger.info(f"Blacklist aktif: {len(cleaned)} saham.")
        return cleaned
    except Exception as e:
        logger.warning(f"Error membaca blacklist: {e}")
        return []


def load_bandar_value_data() -> Optional[pd.DataFrame]:
    """
    Membaca data Bandar Value historis.

    Format yang disarankan:
        Date,Ticker,BandarValue
        2026-04-01,BBCA,1234567890
        2026-04-02,BBCA,1500000000

    Minimal perlu 20 data harian per ticker untuk Bandar Value MA 20.
    """
    if not os.path.exists(BANDAR_VALUE_FILE):
        logger.warning(
            f"File {BANDAR_VALUE_FILE} tidak ditemukan. "
            "Filter 2 Bandar tidak dapat dihitung dari yfinance."
        )
        return None

    try:
        df = pd.read_csv(BANDAR_VALUE_FILE)
        if df.empty:
            logger.warning(f"File {BANDAR_VALUE_FILE} kosong.")
            return None

        ticker_cols = ["Ticker", "ticker", "Kode", "kode", "Code", "code", "Symbol", "symbol"]
        date_cols = ["Date", "date", "Tanggal", "tanggal"]
        value_cols = [
            "BandarValue", "Bandar Value", "bandar_value", "bandar value",
            "NetBandarValue", "Net Bandar Value", "net_bandar_value"
        ]

        ticker_col = next((c for c in ticker_cols if c in df.columns), None)
        value_col = next((c for c in value_cols if c in df.columns), None)
        date_col = next((c for c in date_cols if c in df.columns), None)

        if not ticker_col or not value_col:
            logger.warning(
                f"Kolom wajib tidak ditemukan di {BANDAR_VALUE_FILE}. "
                "Butuh kolom Ticker dan BandarValue."
            )
            return None

        normalized = pd.DataFrame()
        normalized["Ticker"] = df[ticker_col].apply(clean_ticker)
        normalized["BandarValue"] = df[value_col].apply(parse_number)

        if date_col:
            normalized["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            normalized["Date"] = pd.NaT

        normalized.dropna(subset=["Ticker", "BandarValue"], inplace=True)
        normalized = normalized[normalized["Ticker"].str.len() >= 4]

        if normalized.empty:
            logger.warning(f"Tidak ada data Bandar Value valid di {BANDAR_VALUE_FILE}.")
            return None

        if normalized["Date"].notna().any():
            normalized.sort_values(["Ticker", "Date"], inplace=True)
        else:
            normalized.sort_values(["Ticker"], inplace=True)

        logger.info(
            f"Bandar Value loaded: {len(normalized)} baris, "
            f"{normalized['Ticker'].nunique()} ticker."
        )
        return normalized
    except Exception as e:
        logger.warning(f"Gagal membaca data Bandar Value: {e}")
        return None


def get_bandar_metrics(
    ticker: str,
    bandar_df: Optional[pd.DataFrame],
    asof_date: Optional[pd.Timestamp]
) -> Optional[Dict]:
    """Menghitung Bandar Value, previous, MA10, dan MA20 untuk satu ticker."""
    if bandar_df is None:
        return None

    ticker = clean_ticker(ticker)
    data = bandar_df[bandar_df["Ticker"] == ticker].copy()
    if data.empty:
        return None

    if asof_date is not None and "Date" in data.columns and data["Date"].notna().any():
        asof_date = pd.to_datetime(asof_date).tz_localize(None) if getattr(asof_date, "tzinfo", None) else pd.to_datetime(asof_date)
        data = data[data["Date"] <= asof_date]

    if len(data) < MA20_PERIOD:
        return None

    if "Date" in data.columns and data["Date"].notna().any():
        data.sort_values("Date", inplace=True)

    bv = data["BandarValue"].astype(float).reset_index(drop=True)
    bv_ma10 = bv.rolling(MA10_PERIOD).mean().iloc[-1]
    bv_ma20 = bv.rolling(MA20_PERIOD).mean().iloc[-1]

    if len(bv) < 2 or pd.isna(bv_ma10) or pd.isna(bv_ma20):
        return None

    return {
        "bandar_value": float(bv.iloc[-1]),
        "prev_bandar_value": float(bv.iloc[-2]),
        "bandar_value_ma10": float(bv_ma10),
        "bandar_value_ma20": float(bv_ma20),
        "bandar_rows": int(len(bv))
    }


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Mengambil data historis saham IDX dari Yahoo Finance."""
    try:
        clean = clean_ticker(ticker)
        symbol = f"{clean}.JK"

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
# 6. TECHNICAL INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """Menghitung RSI dengan Wilder smoothing."""
    try:
        if len(series) < period + 1:
            return -1.0

        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        current_avg_gain = avg_gain.iloc[-1]
        current_avg_loss = avg_loss.iloc[-1]

        if current_avg_loss == 0:
            return 50.0 if current_avg_gain == 0 else 100.0

        rs = current_avg_gain / current_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return round(float(rsi), 1)
    except Exception as e:
        logger.debug(f"Error menghitung RSI: {e}")
        return -1.0

# ======================================================
# 7. FILTER LOGIC
# ======================================================
def analyze_stock(ticker: str, bandar_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """
    Menganalisis satu saham berdasarkan 4 filter gabungan.
    Return dict jika semua filter lolos, None jika tidak.
    """
    df = fetch_stock_data(ticker)

    # Butuh minimal 51 data point untuk MA50 dan previous close.
    if df.empty or len(df) < (MA50_PERIOD + 1):
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]

    if current["Close"] <= 0 or current["High"] <= 0 or current["Volume"] <= 0:
        return None

    prev_close = float(prev["Close"])
    if prev_close <= 0:
        return None

    close = float(current["Close"])
    high = float(current["High"])
    low = float(current["Low"])
    open_price = float(current["Open"])
    volume = float(current["Volume"])

    close_series = df["Close"].astype(float)
    volume_series = df["Volume"].astype(float)
    value_series = close_series * volume_series

    price_ma20 = close_series.rolling(MA20_PERIOD).mean().iloc[-1]
    price_ma50 = close_series.rolling(MA50_PERIOD).mean().iloc[-1]
    volume_ma5 = volume_series.rolling(MA5_PERIOD).mean().iloc[-1]
    volume_ma20 = volume_series.rolling(MA20_PERIOD).mean().iloc[-1]
    value_today = close * volume
    value_ma20 = value_series.rolling(MA20_PERIOD).mean().iloc[-1]
    price_change_pct = ((close - prev_close) / prev_close) * 100
    rsi_value = calculate_rsi(close_series, RSI_PERIOD)

    if any(pd.isna(v) for v in [price_ma20, price_ma50, volume_ma5, volume_ma20, value_ma20]):
        return None

    if volume_ma20 <= 0 or value_ma20 <= 0:
        return None

    # ==================================================
    # FILTER 1 - Existing BSJP V2
    # ==================================================
    f1_main_1 = close >= (high * CLOSE_HIGH_RATIO)
    f1_main_2 = volume > MIN_FREQUENCY
    f1_main_3 = volume > volume_ma5
    f1_main_4 = price_change_pct >= MIN_PRICE_CHANGE_PCT

    f1_extra_1 = False if rsi_value < 0 else rsi_value < RSI_MAX
    f1_extra_2 = value_today >= MIN_VALUE_IDR
    f1_extra_3 = close > price_ma20
    f1_extra_4 = volume >= (VOL_SPIKE_MULTIPLIER * volume_ma20)
    f1_extra_score = sum([f1_extra_1, f1_extra_2, f1_extra_3, f1_extra_4])

    filter_1_bsjp = all([f1_main_1, f1_main_2, f1_main_3, f1_main_4]) and f1_extra_score >= MIN_EXTRA_SCORE

    # ==================================================
    # FILTER 2 - Bandar Screener
    # ==================================================
    asof_date = df.index[-1]
    bandar_metrics = get_bandar_metrics(ticker, bandar_df, asof_date)

    if bandar_metrics is None:
        filter_2_bandar = False
        bandar_value = 0.0
        prev_bandar_value = 0.0
        bandar_value_ma10 = 0.0
        bandar_value_ma20 = 0.0
    else:
        bandar_value = bandar_metrics["bandar_value"]
        prev_bandar_value = bandar_metrics["prev_bandar_value"]
        bandar_value_ma10 = bandar_metrics["bandar_value_ma10"]
        bandar_value_ma20 = bandar_metrics["bandar_value_ma20"]

        f2_1 = bandar_value > (1 * bandar_value_ma20)
        f2_2 = value_ma20 > MIN_VALUE_IDR
        f2_3 = prev_bandar_value <= (1 * bandar_value)
        f2_4 = bandar_value_ma10 > (1 * bandar_value_ma20)
        filter_2_bandar = all([f2_1, f2_2, f2_3, f2_4])

    if not REQUIRE_BANDAR_FILTER and bandar_metrics is None:
        filter_2_bandar = True

    # ==================================================
    # FILTER 3 - Trend, Volume, Value Screener
    # ==================================================
    f3_1 = close > (1 * price_ma20)
    f3_2 = close > (1 * price_ma50)
    f3_3 = volume >= (VOL_SPIKE_MULTIPLIER * volume_ma20)
    f3_4 = value_today > (1 * value_ma20)
    filter_3_trend_volume_value = all([f3_1, f3_2, f3_3, f3_4])

    # ==================================================
    # FILTER 4 - Basic Liquidity Screener
    # ==================================================
    f4_1 = volume > (VOL_SPIKE_MULTIPLIER * volume_ma20)
    f4_2 = value_today >= MIN_VALUE_BASIC_IDR
    filter_4_basic_liquidity = all([f4_1, f4_2])

    # ==================================================
    # FINAL GATE - Semua 4 filter harus lolos
    # ==================================================
    all_filter_flags = [
        filter_1_bsjp,
        filter_2_bandar,
        filter_3_trend_volume_value,
        filter_4_basic_liquidity
    ]

    if not all(all_filter_flags):
        return None

    entry_price = close
    tp_price = entry_price * (1 + TP_PERCENT)
    cl_price = entry_price * (1 - CL_PERCENT)

    volume_ratio_20 = safe_divide(volume, volume_ma20)
    value_ratio_20 = safe_divide(value_today, value_ma20)
    bandar_ratio_20 = safe_divide(bandar_value, bandar_value_ma20)

    return {
        "ticker": clean_ticker(ticker),
        "date": str(pd.to_datetime(df.index[-1]).date()),
        "open": int(open_price),
        "high": int(high),
        "low": int(low),
        "close": int(close),
        "change_pct": round(float(price_change_pct), 2),
        "volume": int(volume),
        "volume_ma5": int(volume_ma5),
        "volume_ma20": int(volume_ma20),
        "volume_ratio_20": round(volume_ratio_20, 2),
        "value_idr": float(value_today),
        "value_b": round(float(value_today) / 1e9, 2),
        "value_ma20": float(value_ma20),
        "value_ma20_b": round(float(value_ma20) / 1e9, 2),
        "value_ratio_20": round(value_ratio_20, 2),
        "price_ma20": round(float(price_ma20), 2),
        "price_ma50": round(float(price_ma50), 2),
        "rsi": rsi_value if rsi_value >= 0 else 0.0,
        "bandar_value": float(bandar_value),
        "prev_bandar_value": float(prev_bandar_value),
        "bandar_value_ma10": float(bandar_value_ma10),
        "bandar_value_ma20": float(bandar_value_ma20),
        "bandar_ratio_20": round(bandar_ratio_20, 2),
        "filter_1_bsjp": filter_1_bsjp,
        "filter_2_bandar": filter_2_bandar,
        "filter_3_trend_volume_value": filter_3_trend_volume_value,
        "filter_4_basic_liquidity": filter_4_basic_liquidity,
        "filter_score": sum(all_filter_flags),
        "bsjp_extra_score": f1_extra_score,
        "entry": int(entry_price),
        "tp": int(tp_price),
        "cl": int(cl_price),
        "status": "MATCH_4_FILTERS"
    }

# ======================================================
# 8. OUTPUT FORMATTERS
# ======================================================
def format_terminal_table(results: List[Dict]) -> str:
    """Membuat tabel terminal."""
    if not results:
        return "Tidak ada hasil."

    header = (
        f"{'No':>3} | {'Ticker':<6} | {'Close':>7} | {'Chg%':>7} | "
        f"{'RSI':>5} | {'Val(B)':>7} | {'V/MA20':>7} | {'Vol/MA20':>8} | "
        f"{'BV/MA20':>8} | {'F1':>2} {'F2':>2} {'F3':>2} {'F4':>2} | "
        f"{'TP':>7} | {'CL':>7}"
    )
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for i, r in enumerate(results, 1):
        line = (
            f"{i:>3} | {r['ticker']:<6} | {r['close']:>7,} | "
            f"{r['change_pct']:>+6.2f}% | {r['rsi']:>5.1f} | "
            f"{r['value_b']:>6.2f}B | {r['value_ratio_20']:>7.2f} | "
            f"{r['volume_ratio_20']:>8.2f} | {r['bandar_ratio_20']:>8.2f} | "
            f"{'Y' if r['filter_1_bsjp'] else 'N':>2} "
            f"{'Y' if r['filter_2_bandar'] else 'N':>2} "
            f"{'Y' if r['filter_3_trend_volume_value'] else 'N':>2} "
            f"{'Y' if r['filter_4_basic_liquidity'] else 'N':>2} | "
            f"{r['tp']:>7,} | {r['cl']:>7,}"
        )
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def format_telegram_message(
    results: List[Dict],
    scan_time: str,
    total_scanned: int,
    total_skipped: int,
    blacklisted_count: int,
    bandar_data_available: bool
) -> str:
    """Membuat pesan Telegram dengan format HTML."""
    msg_lines = [
        f"<b>ChatGPT Screener V{VERSION}</b>",
        f"<i>{html.escape(scan_time)}</i>",
        "",
        f"Scanned: {total_scanned} | Blacklisted: {blacklisted_count} | Skipped: {total_skipped}",
        f"Bandar Data: {'Available' if bandar_data_available else 'Not Available'}",
        f"<b>Match: {len(results)} saham</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not results:
        msg_lines.append("")
        msg_lines.append("<i>Tidak ada saham yang lolos 4 filter gabungan.</i>")
        return "\n".join(msg_lines)

    for r in results:
        ticker = html.escape(r["ticker"])
        msg_lines.extend([
            f"<b>{ticker}</b> | {r['change_pct']:+.2f}% | RSI {r['rsi']:.1f}",
            f"   Close: {r['close']:,} | Val: {r['value_b']:.2f}B | Vol/MA20: {r['volume_ratio_20']:.2f}x",
            f"   Price MA20: {r['price_ma20']:,.0f} | MA50: {r['price_ma50']:,.0f}",
            f"   Bandar/MA20: {r['bandar_ratio_20']:.2f}x | BSJP Score: {r['bsjp_extra_score']}/4",
            f"   F1 BSJP ✅ | F2 Bandar ✅ | F3 Trend ✅ | F4 Liquidity ✅",
            f"   Entry: {r['entry']:,} → TP: {r['tp']:,} | CL: {r['cl']:,}",
            ""
        ])

    avg_rsi = sum(r["rsi"] for r in results) / len(results)
    avg_change = sum(r["change_pct"] for r in results) / len(results)
    total_value = sum(r["value_b"] for r in results)

    msg_lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━━",
        "<b>SUMMARY</b>",
        f"Avg RSI: {avg_rsi:.1f} | Avg Chg: {avg_change:+.2f}%",
        f"Total Value: {total_value:.2f}B IDR",
        f"<i>TP: +{TP_PERCENT * 100:.0f}% | CL: -{CL_PERCENT * 100:.0f}%</i>"
    ])

    return "\n".join(msg_lines)


def print_summary_stats(
    results: List[Dict],
    total_scanned: int,
    total_skipped: int,
    blacklisted_count: int,
    bandar_data_available: bool
) -> None:
    """Menampilkan ringkasan statistik di terminal."""
    print("\n" + "=" * 70)
    print("RINGKASAN STATISTIK")
    print("=" * 70)
    print(f"  Total saham di-scan       : {total_scanned}")
    print(f"  Blacklisted               : {blacklisted_count}")
    print(f"  Gagal fetch / skip         : {total_skipped}")
    print(f"  Data Bandar Value          : {'Tersedia' if bandar_data_available else 'Tidak tersedia'}")
    print(f"  Total MATCH 4 filter       : {len(results)}")
    print("-" * 45)

    if results:
        avg_rsi = sum(r["rsi"] for r in results) / len(results)
        avg_change = sum(r["change_pct"] for r in results) / len(results)
        avg_volume_ratio = sum(r["volume_ratio_20"] for r in results) / len(results)
        avg_value_ratio = sum(r["value_ratio_20"] for r in results) / len(results)
        total_value = sum(r["value_b"] for r in results)

        print(f"  Rata-rata RSI             : {avg_rsi:.1f}")
        print(f"  Rata-rata Change %        : {avg_change:+.2f}%")
        print(f"  Rata-rata Vol/MA20        : {avg_volume_ratio:.2f}x")
        print(f"  Rata-rata Value/MA20      : {avg_value_ratio:.2f}x")
        print(f"  Total Value               : {total_value:.2f}B IDR")
    else:
        print("  Tidak ada saham yang lolos seluruh filter.")

    print("=" * 70)

# ======================================================
# 9. MAIN EXECUTION
# ======================================================
def run_scanner() -> None:
    """Menjalankan scanner 4 filter gabungan."""
    scan_start = dt.datetime.now()
    scan_time_str = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 70)
    print(f"  ChatGPT SCREENER V{VERSION}")
    print(f"  {scan_time_str}")
    print("=" * 70)
    print()
    logger.info(f"Scanner V{VERSION} dimulai.")

    tickers = load_tickers_from_csv()
    if not tickers:
        logger.error("Tidak ada saham untuk di-scan. Pastikan data/data.csv tersedia.")
        return

    total_scanned = len(tickers)

    blacklist = load_blacklist()
    blacklisted_count = 0

    if blacklist:
        original_count = len(tickers)
        tickers = [t for t in tickers if t not in blacklist]
        blacklisted_count = original_count - len(tickers)
        if blacklisted_count > 0:
            logger.info(f"{blacklisted_count} saham dilewati karena blacklist.")

    bandar_df = load_bandar_value_data()
    bandar_data_available = bandar_df is not None

    if REQUIRE_BANDAR_FILTER and not bandar_data_available:
        print("⚠️  data/bandar_value.csv tidak tersedia.")
        print("   Filter 2 membutuhkan data Bandar Value historis minimal 20 hari per ticker.")
        print("   Scanner tetap berjalan, tetapi saham tidak akan lolos Filter 2 tanpa data tersebut.\n")

    results = []
    skipped = 0

    print(f"Memulai scanning {len(tickers)} saham...\n")

    for i, ticker in enumerate(tickers):
        progress_pct = ((i + 1) / len(tickers)) * 100
        print(f"\r  [{progress_pct:5.1f}%] Scanning {i + 1}/{len(tickers)}: {ticker:<6}", end="", flush=True)

        try:
            res = analyze_stock(ticker, bandar_df)
            if res:
                logger.info(
                    f"HIT: {ticker} | Chg: {res['change_pct']:+.2f}% | "
                    f"Val: {res['value_b']:.2f}B | Vol/MA20: {res['volume_ratio_20']:.2f}x | "
                    f"BV/MA20: {res['bandar_ratio_20']:.2f}x"
                )
                print(
                    f"\n  ✅ HIT: {ticker} "
                    f"({res['change_pct']:+.2f}%, Val:{res['value_b']:.2f}B, Vol/MA20:{res['volume_ratio_20']:.2f}x)"
                )
                results.append(res)
        except KeyboardInterrupt:
            print("\n\n⚠️ Scanner dihentikan oleh user.")
            logger.warning("Scanner dihentikan oleh user.")
            break
        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            skipped += 1
            continue

    print("\r" + " " * 80 + "\r", end="")

    # Sort: change tertinggi, volume ratio tertinggi, value tertinggi.
    results.sort(key=lambda x: (-x["change_pct"], -x["volume_ratio_20"], -x["value_b"]))

    print(f"\n{'=' * 70}")
    print(f"  SCAN SELESAI — Ditemukan: {len(results)} saham")
    print(f"{'=' * 70}")

    if results:
        print(f"\n{format_terminal_table(results)}")

    print_summary_stats(
        results=results,
        total_scanned=total_scanned,
        total_skipped=skipped,
        blacklisted_count=blacklisted_count,
        bandar_data_available=bandar_data_available
    )

    if TELEGRAM_OK:
        logger.info("Mengirim hasil ke Telegram...")
        telegram_msg = format_telegram_message(
            results=results,
            scan_time=scan_time_str,
            total_scanned=total_scanned,
            total_skipped=skipped,
            blacklisted_count=blacklisted_count,
            bandar_data_available=bandar_data_available
        )
        success = send_telegram_message(telegram_msg)
        print("\n✅ Hasil terkirim ke Telegram." if success else "\n❌ Gagal mengirim ke Telegram.")
    else:
        print("\nℹ️  Telegram tidak dikonfigurasi. Hasil hanya ditampilkan di terminal.")

    elapsed = (dt.datetime.now() - scan_start).total_seconds()
    logger.info(f"Scanner selesai dalam {elapsed:.1f} detik.")
    print(f"\n⏱️  Waktu eksekusi: {elapsed:.1f} detik")
    print()

# ======================================================
# 10. ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_scanner()
