import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import logging
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING & PATH
# ======================================================
LOG_DIR = "logs"
CANDIDATES_DIR = "candidates"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CANDIDATES_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"scanner_v2.2.1_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger("GROK_SCANNER_V2.2.1")

# ======================================================
# 1. CONSTANTS - V2.2.1 (ANTI-NOISE + DUAL SESSION)
# ======================================================
VERSION = "2.2.1"

CLOSE_HIGH_RATIO = 0.99
MIN_PRICE_CHANGE_PCT = 4.0
MIN_VALUE_IDR = 5_000_000_000
VOL_SPIKE_MULTIPLIER = 3.0
MIN_CLOSE_PRICE = 150
MIN_GREEN_BODY_PCT = 2.0

RSI_PERIOD = 14
MA20_PERIOD = 20
MA50_PERIOD = 50
RSI_MIN = 55
RSI_MAX = 75

MIN_EXTRA_SCORE = 4

TP_PERCENT = 0.06
CL_PERCENT = 0.05

MAX_MESSAGE_LENGTH = 4096

# Telegram
TELEGRAM_OK = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        TELEGRAM_OK = True
        logger.info("✅ Telegram ready")
except:
    logger.warning("⚠️ config.py tidak ditemukan atau token kosong")

YFINANCE_PERIOD = "3mo"
YFINANCE_INTERVAL = "1d"

# ======================================================
# JSON SANITIZER (FIX BUG UTAMA)
# ======================================================
def make_json_serializable(obj):
    """Convert numpy types & bool ke native Python"""
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    return obj

# ======================================================
# TELEGRAM FULL (di-copy dari kode asli kamu)
# ======================================================
def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        logger.info("Telegram tidak aktif.")
        return False

    messages_to_send = split_telegram_message(message)
    all_success = True
    for i, msg_chunk in enumerate(messages_to_send):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg_chunk, "parse_mode": "HTML"}
            response = requests.post(url, data=data, timeout=15)
            if response.status_code == 200:
                logger.info(f"✅ Telegram terkirim ({i+1}/{len(messages_to_send)})")
            else:
                logger.error(f"❌ Telegram gagal: {response.status_code}")
                all_success = False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            all_success = False
        if len(messages_to_send) > 1:
            time.sleep(1)
    return all_success

def split_telegram_message(message: str) -> List[str]:
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks = []
    current = ""
    for line in message.split("\n"):
        test = current + line + "\n"
        if len(test) > MAX_MESSAGE_LENGTH:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current = test
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [message[:MAX_MESSAGE_LENGTH]]

# ======================================================
# DATA HELPERS (sama)
# ======================================================
def load_tickers_from_csv() -> List[str]:
    file_path = os.path.join("data", "data.csv")
    if not os.path.exists(file_path) and os.path.exists("data.csv"):
        file_path = "data.csv"
    try:
        df = pd.read_csv(file_path)
        col = next((c for c in ['Ticker','ticker','Kode','kode','Code','code','Symbol','symbol'] if c in df.columns), df.columns[0])
        tickers = df[col].dropna().astype(str).str.strip().str.upper().tolist()
        cleaned = list(set(t for t in tickers if len(t) >= 4))
        logger.info(f"Loaded {len(cleaned)} tickers")
        return sorted(cleaned)
    except Exception as e:
        logger.error(f"Error load tickers: {e}")
        return []

def get_candidates_file(date_str: str, session: str) -> Path:
    return Path(CANDIDATES_DIR) / f"candidates_{date_str}_{session}.json"

# ======================================================
# FETCH + INDICATORS (sama)
# ======================================================
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(symbol, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL,
                         progress=False, auto_adjust=False)
        if df.empty or len(df) < MA50_PERIOD + 1:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.debug(f"Fetch error {ticker}: {e}")
        return pd.DataFrame()

def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    try:
        if len(series) < period + 1:
            return -1.0
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        current_avg_gain = avg_gain.iloc[-1]
        current_avg_loss = avg_loss.iloc[-1]
        if current_avg_loss == 0:
            return 100.0 if current_avg_gain > 0 else 50.0
        rs = current_avg_gain / current_avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 1)
    except:
        return -1.0

# ======================================================
# ANALYZE STOCK (dengan explicit bool())
# ======================================================
def analyze_stock(ticker: str) -> Optional[Dict]:
    df = fetch_stock_data(ticker)
    if df.empty:
        return None

    current = df.iloc[-1]
    if current['Close'] < MIN_CLOSE_PRICE or current['Volume'] <= 0:
        return None

    prev_close = df['Close'].iloc[-2]
    price_change_pct = ((current['Close'] - prev_close) / prev_close) * 100

    cond_close_high = bool(current['Close'] >= (current['High'] * CLOSE_HIGH_RATIO))
    cond_change = bool(price_change_pct >= MIN_PRICE_CHANGE_PCT)
    cond_green = bool(((current['Close'] - current['Open']) / current['Open']) * 100 >= MIN_GREEN_BODY_PCT)

    ma20 = df['Close'].rolling(MA20_PERIOD).mean().iloc[-1]
    ma50 = df['Close'].rolling(MA50_PERIOD).mean().iloc[-1]
    cond_ma20 = bool(not pd.isna(ma20) and current['Close'] > ma20)
    cond_ma50 = bool(not pd.isna(ma50) and current['Close'] > ma50)
    cond_ma_trend = bool(cond_ma20 and cond_ma50 and ma20 > ma50)

    current_vol = float(current['Volume'])
    value_today = current['Close'] * current_vol
    vol_ma20 = df['Volume'].rolling(MA20_PERIOD).mean().iloc[-1]
    cond_vol_spike = bool(not pd.isna(vol_ma20) and vol_ma20 > 0 and current_vol >= (VOL_SPIKE_MULTIPLIER * vol_ma20))
    cond_value = bool(value_today >= MIN_VALUE_IDR)

    value_series = df['Close'] * df['Volume']
    value_ma20 = value_series.rolling(MA20_PERIOD).mean().iloc[-1]
    cond_value_strength = bool(not pd.isna(value_ma20) and value_today > value_ma20)

    rsi_value = calculate_rsi(df['Close'])
    cond_rsi = bool(RSI_MIN <= rsi_value <= RSI_MAX)

    if not all([cond_close_high, cond_change, cond_green, cond_ma_trend, cond_vol_spike, cond_value, cond_value_strength, cond_rsi]):
        return None

    extra_score = sum([cond_rsi, cond_value_strength, cond_ma_trend, cond_vol_spike])
    if extra_score < MIN_EXTRA_SCORE:
        return None

    entry = float(current['Close'])
    tp = entry * (1 + TP_PERCENT)
    cl = entry * (1 - CL_PERCENT)

    return {
        "ticker": ticker,
        "close": int(current['Close']),
        "change_pct": round(float(price_change_pct), 2),
        "rsi": float(rsi_value),
        "value_b": round(float(value_today) / 1e9, 2),
        "ma20": round(float(ma20), 0),
        "ma50": round(float(ma50), 0),
        "vol_spike": cond_vol_spike,
        "extra_score": int(extra_score),
        "entry": int(entry),
        "tp": int(tp),
        "cl": int(cl),
        "status": "MATCH"
    }

# ======================================================
# SAVE CANDIDATES (pakai sanitizer)
# ======================================================
def save_candidates(date_str: str, session: str, results: List[Dict]):
    file = get_candidates_file(date_str, session)
    data = {
        "tickers": [r["ticker"] for r in results],
        "full": [make_json_serializable(r) for r in results]
    }
    file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"💾 Saved {len(results)} candidates for {session} on {date_str}")

# ======================================================
# RUN SCANNER
# ======================================================
def run_scanner(session: str = "eod"):
    scan_start = dt.datetime.now()
    date_str = scan_start.strftime("%Y-%m-%d")
    scan_time_str = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"🚀 GROK SCANNER V{VERSION} — SESSION: {session.upper()} | {scan_time_str}")

    tickers = load_tickers_from_csv()
    results = []
    for i, ticker in enumerate(tickers):
        print(f"\r[{((i+1)/len(tickers)*100):5.1f}%] Scanning {ticker:<6}", end="", flush=True)
        res = analyze_stock(ticker)
        if res:
            results.append(res)
            logger.info(f"HIT {ticker} | +{res['change_pct']:.1f}% | RSI:{res['rsi']:.1f} | Val:{res['value_b']}B")

    print("\r" + " " * 80 + "\r", end="")

    # Dual-session logic
    confirmed = results
    if session == "sore":
        siang_file = get_candidates_file(date_str, "siang")
        if siang_file.exists():
            siang_data = json.loads(siang_file.read_text(encoding="utf-8"))
            siang_tickers = set(siang_data["tickers"])
            confirmed = [r for r in results if r["ticker"] in siang_tickers]
            logger.info(f"✅ CONFIRMED siang + sore: {len(confirmed)} saham")

    if session in ["siang", "sore"]:
        save_candidates(date_str, session, results)

    confirmed.sort(key=lambda x: (-x['extra_score'], -x['change_pct']))

    print(f"\n{'='*80}")
    print(f"GROK SCANNER V{VERSION} — {session.upper()} | {len(confirmed)} CONFIRMED")
    print("="*80)

    if confirmed:
        for r in confirmed:
            print(f"✅ {r['ticker']:6} | +{r['change_pct']:5.1f}% | RSI:{r['rsi']:5.1f} | Val:{r['value_b']:6.2f}B | Entry:{r['entry']:,} → TP:{r['tp']:,}")
    else:
        print("❌ Tidak ada sinyal CONFIRMED hari ini.")

    # Telegram hanya kirim CONFIRMED
    if TELEGRAM_OK and confirmed:
        msg = f"<b>GROK SCREENER V{VERSION} — {session.upper()} CONFIRMED</b>\n{scan_time_str}\n\n"
        for r in confirmed[:15]:
            msg += f"<b>{r['ticker']}</b> +{r['change_pct']:.1f}% RSI:{r['rsi']:.0f} Val:{r['value_b']:.2f}B\n"
        send_telegram_message(msg)

    elapsed = (dt.datetime.now() - scan_start).total_seconds()
    print(f"⏱️  Selesai dalam {elapsed:.1f} detik")

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    import requests  # dipindah ke sini agar tidak error kalau import gagal
    parser = argparse.ArgumentParser(description="GROK SCREENER V2.2.1")
    parser.add_argument("--session", choices=["siang", "sore", "eod"], default="eod")
    args = parser.parse_args()
    run_scanner(session=args.session)