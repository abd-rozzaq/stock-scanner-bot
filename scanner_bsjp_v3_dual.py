# ================================================================
# scanner_bsjp_v3_dual.py - DUAL SESSION & LOW NOISE SCANNER
# Desc    : Revisi total berdasarkan evaluasi 7 hari trading.
#           Fokus: Konfirmasi Ulang (Dual-Session), Money Flow,
#           Trend Strength, dan Volatility Filter.
# ================================================================
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import json
import logging
import time
import argparse
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
from typing import Optional, Dict, List
warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING & PATH SETUP
# ======================================================
LOG_DIR = "logs"
CACHE_DIR = "cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"scanner_{dt.datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BSJP_V3")

# ======================================================
# 1. CONFIG & FILTER THRESHOLDS (DIPERKETAT)
# ======================================================
VERSION = "3.0.0-DualSession"
MODE = "auto"  # 'midday', 'afternoon', 'auto'

# --- Filter 1: Price & Momentum (Anti-Overbought) ---
CLOSE_HIGH_RATIO = 0.97          # Close >= 97% High (longgar sedikit agar tidak miss, tapi dikompensasi filter lain)
MIN_PRICE_CHANGE_PCT = 2.0       # Minimal kenaikan 2%
RSI_MIN = 45                     # Hindari saham lemah
RSI_MAX = 68                     # Hindari overbought ekstrem (sumber false signal utama)

# --- Filter 2: Trend & Strength ---
MA20_PERIOD = 20
MA50_PERIOD = 50
MA20_BUFFER = 1.02               # Close harus > MA20 * 1.02 (memastikan breakout valid, bukan sekadar menyentuh)
ADX_PERIOD = 14
ADX_THRESHOLD = 22               # Tren harus cukup kuat
VOL_RATIO_MIN = 1.8              # Volume vs MA5
VOL_RATIO_MAX = 5.0              # Hindari blow-off top (volume >5x biasanya climax)

# --- Filter 3: Liquidity & Volatility (Noise Filter) ---
MIN_VALUE_IDR = 75_000_000       # Turunkan sedikit agar tidak terlalu ketat, tapi dikombinasi ATR
ATR_PERIOD = 14
ATR_MAX_PCT = 0.045              # Volatilitas maksimal 4.5% (hindari saham gergaji/chaos)

# --- Filter 4: Bandarmologi Proxy (Money Flow) ---
CMF_PERIOD = 20
CMF_THRESHOLD = 0.15             # Arus dana positif kuat
OBV_SLOPE_MIN = 0.001            # Kemiringan OBV positif

# --- Risk Management ---
TP_PERCENT = 0.06
CL_DAYS = 7
YFINANCE_PERIOD = "3mo"          # 3 bulan cukup untuk indikator
YFINANCE_INTERVAL = "1d"

# Telegram
TELEGRAM_OK = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
MAX_MESSAGE_LENGTH = 4096

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if REQUESTS_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        TELEGRAM_OK = True
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram dinonaktifkan.")

# ======================================================
# 2. INDICATOR CALCULATIONS (VECTORIZED)
# ======================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df['TR'] = np.maximum(df['High'] - df['Low'],
               np.maximum(abs(df['High'] - df['Close'].shift(1)),
               abs(df['Low'] - df['Close'].shift(1))))
    df['DM_plus'] = np.where((df['High'].diff() > df['Low'].abs().diff()) & (df['High'].diff() > 0), df['High'].diff(), 0)
    df['DM_minus'] = np.where((df['Low'].abs().diff() > df['High'].diff()) & (df['Low'].diff() < 0), -df['Low'].diff(), 0)
    
    atr = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    di_plus = (df['DM_plus'].ewm(alpha=1/period, adjust=False).mean() / atr) * 100
    di_minus = (df['DM_minus'].ewm(alpha=1/period, adjust=False).mean() / atr) * 100
    
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    mf_volume = mf_multiplier * df['Volume']
    cmf = mf_volume.rolling(period).sum() / df['Volume'].rolling(period).sum()
    return cmf

def calculate_obv_slope(series: pd.Series, lookback: int = 5) -> float:
    # Slope sederhana: (OBV[-1] - OBV[-lookback]) / OBV[-lookback]
    if len(series) < lookback + 1: return -999
    return (series.iloc[-1] - series.iloc[-lookback]) / series.iloc[-lookback].abs()

def calculate_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return (tr.ewm(alpha=1/period, adjust=False).mean() / df['Close'])

# ======================================================
# 3. CORE SCANNING LOGIC
# ======================================================
def fetch_and_prepare(ticker: str) -> Optional[pd.DataFrame]:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(symbol, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL, progress=False, auto_adjust=False)
        if df.empty or len(df) < 55: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required): return None
        return df[required].copy().dropna()
    except Exception as e:
        logger.debug(f"Fetch error {ticker}: {e}")
        return None

def analyze_stock(ticker: str, force_check: bool = False) -> Optional[Dict]:
    df = fetch_and_prepare(ticker)
    if df is None: return None

    # Calculate Indicators
    df['MA20'] = df['Close'].rolling(MA20_PERIOD).mean()
    df['MA50'] = df['Close'].rolling(MA50_PERIOD).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['ADX'] = calculate_adx(df)
    df['CMF'] = calculate_cmf(df)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_SLOPE'] = df['OBV'].rolling(5).apply(calculate_obv_slope, raw=False)
    df['ATR_PCT'] = calculate_atr_pct(df)
    df['Value'] = df['Close'] * df['Volume']
    df['Value_MA20'] = df['Value'].rolling(MA20_PERIOD).mean()
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()

    c = df.iloc[-1]
    p = df.iloc[-2]

    # --- FILTERS (CONFLUENCE) ---
    # F1: Momentum & Price Action
    f1 = (
        c['Close'] >= (c['High'] * CLOSE_HIGH_RATIO) and
        ((c['Close'] - p['Close']) / p['Close']) * 100 >= MIN_PRICE_CHANGE_PCT and
        RSI_MIN <= c['RSI'] <= RSI_MAX
    )

    # F2: Trend Strength & Volume Quality
    vol_ratio = c['Volume'] / c['Vol_MA5'] if c['Vol_MA5'] > 0 else 0
    f2 = (
        c['Close'] > (c['MA20'] * MA20_BUFFER) and
        c['MA20'] > c['MA50'] and
        c['ADX'] >= ADX_THRESHOLD and
        VOL_RATIO_MIN <= vol_ratio <= VOL_RATIO_MAX
    )

    # F3: Liquidity & Low Volatility (Noise Filter)
    f3 = (
        c['Value'] >= MIN_VALUE_IDR and
        c['Value'] > c['Value_MA20'] * 0.9 and
        c['ATR_PCT'] <= ATR_MAX_PCT
    )

    # F4: Bandarmology Proxy (Money Flow & Accumulation)
    f4 = (
        c['CMF'] >= CMF_THRESHOLD and
        c['OBV_SLOPE'] >= OBV_SLOPE_MIN
    )

    if not (f1 and f2 and f3 and f4):
        return None

    entry = int(c['Close'])
    return {
        "ticker": ticker,
        "entry": entry,
        "tp": int(entry * (1 + TP_PERCENT)),
        "cl_days": CL_DAYS,
        "rsi": round(c['RSI'], 1),
        "adx": round(c['ADX'], 1),
        "cmf": round(c['CMF'], 2),
        "change_pct": round(((c['Close'] - p['Close']) / p['Close']) * 100, 2),
        "value_b": round(c['Value'] / 1e9, 2),
        "vol_ratio": round(vol_ratio, 1)
    }

# ======================================================
# 4. DUAL-SESSION MANAGER
# ======================================================
def get_session_mode() -> str:
    if MODE != "auto": return MODE
    hour = dt.datetime.now().hour
    if 11 <= hour <= 13: return "midday"
    if 14 <= hour <= 16: return "afternoon"
    return "midday" # fallback

def save_midday_cache(results: List[Dict], date_str: str):
    path = os.path.join(CACHE_DIR, f"midday_{date_str}.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"[CACHE] Disimpan {len(results)} kandidat siang ke {path}")

def load_midday_cache(date_str: str) -> List[str]:
    path = os.path.join(CACHE_DIR, f"midday_{date_str}.json")
    if not os.path.exists(path):
        logger.warning(f"[CACHE] File siang {path} tidak ditemukan.")
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    return [d['ticker'] for d in data]

def run_scanner():
    session = get_session_mode()
    date_str = dt.datetime.now().strftime("%Y-%m-%d")
    logger.info(f"{'='*50}\nSCANNER V{VERSION} | Mode: {session.upper()} | Date: {date_str}\n{'='*50}")
    
    tickers = []
    # Load tickers from CSV or fallback
    csv_path = "data.csv"
    if os.path.exists("data/data.csv"): csv_path = "data/data.csv"
    try:
        df_csv = pd.read_csv(csv_path)
        col = next((c for c in df_csv.columns if 'ticker' in c.lower() or 'kode' in c.lower()), df_csv.columns[0])
        tickers = sorted(list(set([str(t).strip().upper() for t in df_csv[col].dropna() if len(str(t).strip()) >= 4])))
    except Exception:
        logger.error("Gagal load CSV ticker.")
        return

    results = []
    total = len(tickers)
    
    print(f"🔍 Scanning {total} saham...")
    for i, t in enumerate(tickers):
        print(f"\r  [{(i+1)/total*100:5.1f}%] {t}", end="", flush=True)
        try:
            res = analyze_stock(t)
            if res:
                results.append(res)
                print(f"\n✅ MATCH: {t} | Chg: +{res['change_pct']}% | ADX: {res['adx']} | CMF: {res['cmf']}")
        except KeyboardInterrupt:
            break
        except Exception:
            continue
    print()

    if session == "midday":
        if results:
            save_midday_cache(results, date_str)
            logger.info("📥 Sesi Siang Selesai. Kandidat disimpan untuk konfirmasi sore.")
        else:
            logger.info("📥 Sesi Siang Selesai. Tidak ada kandidat.")
            
    elif session == "afternoon":
        candidates = load_midday_cache(date_str)
        if not candidates:
            logger.info("🌇 Tidak ada kandidat dari siang. Sesi sore dilewati.")
            return
            
        logger.info(f"🌇 Validasi ulang {len(candidates)} kandidat siang...")
        confirmed = []
        for t in candidates:
            res = analyze_stock(t, force_check=True)
            if res:
                confirmed.append(res)
                
        logger.info(f"✅ Konfirmasi Sore Selesai: {len(confirmed)} saham lolos double-check.")
        if confirmed:
            # Sort by strength
            confirmed.sort(key=lambda x: -x['adx'])
            # Telegram & Print
            msg = f"<b>🚀 BSJP DUAL-SESSION CONFIRMED</b>\n📅 {date_str}\n🔢 Match: {len(confirmed)}\n━━━━━━━━━━━━━━━━━━━━━━\n"
            for r in confirmed:
                msg += f"<b>{r['ticker']}</b> | Entry: {r['entry']} → TP: {r['tp']}\n   RSI:{r['rsi']} ADX:{r['adx']} CMF:{r['cmf']} | Chg:+{r['change_pct']}%\n\n"
            
            if TELEGRAM_OK:
                send_telegram_message(msg)
                print("📤 Hasil terkonfirmasi dikirim ke Telegram.")
            else:
                print(msg)
        else:
            logger.info("⚠️ Semua kandidat siang gugur saat validasi sore (False Breakout).")

# ======================================================
# 5. TELEGRAM & MAIN
# ======================================================
def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK: return False
    # Split logic (same as before, simplified for brevity)
    chunks = [message[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(message), MAX_MESSAGE_LENGTH)]
    success = True
    for chunk in chunks:
        try:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                              json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}, timeout=10)
            if r.status_code != 200: success = False
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Telegram Error: {e}")
            success = False
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["midday", "afternoon", "auto"], default="auto", help="Override session detection")
    args = parser.parse_args()
    # Update global MODE if passed
    if args.mode != "auto":
        globals()['MODE'] = args.mode
    run_scanner()