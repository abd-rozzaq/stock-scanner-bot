# ================================================================
# scanner_dual_session.py - DUAL-SESSION BSJP & TREND SCANNER
# Desc    : Revisi Screener saham IDX untuk meminimalkan false signal.
#           Fitur baru:
#           - Mode Dual-Session (Siang & Sore) via JSON cache
#           - Filter trend lebih ketat (MA20 > MA50 mutlak)
#           - Syarat likuiditas dinaikkan (Minimal Value 5 Miliar)
# ================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import logging
import time
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING & CACHE SETUP
# ======================================================
LOG_DIR = "logs"
DATA_DIR = "data"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"scanner_{dt.datetime.now().strftime('%Y%m%d')}.log")
CACHE_FILE = os.path.join(DATA_DIR, "session1_cache.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BSJP_DUAL")

# ======================================================
# 1. CONSTANTS & STRICT FILTER THRESHOLDS
# ======================================================
VERSION = "4.0.0-DualSession"

# Filter 1: BSJP Base
CLOSE_HIGH_RATIO = 0.96          # Sedikit dilonggarkan dari 0.98 untuk menoleransi volatilitas wajar
MIN_FREQUENCY = 2000             # Dinaikkan agar terhindar dari saham illikuid
MIN_PRICE_CHANGE_PCT = 2.0       # Minimum naik 2% 
RSI_MAX = 75                     # Diturunkan dari 80 agar tidak overbought pucuk

# Filter 2: Trend & Volume
MA20_PERIOD = 20
MA50_PERIOD = 50
VOL_SPIKE_MULTIPLIER = 1.5       # Spike volume minimal 1.5x rata-rata 20 hari

# Filter 3: Liquidity (Sangat Diperketat)
MIN_VALUE_IDR = 3_000_000_000         # Transaksi hari ini min 3 Miliar
MIN_VALUE_MA20_IDR = 5_000_000_000    # Rata-rata transaksi 20 hari min 5 Miliar

# Filter 4: Bandarmologi (Proxy)
BANDAR_MA10_PERIOD = 10
BANDAR_MA20_PERIOD = 20

# Risk Management User
TP_PERCENT = 0.06                # Target Profit 6% sesuai rules user
CL_PERCENT = 0.05                # Cut loss tolerance

# Data fetch
YFINANCE_PERIOD = "6mo"  
YFINANCE_INTERVAL = "1d"
MAX_MESSAGE_LENGTH = 4096        

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
except ImportError:
    pass

def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=data, timeout=15)
        return True
    except Exception as e:
        logger.error(f"Error Telegram: {e}")
        return False

# ======================================================
# 3. CORE LOGIC (STRICT FILTERS)
# ======================================================
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(symbol, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL, progress=False, auto_adjust=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols): return pd.DataFrame()
        return df[required_cols].copy().dropna()
    except Exception:
        return pd.DataFrame()

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else 100
        return round(100.0 - (100.0 / (1.0 + rs)), 1)
    except Exception:
        return -1.0

def analyze_stock(ticker: str) -> Optional[Dict]:
    df = fetch_stock_data(ticker)
    if df.empty or len(df) < (MA50_PERIOD + 1):
        return None

    df['Value'] = df['Close'] * df['Volume']
    df['Value_MA20'] = df['Value'].rolling(MA20_PERIOD).mean()
    df['Vol_MA20'] = df['Volume'].rolling(MA20_PERIOD).mean()
    df['MA20'] = df['Close'].rolling(MA20_PERIOD).mean()
    df['MA50'] = df['Close'].rolling(MA50_PERIOD).mean()
    
    df['Bandar_Value'] = np.where(df['Close'] > df['Open'], df['Value'] * 0.5, df['Value'] * -0.5)
    df['Bandar_MA10'] = df['Bandar_Value'].rolling(BANDAR_MA10_PERIOD).mean()
    df['Bandar_MA20'] = df['Bandar_Value'].rolling(BANDAR_MA20_PERIOD).mean()

    current = df.iloc[-1]
    prev = df.iloc[-2]

    if current['Close'] < 50 or current['Volume'] <= 0: return None # Hindari saham gocap mutlak

    price_change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
    rsi_value = calculate_rsi(df['Close'])

    # F1: Price Action & Momentum
    pass_f1 = all([
        current['Close'] >= (current['High'] * CLOSE_HIGH_RATIO),
        current['Volume'] > MIN_FREQUENCY,
        price_change_pct >= MIN_PRICE_CHANGE_PCT,
        rsi_value < RSI_MAX
    ])

    # F2: Trend Strict (Mutlak Uptrend)
    pass_f2 = all([
        current['Close'] > current['MA20'],
        current['MA20'] > current['MA50'], # SYARAT BARU: MA20 harus di atas MA50 (Uptrend Valid)
        current['Volume'] >= (VOL_SPIKE_MULTIPLIER * current['Vol_MA20'])
    ])

    # F3: Liquidity Strict
    pass_f3 = all([
        current['Value'] >= MIN_VALUE_IDR,
        current['Value_MA20'] >= MIN_VALUE_MA20_IDR
    ])

    # F4: Bandarmologi Proxy
    pass_f4 = all([
        current['Bandar_Value'] > current['Bandar_MA20'],
        current['Bandar_MA10'] > current['Bandar_MA20']
    ])

    if not (pass_f1 and pass_f2 and pass_f3 and pass_f4):
        return None

    entry_price = int(current['Close'])
    return {
        "ticker": ticker,
        "close": entry_price,
        "change_pct": round(float(price_change_pct), 2),
        "value_b": round(float(current['Value']) / 1e9, 2),
        "rsi": rsi_value,
        "entry": entry_price,
        "tp": int(entry_price * (1 + TP_PERCENT)),
        "cl": int(entry_price * (1 - CL_PERCENT))
    }

# ======================================================
# 4. DUAL SESSION LOGIC
# ======================================================
def load_tickers_from_csv() -> List[str]:
    file_path = os.path.join(DATA_DIR, "data.csv") if os.path.exists(os.path.join(DATA_DIR, "data.csv")) else "data.csv"
    try:
        df = pd.read_csv(file_path)
        col = next((c for c in ['Ticker','ticker','Kode','kode'] if c in df.columns), df.columns[0])
        return sorted(list(set([str(t).strip().upper() for t in df[col].dropna() if len(str(t).strip()) >= 4])))
    except Exception: 
        logger.error("File data.csv tidak ditemukan atau format salah.")
        return []

def save_cache(data: List[str]):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)

def load_cache() -> List[str]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return []

def run_scanner():
    print(f"\n{'='*60}\n  DUAL-SESSION SCANNER V{VERSION}\n{'='*60}")
    print("Pilih Sesi Scanning:")
    print("  [1] Sesi Siang (Break Sesi 1 - ±12:00 WIB)")
    print("  [2] Sesi Sore (Jelang Tutup - ±15:45 WIB)")
    sesi = input("Masukkan pilihan (1/2): ").strip()

    if sesi not in ["1", "2"]:
        print("Pilihan tidak valid. Membatalkan scan.")
        return

    tickers = load_tickers_from_csv()
    if not tickers: return

    results = []
    print(f"\nMemulai Scanning {len(tickers)} saham dengan filter ketat...\n")
    
    for i, ticker in enumerate(tickers):
        print(f"\r  [{(i+1)/len(tickers)*100:5.1f}%] Cek: {ticker:<6}", end="", flush=True)
        try:
            res = analyze_stock(ticker)
            if res:
                results.append(res)
        except KeyboardInterrupt:
            break
        except Exception:
            continue

    results.sort(key=lambda x: -x['change_pct'])
    current_tickers = [r['ticker'] for r in results]

    print("\n\n" + "="*60)
    
    if sesi == "1":
        # Simpan Cache Sesi 1
        save_cache(current_tickers)
        print(f"  HASIL SESI 1 — Ditemukan {len(results)} kandidat.")
        print("  Data telah disimpan. Silakan jalankan Mode 2 nanti sore.")
        print("="*60)
        
        telegram_msg = f"<b>Gemini Screener Kandidat Pantau - Sesi Siang:</b>\n" + ", ".join(current_tickers) if current_tickers else "<i>Sesi Siang: Tidak ada kandidat lolos filter.</i>"
        
        for r in results:
            print(f"[{r['ticker']}] Chg: +{r['change_pct']}% | Val: {r['value_b']}B | RSI: {r['rsi']}")

    elif sesi == "2":
        # Bandingkan dengan Cache Sesi 1
        sesi1_tickers = load_cache()
        if not sesi1_tickers:
            print("  ⚠️ Peringatan: Tidak ada data Cache Sesi 1. Scan ini akan memunculkan hasil independen Sesi 2.")
            final_results = results
        else:
            # INTERSECTION (Hanya yang lolos Sesi 1 & Sesi 2)
            final_results = [r for r in results if r['ticker'] in sesi1_tickers]
            print(f"  HASIL SESI 2 — {len(results)} Lolos Filter.")
            print(f"  HASIL FINAL (Lolos Dual-Session) — Ditemukan {len(final_results)} saham mantap.")
        
        print("="*60)
        telegram_msg = f"<b>Gemini Screener - HASIL FINAL DUAL-SESSION</b>\n\n"
        if final_results:
            for r in final_results:
                print(f"⭐ MATCH FINAL: [{r['ticker']}] Entry: {r['entry']} | TP: {r['tp']} (+6%) | CL: {r['cl']} (-5%)")
                telegram_msg += f"<b>{r['ticker']}</b> | {r['change_pct']:+.2f}%\nEntry: {r['entry']} | TP: {r['tp']} | CL: {r['cl']}\n\n"
        else:
            print("Tidak ada saham yang berhasil bertahan dari Sesi 1 hingga Sesi 2 hari ini.")
            telegram_msg += "<i>Tidak ada saham valid hari ini. Cash is King.</i>"

    # Kirim Telegram
    if TELEGRAM_OK:
        send_telegram_message(telegram_msg)
        print("\n✅ Update Telegram terkirim.")

if __name__ == "__main__":
    run_scanner()