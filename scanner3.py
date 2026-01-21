# improved_scanner.py - UT BOT SCANNER v6.0 (OPTIMIZED)
# STRATEGY: "Akumulasi Pelan Naik Konstan" (Smart Money Footprint)
# IMPROVEMENTS: Fresh Signal Logic, Robust HAKA Proxy, Type Hinting

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import time
import warnings
from typing import Dict, List, Optional, Union

warnings.filterwarnings("ignore")

# ======================================================
# 1. CONFIGURATION
# ======================================================
# Coba load config, jika gagal pakai dummy (agar tidak error saat copy-paste)
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    print("⚠️  config.py tidak ditemukan. Telegram notifikasi dinonaktifkan.")

# SETTINGS
JAKARTA_TZ = dt.timezone(dt.timedelta(hours=7))
LOOKBACK_PERIOD = "6mo"

# Strategy Parameters
PARAMS = {
    "MAX_DAILY_GAIN": 3.0,          # % (Jangan terlalu mencolok)
    "MAX_ATR_PERCENT": 3.0,         # % (Volatilitas rendah/stabil)
    "MIN_BUYING_PRESSURE": 60.0,    # % (Proxy untuk HAKA)
    "MIN_TURNOVER_IDR": 100_000_000, # Min transaksi 100 Juta (agar likuid)
    "VOL_RATIO_MIN": 1.0,           # Volume > MA20
    "VOL_RATIO_MAX": 2.5,           # Volume < 2.5x (Jangan Euforia)
    "EMA_SHORT": 5,
    "EMA_MID": 10,
    "EMA_LONG": 50
}

# ======================================================
# 2. DATA UTILITIES
# ======================================================
def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch data OHLCV dari Yahoo Finance dengan format yang bersih.
    Menambahkan suffix .JK otomatis.
    """
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    try:
        # Auto adjust=False agar kita dapat raw price untuk teknikal murni
        df = yf.download(
            ticker, 
            period=period, 
            interval="1d", 
            progress=False,
            auto_adjust=False,
            multi_level_index=False # Fix untuk yfinance versi baru
        )
        
        # Data Cleaning
        if df.empty or len(df) < 50:
            return pd.DataFrame()
            
        # Ensure columns exist (Handle case sensitive)
        df.columns = [c.capitalize() for c in df.columns]
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            return pd.DataFrame()
            
        return df[required]
        
    except Exception as e:
        # print(f"Error fetching {ticker}: {e}") # Debug only
        return pd.DataFrame()

# ======================================================
# 3. TECHNICAL INDICATORS (VECTORIZED)
# ======================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung semua indikator teknikal secara vektor.
    [FIXED] Masalah HAKA NaN karena Index Alignment.
    """
    # 1. EMAs
    df['EMA5'] = df['Close'].ewm(span=PARAMS['EMA_SHORT'], adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=PARAMS['EMA_MID'], adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=PARAMS['EMA_LONG'], adjust=False).mean()
    
    # 2. Daily Gain (%)
    df['PrevClose'] = df['Close'].shift(1)
    df['Gain'] = ((df['Close'] - df['PrevClose']) / df['PrevClose']) * 100
    
    # 3. ATR Percentage (Volatility)
    h_l = df['High'] - df['Low']
    h_pc = (df['High'] - df['PrevClose']).abs()
    l_pc = (df['Low'] - df['PrevClose']).abs()
    
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATRP'] = (atr / df['Close']) * 100
    
    # 4. Volume Patterns
    df['VolMA20'] = df['Volume'].rolling(20).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA20']
    
    # 5. Buying Pressure (Proxy HAKA) - FIXED
    range_bar = df['High'] - df['Low']
    
    # Hitung raw intensity menggunakan numpy
    raw_haka = np.where(
        range_bar > 0, 
        (df['Close'] - df['Low']) / range_bar, 
        0.5 # Jika High=Low (Doji), asumsikan Netral (0.5)
    )
    
    # CRITICAL FIX: Masukkan kembali ke Pandas Series DENGAN INDEX ASLI
    haka_series = pd.Series(raw_haka, index=df.index)
    
    # Rolling mean 5 hari & dikali 100 untuk jadi persen
    df['BuyPressMA5'] = haka_series.rolling(window=5).mean() * 100
    
    # Isi NaN awal (karena rolling) dengan nilai pertama yang tersedia agar tidak error
    df['BuyPressMA5'] = df['BuyPressMA5'].fillna(method='bfill')

    # 6. Turnover (Value)
    df['Turnover'] = df['Close'] * df['Volume']
    
    return df

# ======================================================
# 4. SIGNAL LOGIC & FRESHNESS
# ======================================================
def check_strategy(df: pd.DataFrame) -> Optional[Dict]:
    """
    Mengecek apakah candle TERAKHIR memenuhi kriteria strategi.
    Mengembalikan Dict summary jika valid, None jika tidak.
    """
    if len(df) < 50: return None
    
    # Kita butuh baris terakhir (current) dan sebelumnya (prev) untuk logic Fresh
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- LOGIC FILTER (HARD RULES) ---
    
    # 1. Trend Filter
    if not (curr['Close'] > curr['EMA5']): return None        # Short term UP
    if not (curr['EMA10'] > curr['EMA50']): return None      # Mid term UP
    
    # 2. Stability Filter (Akumulasi Tenang)
    if curr['Gain'] > PARAMS['MAX_DAILY_GAIN']: return None   # Tidak boleh spike
    if curr['ATRP'] > PARAMS['MAX_ATR_PERCENT']: return None  # Tidak boleh volatile
    
    # 3. Volume Filter (Interest but not Euphoria)
    if curr['VolRatio'] < PARAMS['VOL_RATIO_MIN']: return None # Sepi
    if curr['VolRatio'] > PARAMS['VOL_RATIO_MAX']: return None # Terlalu ramai
    
    # 4. HAKA / Buying Pressure
    if curr['BuyPressMA5'] < PARAMS['MIN_BUYING_PRESSURE']: return None
    
    # 5. Liquidity
    if curr['Turnover'] < PARAMS['MIN_TURNOVER_IDR']: return None
    
    # --- FRESH SIGNAL CHECK ---
    # Definisikan "Fresh" sebagai momen ketika salah satu indikator kunci baru saja cross
    # Misal: Kemarin Harga di bawah EMA5, atau Kemarin Volume sepi.
    
    is_fresh_breakout = (prev['Close'] <= prev['EMA5']) and (curr['Close'] > curr['EMA5'])
    is_fresh_volume   = (prev['VolRatio'] < 1.0) and (curr['VolRatio'] >= 1.0)
    
    status_label = "FRESH 🟢" if (is_fresh_breakout or is_fresh_volume) else "ACCUM 🟡"
    
    return {
        "price": curr['Close'],
        "gain": curr['Gain'],
        "vol_ratio": curr['VolRatio'],
        "haka_proxy": curr['BuyPressMA5'],
        "turnover_b": curr['Turnover'] / 1_000_000_000, # Milyar
        "status": status_label,
        "score": calculate_score(curr)
    }

def calculate_score(row) -> int:
    """Menghitung skor kualitas setup (0-100)"""
    score = 60 # Base score karena sudah lolos filter
    
    # Bonus HAKA kuat
    if row['BuyPressMA5'] > 70: score += 10
    # Bonus Stability super tenang
    if row['ATRP'] < 2.0: score += 10
    # Bonus Volume ideal (sedikit di atas rata-rata)
    if 1.2 <= row['VolRatio'] <= 1.8: score += 10
    # Bonus Trend EMA Rapi (Perfect Alignment)
    if row['EMA5'] > row['EMA10'] > row['EMA50']: score += 10
    
    return min(score, 100)

# ======================================================
# 5. MAIN ORCHESTRATOR
# ======================================================
def send_telegram_report(results: List[Dict]):
    if not TELEGRAM_OK: return
    
    msg = f"🔍 <b>QUIET ACCUMULATION SCANNER</b>\n"
    msg += f"📅 {dt.datetime.now(JAKARTA_TZ).strftime('%d-%m-%Y %H:%M WIB')}\n"
    msg += f"Strategi: Akumulasi Pelan, Naik Konstan\n\n"
    
    if not results:
        msg += "💤 <b>HASIL SCAN: NIHIL (ZONK)</b>\n"
        msg += "Tidak ada saham yang memenuhi kriteria 'Quiet Accumulation' hari ini.\n"
        msg += "Pasar mungkin sedang terlalu volatil atau sepi.\n"
    else:
        # Group by Status
        fresh_signals = [r for r in results if "FRESH" in r['status']]
        accum_signals = [r for r in results if "ACCUM" in r['status']]
        
        # --- SECTION FRESH SIGNALS ---
        if fresh_signals:
            msg += "🚀 <b>FRESH SIGNALS (Baru Mulai):</b>\n"
            for r in fresh_signals[:5]:
                msg += f"• <b>{r['ticker']}</b> ({r['score']})\n"
                msg += f"  P:{r['price']:.0f} | HAKA:{r['haka_proxy']:.0f}% | Vol:{r['vol_ratio']:.1f}x\n"
            msg += "\n"
            
        # --- SECTION ONGOING ACCUMULATION ---
        if accum_signals:
            msg += "⏳ <b>ONGOING ACCUMULATION:</b>\n"
            for r in accum_signals[:5]:
                # [UPDATED] Sekarang menampilkan detail yang sama
                msg += f"• {r['ticker']} ({r['score']})\n"
                msg += f"  P:{r['price']:.0f} | HAKA:{r['haka_proxy']:.0f}% | Vol:{r['vol_ratio']:.1f}x\n"
                
        msg += "\n<i>Note: Fresh = Baru cross EMA5 atau Vol > MA20 hari ini.</i>"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})
        print("✅ Telegram sent.")
    except Exception as e:
        print(f"❌ Telegram failed: {e}")

def main():
    print("="*60)
    print("🚀 STARTING SCANNER: Quiet Accumulation Protocol")
    print("="*60)
    
    # 1. Load Tickers
    try:
        # Ganti path ini sesuai lokasi file CSV anda
        tickers_df = pd.read_csv("data/data.csv", header=None) 
        all_tickers = tickers_df.iloc[:, 0].tolist() # Asumsi kolom pertama adalah Kode
    except Exception:
        print("⚠️ File data/data.csv tidak ditemukan. Menggunakan sample tickers.")
        all_tickers = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "UNTR", "ICBP"] # Sample
        
    results = []
    start_time = time.time()
    
    for i, ticker in enumerate(all_tickers):
        # Progress Log
        print(f"\rScanning {i+1}/{len(all_tickers)}: {ticker}...", end="", flush=True)
        
        # 1. Fetch
        df = fetch_stock_data(ticker, period=LOOKBACK_PERIOD)
        if df.empty: continue
            
        # 2. Indicators
        df = calculate_indicators(df)
        
        # 3. Check Logic
        res = check_strategy(df)
        if res:
            res['ticker'] = ticker
            results.append(res)
            # print(f" [FOUND] {ticker} Score: {res['score']}") # Optional Log
            
    print(f"\n\n✅ Scan Finished in {time.time() - start_time:.2f}s")
    print(f"Saham Terpilih: {len(results)}")
    
    # Sort by Status (Fresh first) then Score
    results.sort(key=lambda x: (x['status'], x['score']), reverse=True)
    
    # Print to console
    print("-" * 50)
    print(f"{'TICKER':<8} | {'STATUS':<10} | {'PRICE':<8} | {'HAKA%':<6} | {'VOL(x)':<6}")
    print("-" * 50)
    for r in results:
        print(f"{r['ticker']:<8} | {r['status']:<10} | {r['price']:<8.0f} | {r['haka_proxy']:<6.0f} | {r['vol_ratio']:<6.1f}")
        
    # Send Report
    send_telegram_report(results)

if __name__ == "__main__":
    main()
