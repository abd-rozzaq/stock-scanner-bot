#!/usr/bin/env python3
# ================================================================================
# scanner_bsjp_v3_dualsession.py — BSJP ULTIMATE DUAL-SESSION SCANNER
# Desc    : Scanner saham IDX dengan konfirmasi Dual Session untuk
#           meminimalkan false signal dan noise.
#           Sesi 1 (12:00-13:30) → Sesi 2 (15:45-15:55) → Intersection = TRADE
# Version : 3.0.0-DualSession
# ================================================================================

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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ======================================================
# 0. KONFIGURASI & SETUP
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
DATA_DIR = os.path.join(BASE_DIR, "data")

for d in [LOG_DIR, SESSION_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"scanner_v3_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
# 1. THRESHOLD FILTER (DIKERASKAN UNTUK V3)
# ======================================================

# --- F1: PRICE ACTION & MOMENTUM (BSJP Base) ---
CLOSE_TO_HIGH_RATIO = 0.97          # Lebih ketat: 0.98 → 0.97
MIN_BODY_RATIO = 0.40               # Body candle minimal 40% dari range
MAX_UPPER_WICK_RATIO = 0.15         # Upper shadow maksimal 15% (anti-rejection)
MIN_CLOSE_POSITION = 0.65           # Close di upper 65% range (0.5 = tengah)
MIN_PRICE_CHANGE_PCT = 3.0
MAX_PRICE_CHANGE_PCT = 12.0         # Batas atas: hindari blow-off top
MAX_GAP_UP_PCT = 5.0                # Hindari gap up terlalu besar (gap & crap)
BULLISH_CANDLE = True               # Close harus > Open

# --- F2: TREND & STRUCTURE ---
MA20_PERIOD = 20
MA50_PERIOD = 50
MAX_DISTANCE_FROM_MA20 = 0.08       # Close tidak boleh > 8% dari MA20 (anti-chasing)
MA20_SLOPE_MIN = 0.0                # MA20 harus flat atau naik
REQUIRE_BREAKOUT = True             # Harus break high 10 hari terakhir atau consolidation
REQUIRE_GOLDEN_STRUCTURE = True     # Close > MA20 > MA50

# --- F3: VOLUME & LIKUIDITAS ---
VOL_SPIKE_MIN = 1.5                 # Minimal 1.5x Vol_MA20 (dari 2x, lebih awal)
VOL_SPIKE_MAX = 4.0                 # Maksimal 4x Vol_MA20 (anti anomali/distribusi)
VOL_MA5_GT_MA20 = True              # Volume trend harus naik
MIN_VALUE_IDR = 100_000_000
MIN_VALUE_MA20_IDR = 2_000_000_000  # Dikeraskan dari 1B → 2B (hindari saham kecil)
MIN_FREQUENCY = 1000

# --- F4: MOMENTUM INDICATOR ---
RSI_PERIOD = 14
RSI_MIN = 45                        # Hindari saham lemah
RSI_MAX = 65                        # Dikeraskan dari 80 → 65 (hindari overbought!)
RSI_SLOPE_MIN = 0                   # RSI 3-hari harus flat atau naik
USE_MACD_CONFIRMATION = True        # MACD histogram positif atau improving

# --- F5: BANDARMOLOGI (AKUMULASI) ---
BANDAR_MA10_PERIOD = 10
BANDAR_MA20_PERIOD = 20
MIN_POSITIVE_BANDAR_DAYS = 3        # Minimal 3 dari 5 hari terakhir akumulasi positif
REQUIRE_BANDAR_MA_CROSS = True      # Bandar_MA10 > Bandar_MA20
REQUIRE_BANDAR_MA20_POSITIVE = True # Bandar_MA20 harus positif

# --- F6: VOLATILITAS & RISK ---
ATR_PERIOD = 14
MAX_ATR_PCT = 8.0                   # ATR tidak boleh > 8% dari harga (anti gorengan)
MAX_RANGE_VS_ATR = 2.0              # Range hari ini tidak > 2x ATR

# --- Risk Management ---
TP_PERCENT = 0.06                   # 6% sesuai request
CL_PERCENT = 0.05                   # -5% hard stop
MAX_HOLDING_DAYS = 7                # 7 hari sesuai request

# --- Data Fetch ---
YFINANCE_PERIOD = "6mo"
YFINANCE_INTERVAL = "1d"
MAX_MESSAGE_LENGTH = 4096

# --- Dual Session ---
SESSION_1_START = (12, 0)           # 12:00
SESSION_1_END = (13, 30)            # 13:30
SESSION_2_START = (15, 45)          # 15:45
SESSION_2_END = (15, 55)            # 15:55
MAX_SESSION_PRICE_DROP = 0.02       # S2 price tidak boleh turun > 2% dari S1

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
        logger.info(f"Telegram aktif. Chat ID: {TELEGRAM_CHAT_ID}")
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram dinonaktifkan.")

def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        return False
    chunks = []
    current = ""
    for line in message.split("\n"):
        if len(line) > MAX_MESSAGE_LENGTH:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(line), MAX_MESSAGE_LENGTH):
                chunks.append(line[i:i+MAX_MESSAGE_LENGTH])
            continue
        test = current + line + "\n"
        if len(test) > MAX_MESSAGE_LENGTH:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current = test
    if current.strip():
        chunks.append(current.strip())
    
    success = True
    for i, chunk in enumerate(chunks):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}
            resp = requests.post(url, data=data, timeout=15)
            if resp.status_code != 200:
                logger.error(f"Telegram error: {resp.text}")
                success = False
        except Exception as e:
            logger.error(f"Telegram exception: {e}")
            success = False
        if i < len(chunks) - 1:
            time.sleep(1)
    return success

# ======================================================
# 3. DATA FETCHING
# ======================================================
def fetch_stock_data(ticker: str, period: str = YFINANCE_PERIOD, 
                     interval: str = YFINANCE_INTERVAL) -> pd.DataFrame:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(
            symbol, 
            period=period, 
            interval=interval, 
            progress=False, 
            auto_adjust=False,
            threads=False
        )
        if df.empty or len(df) < MA50_PERIOD + 5:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        
        df = df[required].copy().dropna()
        return df
    except Exception as e:
        logger.debug(f"Fetch error {ticker}: {e}")
        return pd.DataFrame()

# ======================================================
# 4. TECHNICAL INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def calculate_slope(series: pd.Series, lookback: int = 5) -> float:
    """Return slope using linear regression on last N points."""
    if len(series) < lookback:
        return 0.0
    y = series.iloc[-lookback:].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

# ======================================================
# 5. CORE ANALYSIS ENGINE
# ======================================================
@dataclass
class ScanResult:
    ticker: str
    close: int
    open: int
    high: int
    low: int
    change_pct: float
    volume: int
    value_b: float
    rsi: float
    rsi_slope: float
    ma20: float
    ma50: float
    dist_ma20_pct: float
    vol_ratio: float
    atr_pct: float
    bandar_ma20: float
    entry: int
    tp: int
    cl: int
    session1_price: Optional[float] = None
    session2_price: Optional[float] = None
    session_diff_pct: Optional[float] = None

def analyze_stock(ticker: str) -> Optional[ScanResult]:
    df = fetch_stock_data(ticker)
    if df.empty or len(df) < MA50_PERIOD + 5:
        return None
    
    # --- Calculate Indicators ---
    df['Value'] = df['Close'] * df['Volume']
    df['Value_MA20'] = df['Value'].rolling(MA20_PERIOD).mean()
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()
    df['Vol_MA20'] = df['Volume'].rolling(MA20_PERIOD).mean()
    df['MA20'] = df['Close'].rolling(MA20_PERIOD).mean()
    df['MA50'] = df['Close'].rolling(MA50_PERIOD).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['ATR'] = calculate_atr(df)
    
    # MACD
    _, _, df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Bandarmologi Proxy
    df['Bandar_Value'] = np.where(
        df['Close'] > df['Open'], 
        df['Value'] * 0.5, 
        df['Value'] * -0.5
    )
    df['Bandar_MA10'] = df['Bandar_Value'].rolling(BANDAR_MA10_PERIOD).mean()
    df['Bandar_MA20'] = df['Bandar_Value'].rolling(BANDAR_MA20_PERIOD).mean()
    
    # High/Low references
    df['High_10d'] = df['High'].rolling(10).max()
    df['Prev_High'] = df['High'].shift(1)
    
    # Drop NaN
    df = df.dropna()
    if len(df) < 5:
        return None
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) > 2 else prev
    
    if current['Close'] <= 0 or current['Volume'] <= 0:
        return None
    
    # --- Basic Metrics ---
    price_change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
    gap_pct = ((current['Open'] - prev['Close']) / prev['Close']) * 100
    daily_range = current['High'] - current['Low']
    body = abs(current['Close'] - current['Open'])
    upper_wick = current['High'] - max(current['Close'], current['Open'])
    lower_wick = min(current['Close'], current['Open']) - current['Low']
    close_position = (current['Close'] - current['Low']) / daily_range if daily_range > 0 else 0.5
    
    rsi_val = current['RSI']
    rsi_slope = calculate_slope(df['RSI'].dropna(), 3)
    ma20_val = current['MA20']
    ma50_val = current['MA50']
    dist_ma20_pct = (current['Close'] - ma20_val) / ma20_val
    vol_ratio = current['Volume'] / current['Vol_MA20'] if current['Vol_MA20'] > 0 else 0
    atr_pct = (current['ATR'] / current['Close']) * 100
    
    # --- FILTER 1: PRICE ACTION (Strict) ---
    f1_checks = [
        current['Close'] >= (current['High'] * CLOSE_TO_HIGH_RATIO),
        close_position >= MIN_CLOSE_POSITION,
        (upper_wick / daily_range <= MAX_UPPER_WICK_RATIO) if daily_range > 0 else False,
        (body / daily_range >= MIN_BODY_RATIO) if daily_range > 0 else False,
        MIN_PRICE_CHANGE_PCT <= price_change_pct <= MAX_PRICE_CHANGE_PCT,
        gap_pct <= MAX_GAP_UP_PCT,
        (not BULLISH_CANDLE) or (current['Close'] > current['Open']),
    ]
    pass_f1 = all(f1_checks)
    
    # --- FILTER 2: TREND & STRUCTURE ---
    ma20_slope = calculate_slope(df['MA20'].dropna(), 5)
    f2_checks = [
        current['Close'] > ma20_val,
        current['Close'] > ma50_val,
        ma20_val > ma50_val if REQUIRE_GOLDEN_STRUCTURE else True,
        dist_ma20_pct <= MAX_DISTANCE_FROM_MA20,
        ma20_slope >= MA20_SLOPE_MIN,
    ]
    if REQUIRE_BREAKOUT:
        # Harus break high kemarin ATAU high 10 hari (fresh momentum)
        breakout = (current['Close'] > current['Prev_High']) or (current['Close'] >= current['High_10d'] * 0.995)
        f2_checks.append(breakout)
    pass_f2 = all(f2_checks)
    
    # --- FILTER 3: VOLUME & LIKUIDITAS (Strict) ---
    f3_checks = [
        current['Volume'] > MIN_FREQUENCY,
        VOL_SPIKE_MIN <= vol_ratio <= VOL_SPIKE_MAX,
        current['Volume'] > current['Vol_MA5'],
        current['Value'] >= MIN_VALUE_IDR,
        current['Value_MA20'] >= MIN_VALUE_MA20_IDR,
    ]
    if VOL_MA5_GT_MA20:
        f3_checks.append(current['Vol_MA5'] > current['Vol_MA20'])
    pass_f3 = all(f3_checks)
    
    # --- FILTER 4: MOMENTUM (RSI & MACD) ---
    f4_checks = [
        RSI_MIN <= rsi_val <= RSI_MAX,
        rsi_slope >= RSI_SLOPE_MIN,
    ]
    if USE_MACD_CONFIRMATION:
        macd_ok = (current['MACD_Hist'] > 0) or (current['MACD_Hist'] > prev['MACD_Hist'])
        f4_checks.append(macd_ok)
    pass_f4 = all(f4_checks)
    
    # --- FILTER 5: BANDARMOLOGI (Akumulasi Berkelanjutan) ---
    bandar_last_5 = df['Bandar_Value'].iloc[-5:].values
    positive_days = sum(1 for v in bandar_last_5 if v > 0)
    
    f5_checks = [
        current['Bandar_Value'] > 0,
        positive_days >= MIN_POSITIVE_BANDAR_DAYS,
    ]
    if REQUIRE_BANDAR_MA_CROSS:
        f5_checks.append(current['Bandar_MA10'] > current['Bandar_MA20'])
    if REQUIRE_BANDAR_MA20_POSITIVE:
        f5_checks.append(current['Bandar_MA20'] > 0)
    pass_f5 = all(f5_checks)
    
    # --- FILTER 6: VOLATILITAS ---
    range_vs_atr = daily_range / current['ATR'] if current['ATR'] > 0 else 999
    f6_checks = [
        atr_pct <= MAX_ATR_PCT,
        range_vs_atr <= MAX_RANGE_VS_ATR,
    ]
    pass_f6 = all(f6_checks)
    
    # --- FINAL GATE ---
    if not (pass_f1 and pass_f2 and pass_f3 and pass_f4 and pass_f5 and pass_f6):
        return None
    
    entry = int(current['Close'])
    return ScanResult(
        ticker=ticker,
        close=entry,
        open=int(current['Open']),
        high=int(current['High']),
        low=int(current['Low']),
        change_pct=round(float(price_change_pct), 2),
        volume=int(current['Volume']),
        value_b=round(float(current['Value']) / 1e9, 2),
        rsi=round(float(rsi_val), 1),
        rsi_slope=round(float(rsi_slope), 2),
        ma20=round(float(ma20_val), 1),
        ma50=round(float(ma50_val), 1),
        dist_ma20_pct=round(float(dist_ma20_pct) * 100, 2),
        vol_ratio=round(float(vol_ratio), 2),
        atr_pct=round(float(atr_pct), 2),
        bandar_ma20=round(float(current['Bandar_MA20']) / 1e9, 2),
        entry=entry,
        tp=int(entry * (1 + TP_PERCENT)),
        cl=int(entry * (1 - CL_PERCENT))
    )

# ======================================================
# 6. DUAL SESSION MANAGER
# ======================================================
def get_session_file(date_str: str) -> str:
    return os.path.join(SESSION_DIR, f"session1_{date_str}.json")

def save_session1(results: List[Dict], date_str: str):
    filepath = get_session_file(date_str)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Session 1 disimpan: {filepath} ({len(results)} kandidat)")

def load_session1(date_str: str) -> List[Dict]:
    filepath = get_session_file(date_str)
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_current_session() -> Tuple[int, dt.datetime]:
    """Return (session_number, datetime_now). 0=off, 1=sesi1, 2=sesi2"""
    now = dt.datetime.now()
    t = now.time()
    
    s1_start = dt.time(*SESSION_1_START)
    s1_end = dt.time(*SESSION_1_END)
    s2_start = dt.time(*SESSION_2_START)
    s2_end = dt.time(*SESSION_2_END)
    
    if s1_start <= t <= s1_end:
        return 1, now
    elif s2_start <= t <= s2_end:
        return 2, now
    return 0, now

# ======================================================
# 7. FORMATTING & OUTPUT
# ======================================================
def format_telegram_message(results: List[ScanResult], scan_time: str, 
                           total_scanned: int, session_label: str) -> str:
    lines = [
        f"<b>Qwen Screener</b>",
        f"<i>{scan_time} | {session_label}</i>",
        "",
        f"📊 Scanned: {total_scanned} | <b>✅ Match: {len(results)} saham</b>",
        f"⏳ Max Hold: {MAX_HOLDING_DAYS} hari | TP: {int(TP_PERCENT*100)}% | CL: -{int(CL_PERCENT*100)}%",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]
    
    if not results:
        lines.append("\n<i>Filter ketat aktif. Tidak ada saham yang lolos semua gate hari ini.</i>")
        return "\n".join(lines)
    
    for r in results:
        detail = (f"<b>{r.ticker}</b> | {r.change_pct:+.2f}% | "
                 f"RSI: {r.rsi:.0f} | Val: {r.value_b:.2f}B")
        risk = f"   🎯 Entry: {r.entry:,} → TP: {r.tp:,} | CL: {r.cl:,}"
        tech = (f"   📈 MA20: {r.dist_ma20_pct:+.1f}% | Vol: {r.vol_ratio:.1f}x | "
               f"ATR: {r.atr_pct:.1f}%")
        session_info = ""
        if r.session_diff_pct is not None:
            session_info = f"   🔄 S1→S2: {r.session_diff_pct:+.2f}%"
        
        lines.extend([detail, risk, tech, session_info, ""])
    
    return "\n".join(lines)

def print_results_table(results: List[ScanResult]):
    if not results:
        print("\n⚠️  Tidak ada saham yang lolos filter.\n")
        return
    
    print(f"\n{'='*90}")
    print(f"{'TICKER':<8} {'CLOSE':>8} {'CHG%':>7} {'RSI':>5} {'VAL(B)':>8} "
          f"{'VOLx':>6} {'MA20%':>7} {'ATR%':>6} {'ENTRY':>8} {'TP':>8} {'CL':>8}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r.ticker:<8} {r.close:>8,} {r.change_pct:>+7.2f} {r.rsi:>5.0f} "
              f"{r.value_b:>8.2f} {r.vol_ratio:>6.1f} {r.dist_ma20_pct:>+7.1f} "
              f"{r.atr_pct:>6.1f} {r.entry:>8,} {r.tp:>8,} {r.cl:>8,}")
    print(f"{'='*90}\n")

# ======================================================
# 8. MAIN EXECUTION
# ======================================================
def load_tickers() -> List[str]:
    filepath = os.path.join(DATA_DIR, "data.csv")
    if not os.path.exists(filepath):
        filepath = "data.csv"  # fallback
    try:
        df = pd.read_csv(filepath)
        col = next((c for c in ['Ticker','ticker','Kode','kode','Symbol'] 
                   if c in df.columns), df.columns[0])
        tickers = []
        for t in df[col].dropna():
            t = str(t).strip().upper().replace(".JK", "")
            if len(t) >= 3:
                tickers.append(t)
        return sorted(list(set(tickers)))
    except Exception as e:
        logger.error(f"Gagal load ticker: {e}")
        return []

def scan_all_tickers(tickers: List[str]) -> List[ScanResult]:
    results = []
    total = len(tickers)
    
    print(f"\n🔍 Scanning {total} saham dengan 6-Filter Gate (V3)...\n")
    
    for i, ticker in enumerate(tickers):
        print(f"\r  [{'█' * int((i+1)/total*20):<20}] {i+1}/{total} ({(i+1)/total*100:5.1f}%) | Checking: {ticker:<6}", end="", flush=True)
        try:
            res = analyze_stock(ticker)
            if res:
                results.append(res)
                print(f"\n  ✅ MATCH: {ticker} (+{res.change_pct}%, RSI:{res.rsi:.0f}, Vol:{res.vol_ratio:.1f}x)")
        except KeyboardInterrupt:
            print("\n\n⚠️  Dihentikan user.")
            break
        except Exception as e:
            logger.debug(f"Error {ticker}: {e}")
            continue
    
    print("\n")
    results.sort(key=lambda x: (-x.change_pct, -x.vol_ratio))
    return results

def run_session1(tickers: List[str], date_str: str):
    """Jalankan screening sesi 1 (12:00-13:30) dan simpan kandidat."""
    print(f"\n{'='*60}")
    print(f"  ☀️  SESI 1: BREAK SCREENING (12:00-13:30)")
    print(f"{'='*60}")
    
    results = scan_all_tickers(tickers)
    dict_results = [asdict(r) for r in results]
    save_session1(dict_results, date_str)
    
    print(f"\n📦 {len(results)} kandidat disimpan untuk konfirmasi Sesi 2.")
    print(f"   ⏰ Jalankan lagi pukul 15:45-15:55 untuk konfirmasi final.\n")
    
    if TELEGRAM_OK:
        msg = format_telegram_message(results, dt.datetime.now().strftime("%Y-%m-%d %H:%M"), len(tickers), "Sesi 1 (Pending)")
        send_telegram_message(msg)
    
    return results

def run_session2(tickers: List[str], date_str: str):
    """Jalankan screening sesi 2 (15:45-15:55), intersect dengan sesi 1."""
    print(f"\n{'='*60}")
    print(f"  🌙 SESI 2: FINAL CONFIRMATION (15:45-15:55)")
    print(f"{'='*60}")
    
    s1_data = load_session1(date_str)
    if not s1_data:
        print("❌ Tidak ada data Sesi 1. Jalankan Sesi 1 terlebih dahulu (12:00-13:30).")
        return []
    
    s1_tickers = {d['ticker'] for d in s1_data}
    s1_prices = {d['ticker']: d['close'] for d in s1_data}
    
    print(f"📥 Loaded {len(s1_tickers)} kandidat dari Sesi 1.")
    
    s2_results = scan_all_tickers(tickers)
    
    # Intersection: hanya yang ada di S1 dan S2
    confirmed = []
    for r in s2_results:
        if r.ticker in s1_tickers:
            s1_price = s1_prices[r.ticker]
            # Konfirmasi: harga S2 tidak boleh turun lebih dari 2% dari S1
            # Ini menunjukkan kekuatan sustain sepanjang hari
            if r.close >= s1_price * (1 - MAX_SESSION_PRICE_DROP):
                r.session1_price = s1_price
                r.session2_price = r.close
                r.session_diff_pct = round((r.close - s1_price) / s1_price * 100, 2)
                confirmed.append(r)
            else:
                print(f"   ⛔ {r.ticker} ditolak: turun {((r.close-s1_price)/s1_price*100):+.2f}% dari Sesi 1")
    
    print(f"\n{'='*60}")
    print(f"  🎯 FINAL TRADE SIGNALS: {len(confirmed)} saham")
    print(f"{'='*60}")
    
    print_results_table(confirmed)
    
    # Telegram
    if TELEGRAM_OK:
        msg = format_telegram_message(confirmed, dt.datetime.now().strftime("%Y-%m-%d %H:%M"), len(tickers), "Sesi 2 (CONFIRMED)")
        send_telegram_message(msg)
    
    # Cleanup session file
    session_file = get_session_file(date_str)
    if os.path.exists(session_file):
        os.remove(session_file)
        logger.info(f"Session file dihapus: {session_file}")
    
    return confirmed

def run_full_backtest_mode(tickers: List[str]):
    """Mode khusus: simulasi dual session dalam 1 run (untuk testing)."""
    print(f"\n{'='*60}")
    print(f"  🧪 FULL MODE: Simulasi Dual Session (TESTING)")
    print(f"{'='*60}")
    
    results = scan_all_tickers(tickers)
    print(f"\n⚠️  Mode testing: menampilkan semua hasil (tanpa konfirmasi dual session).")
    print(f"   Untuk produksi, jalankan 2x: Sesi 1 (siang) dan Sesi 2 (sore).\n")
    
    print_results_table(results)
    
    if TELEGRAM_OK:
        msg = format_telegram_message(results, dt.datetime.now().strftime("%Y-%m-%d %H:%M"), len(tickers), "TEST MODE")
        send_telegram_message(msg)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="BSJP V3 Dual-Session Scanner")
    parser.add_argument("--mode", choices=["auto", "session1", "session2", "full"], 
                       default="auto", help="Mode eksekusi")
    parser.add_argument("--force-date", type=str, default=None, 
                       help="Force tanggal untuk session file (YYYYMMDD)")
    args = parser.parse_args()
    
    tickers = load_tickers()
    if not tickers:
        print("❌ Data ticker tidak ditemukan. Pastikan data/data.csv tersedia.")
        return
    
    now = dt.datetime.now()
    date_str = args.force_date or now.strftime("%Y%m%d")
    session, _ = get_current_session()
    
    print(f"\n{'='*60}")
    print(f"  BSJP V3 DUAL-SESSION SCANNER")
    print(f"  Tanggal: {now.strftime('%Y-%m-%d')} | Jam: {now.strftime('%H:%M')}")
    print(f"  Total Ticker: {len(tickers)}")
    print(f"{'='*60}")
    
    if args.mode == "session1":
        run_session1(tickers, date_str)
    elif args.mode == "session2":
        run_session2(tickers, date_str)
    elif args.mode == "full":
        run_full_backtest_mode(tickers)
    else:  # auto
        if session == 1:
            run_session1(tickers, date_str)
        elif session == 2:
            run_session2(tickers, date_str)
        else:
            print(f"\n⏰ Di luar jam trading scanner.")
            print(f"   Sesi 1: {SESSION_1_START[0]:02d}:{SESSION_1_START[1]:02d}-{SESSION_1_END[0]:02d}:{SESSION_1_END[1]:02d}")
            print(f"   Sesi 2: {SESSION_2_START[0]:02d}:{SESSION_2_START[1]:02d}-{SESSION_2_END[0]:02d}:{SESSION_2_END[1]:02d}")
            print(f"\n   Gunakan --mode session1 / session2 / full untuk force run.\n")

if __name__ == "__main__":
    main()