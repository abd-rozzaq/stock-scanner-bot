# scanner_pro.py - UT BOT v4.0 (ULTRA SELECTIVE FILTERS)
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import time
from typing import List, Dict, Optional, Tuple

# ======================================================
# 1. CONFIGURATION
# ======================================================
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    print("⚠️  config.py tidak ditemukan/tidak lengkap - Telegram dinonaktifkan.")

# Strategy Settings (TIGHTENED)
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 50
RSI_MAX_THRESHOLD = 70           # 🆕 Hindari Overbought
EMA_TREND_PERIOD = 200
MIN_DAILY_VALUE = 1_000_000_000  

# 🆕 NEW FILTERS
MIN_VOLUME_RATIO = 1.5           # Volume hari ini >= 1.5x rata-rata 20 hari
MIN_ADX_VALUE = 25.0             # Trend Strength (ADX minimal 25)
MIN_PRICE_ROC = 3.0              # Harga naik minimal 3% dalam 5 hari
MAX_DISTANCE_FROM_EMA200 = 15.0  # Max 15% di atas EMA200 (hindari overextended)

# ======================================================
# 2. DATA LOADER
# ======================================================
def load_bei_tickers() -> List[str]:
    """Load tickers saham BEI dari file CSV."""
    try:
        df = pd.read_csv("data/data.csv", header=None)
        tickers = df[0].astype(str).tolist()
        tickers = [t.strip().upper() for t in tickers]
        print(f"✅ {len(tickers)} saham BEI berhasil dimuat.")
        return tickers
    except Exception as e:
        print(f"❌ Gagal load data.csv: {e}")
        return ["BBCA", "BBRI", "TLKM", "ASII", "BMRI"]

# ======================================================
# 3. TELEGRAM UTILS
# ======================================================
def send_telegram_message(text: str) -> None:
    if not TELEGRAM_OK:
        print(f"📨 [LOG ONLY] Pesan:\n{text}")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# ======================================================
# 4. TECHNICAL ANALYSIS ENGINE (EXPANDED)
# ======================================================
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Menghitung Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """Menghitung nilai RSI terakhir."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Menghitung Average Directional Index (ADX) untuk trend strength.
    ADX > 25 = Trend Kuat, ADX < 20 = Sideways/Lemah
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0

def calculate_roc(series: pd.Series, period: int = 5) -> float:
    """
    Rate of Change (ROC) - Persentase perubahan harga.
    ROC positif = Momentum naik.
    """
    current = series.iloc[-1]
    past = series.iloc[-period-1]
    roc = ((current - past) / past) * 100 if past != 0 else 0.0
    return float(roc)

def calculate_ut_bot(
    df: pd.DataFrame, 
    key_value: float = 2.0, 
    atr_period: int = 10
) -> Tuple[bool, bool]:
    """
    Menghitung sinyal UT Bot menggunakan Numpy untuk performa tinggi.
    Returns: (is_fresh_buy, is_fresh_sell)
    """
    if len(df) < max(atr_period, 50):
        return False, False

    open_arr = df['Open'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    close_arr = df['Close'].values
    
    src = (open_arr + high_arr + low_arr + close_arr) / 4

    prev_close = np.roll(close_arr, 1)
    prev_close[0] = close_arr[0]
    
    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - prev_close)
    tr3 = np.abs(low_arr - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=atr_period).mean().values
    
    nloss = key_value * atr
    xatr = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if np.isnan(nloss[i]): 
            continue
            
        prev_xatr = xatr[i-1]
        curr_src = src[i]
        prev_src = src[i-1]
        curr_nloss = nloss[i]

        if (prev_src > prev_xatr) and (curr_src > prev_xatr):
            xatr[i] = max(prev_xatr, curr_src - curr_nloss)
        elif (prev_src < prev_xatr) and (curr_src < prev_xatr):
            xatr[i] = min(prev_xatr, curr_src + curr_nloss)
        elif curr_src > prev_xatr:
            xatr[i] = curr_src - curr_nloss
        else:
            xatr[i] = curr_src + curr_nloss

    ema_signal = np.where(src > xatr, 1, -1)
    
    last_signal = ema_signal[-1]
    prev_signal = ema_signal[-2]
    
    is_fresh_buy = (last_signal == 1) and (prev_signal == -1)
    is_fresh_sell = (last_signal == -1) and (prev_signal == 1)
    
    return bool(is_fresh_buy), bool(is_fresh_sell)

# ======================================================
# 5. MAIN SCANNER LOGIC (ULTRA SELECTIVE)
# ======================================================
def fetch_and_analyze(ticker: str, settings: List[Tuple]) -> List[Dict]:
    """Mengambil data 1 saham dan mengecek semua setting."""
    detected_signals = []
    
    try:
        symbol = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
        
        df = yf.download(
            symbol, 
            period="1y", 
            interval="1d", 
            progress=False, 
            auto_adjust=False,
            timeout=10
        )
        
        if df.empty or len(df) < EMA_TREND_PERIOD:
            return []

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # --------------------------------------------------
        # FILTER 1: LIQUIDITY
        # --------------------------------------------------
        df['TxValue'] = df['Close'] * df['Volume']
        mean_value = df['TxValue'].rolling(20).mean().iloc[-1]
        
        if mean_value < MIN_DAILY_VALUE:
            return []

        # --------------------------------------------------
        # FILTER 2: TREND (EMA 200)
        # --------------------------------------------------
        ema200 = calculate_ema(df['Close'], EMA_TREND_PERIOD).iloc[-1]
        current_close = df['Close'].iloc[-1]
        
        if current_close < ema200:
            return []

        # 🆕 FILTER 2B: Distance from EMA200 (Jangan terlalu jauh)
        distance_pct = ((current_close - ema200) / ema200) * 100
        if distance_pct > MAX_DISTANCE_FROM_EMA200:
            return []  # Skip overextended stocks

        # --------------------------------------------------
        # FILTER 3: RSI (Range Optimal)
        # --------------------------------------------------
        rsi_val = calculate_rsi(df['Close'], RSI_PERIOD)
        if rsi_val <= RSI_BUY_THRESHOLD or rsi_val >= RSI_MAX_THRESHOLD:
            return []  # Skip oversold DAN overbought

        # --------------------------------------------------
        # 🆕 FILTER 4: VOLUME SPIKE
        # --------------------------------------------------
        volume_avg = df['Volume'].rolling(20).mean().iloc[-1]
        volume_today = df['Volume'].iloc[-1]
        volume_ratio = volume_today / volume_avg if volume_avg > 0 else 0
        
        if volume_ratio < MIN_VOLUME_RATIO:
            return []  # Volume harus spike (minat beli kuat)

        # --------------------------------------------------
        # 🆕 FILTER 5: ADX (TREND STRENGTH)
        # --------------------------------------------------
        adx_val = calculate_adx(df, period=14)
        if adx_val < MIN_ADX_VALUE:
            return []  # Trend terlalu lemah/sideways

        # --------------------------------------------------
        # 🆕 FILTER 6: PRICE MOMENTUM (ROC)
        # --------------------------------------------------
        roc_val = calculate_roc(df['Close'], period=5)
        if roc_val < MIN_PRICE_ROC:
            return []  # Momentum naik harus kuat

        # --------------------------------------------------
        # CHECK UT BOT SIGNALS (Fresh Buy Only)
        # --------------------------------------------------
        for name, key_val, atr_per in settings:
            fresh_buy, _ = calculate_ut_bot(df, key_val, atr_per)
            
            if fresh_buy:
                detected_signals.append({
                    "ticker": ticker,
                    "setting": name,
                    "price": current_close,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "vol_ratio": volume_ratio,
                    "roc_5d": roc_val,
                    "ema_dist": distance_pct,
                    "value_avg_bn": mean_value / 1_000_000_000
                })

    except Exception as e:
        pass
        
    return detected_signals

def main():
    print("\n" + "="*60)
    print("🚀 UT BOT PRO v4.0 - ULTRA SELECTIVE MODE")
    print(f"⚙️  6 Filters Active:")
    print(f"   • EMA200 Trend + Max Distance {MAX_DISTANCE_FROM_EMA200}%")
    print(f"   • RSI {RSI_BUY_THRESHOLD}-{RSI_MAX_THRESHOLD}")
    print(f"   • Volume Spike >{MIN_VOLUME_RATIO}x")
    print(f"   • ADX >{MIN_ADX_VALUE}")
    print(f"   • ROC 5D >{MIN_PRICE_ROC}%")
    print(f"   • Liquidity >{MIN_DAILY_VALUE/1e9}M")
    print("="*60)

    start_time = time.time()
    tickers = load_bei_tickers()
    
    bot_settings = [
        ("AGRESIF", 1.0, 10),
        ("STANDAR", 2.0, 10),
        ("KONSERVATIF", 3.0, 14)
    ]
    
    results = {s[0]: [] for s in bot_settings}
    
    print(f"🔍 Scanning {len(tickers)} saham...")
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"   Processed {i}/{len(tickers)}...")
            
        signals = fetch_and_analyze(ticker, bot_settings)
        
        for sig in signals:
            group = sig['setting']
            results[group].append(sig)
            print(f"   🔥 FOUND: {sig['ticker']} ({group}) | RSI:{sig['rsi']:.0f} ADX:{sig['adx']:.0f} VOL:{sig['vol_ratio']:.1f}x")

    # ======================================================
    # REPORTING (Enhanced with new metrics)
    # ======================================================
    duration = (time.time() - start_time) / 60
    
    report = f"🤖 *UT BOT PRO v4.0 - ULTRA SELECTIVE*\n"
    report += f"📅 {dt.datetime.now().strftime('%d-%m-%Y %H:%M')} WIB\n"
    report += f"⏱️ Scan Time: {duration:.1f} min\n"
    report += f"🛡️ *6 Filters Active* (EMA200 + RSI + VOL + ADX + ROC + LIQ)\n\n"
    
    total_found = 0
    for name, data in results.items():
        if not data:
            continue
            
        total_found += len(data)
        report += f"📌 *MODE {name}* ({len(data)} saham)\n"
        
        # Sort by ADX (trend strength) descending
        data.sort(key=lambda x: x['adx'], reverse=True)
        
        for s in data:
            report += f"• `{s['ticker']}`: {s['price']:.0f}\n"
            report += f"  RSI:{s['rsi']:.0f} | ADX:{s['adx']:.0f} | VOL:{s['vol_ratio']:.1f}x | ROC:{s['roc_5d']:.1f}%\n"
        report += "\n"
        
    if total_found == 0:
        report += "ℹ️ Tidak ada saham yang lolos 6 filter ketat hari ini.\n"
        report += "💡 Coba turunkan MIN_VOLUME_RATIO atau MIN_ADX_VALUE di config."
    else:
        report += f"💎 *{total_found} saham premium* yang lolos screening ultra ketat.\n"
        report += "🎯 Saham-saham ini memiliki kombinasi: Uptrend + Strong Momentum + Volume Spike."

    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    send_telegram_message(report)

if __name__ == "__main__":
    main()
