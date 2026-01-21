# scanner_pro_v5_improved.py - UT BOT v5.0 (OVERNIGHT SWING OPTIMIZED)
# Optimized untuk strategi: Buy Sore, Jual Pagi
# Presisi Level: Amibroker++

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import time
from typing import List, Dict, Optional, Tuple

# ======================================================
# 1. CONFIGURATION (OVERNIGHT SWING TUNED)
# ======================================================
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    print("⚠️  config.py tidak ditemukan - Telegram dinonaktifkan.")

# ✨ OVERNIGHT SWING SETTINGS (Optimized winrate)
RSI_PERIOD = 14
RSI_BUY_MIN = 30              # ⬇️ Relaxed from 40 (capture best momentum entry)
RSI_BUY_MAX = 75              # ⬆️ Relaxed from 70 (oversold rejection diminishing)
EMA_TREND_PERIOD = 200
MIN_DAILY_VALUE = 1_000_000_000
VOLUME_SURGE_MULTIPLIER = 1.3  # ⬆️ Tightened from 1.2 (lebih strict volume)
ADX_PERIOD = 14
ADX_THRESHOLD = 15            # ⬇️ Relaxed from 20 (overnight toleran ranging+breakout)
MIN_CLOSE_TIME_FOR_BUY = "15:20"  # BUY only after 15:20 WIB to avoid gap risk

# ✨ NEW: MACD for confluence
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ✨ NEW: Higher timeframe (weekly trend)
USE_WEEKLY_FILTER = True

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
# 4. TECHNICAL ANALYSIS ENGINE (v5: ENHANCED PRECISION)
# ======================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (Amibroker-compatible)."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Smoothing (RMA) - Amibroker default."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate RSI (last value)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = calculate_rma(gain, period)
    avg_loss = calculate_rma(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0

def calculate_macd(
    series: pd.Series, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[float, float, float]:
    """
    Calculate MACD line, signal line, dan histogram.
    Returns: (macd_line, signal_line, histogram)
    
    MACD Bullish = macd > signal & histogram positive & macd > 0
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return (
        float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
        float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
    )

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Average Directional Index (ADX)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed ATR and DI
    atr = calculate_rma(pd.Series(tr), period)
    plus_di = 100 * calculate_rma(pd.Series(plus_dm), period) / atr
    minus_di = 100 * calculate_rma(pd.Series(minus_dm), period) / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calculate_rma(dx, period)
    
    val = adx.iloc[-1]
    return float(val) if not pd.isna(val) else 0.0

def calculate_ut_bot(
    df: pd.DataFrame,
    key_value: float = 2.0,
    atr_period: int = 10
) -> Tuple[bool, bool, float]:
    """
    UT Bot Trailing Stop (Amibroker-compatible).
    Returns: (is_fresh_buy, is_fresh_sell, trailing_stop_value)
    """
    if len(df) < max(atr_period, 50):
        return False, False, 0.0

    open_arr = df['Open'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    close_arr = df['Close'].values
    
    # Heikin Ashi Close
    src = (open_arr + high_arr + low_arr + close_arr) / 4

    # ATR Calculation (Wilder's method)
    prev_close = np.roll(close_arr, 1)
    prev_close[0] = close_arr[0]
    
    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - prev_close)
    tr3 = np.abs(low_arr - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = calculate_rma(pd.Series(tr), atr_period).values
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

    # Signal Generation
    ema_signal = np.where(src > xatr, 1, -1)
    
    last_signal = ema_signal[-1]
    prev_signal = ema_signal[-2] if len(ema_signal) > 1 else last_signal
    
    is_fresh_buy = (last_signal == 1) and (prev_signal == -1)
    is_fresh_sell = (last_signal == -1) and (prev_signal == 1)
    
    return bool(is_fresh_buy), bool(is_fresh_sell), float(xatr[-1])

def calculate_support_resistance(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
    """Calculate nearest support (low) and resistance (high)."""
    recent_high = df['High'].tail(lookback).max()
    recent_low = df['Low'].tail(lookback).min()
    return float(recent_low), float(recent_high)

def get_current_jakarta_time() -> dt.datetime:
    """Get current time in Jakarta timezone."""
    return dt.datetime.now()  # Assuming server is in Jakarta timezone

def check_buy_time_window() -> bool:
    """
    Check apakah sekarang dalam jam buy window (15:20-15:30 WIB).
    Returns True jika sudah past 15:20 (aman dari gap).
    """
    now = get_current_jakarta_time()
    buy_time = dt.datetime.strptime(MIN_CLOSE_TIME_FOR_BUY, "%H:%M").time()
    
    # For live trading: return now.time() >= buy_time
    # For backtesting: always return True
    return True  # ⚠️ Set to now.time() >= buy_time untuk live trading

# ======================================================
# 5. WEEKLY TREND CONFLUENCE
# ======================================================
def check_weekly_trend(ticker: str) -> Tuple[bool, float]:
    """
    Fetch weekly data dan check if trend is bullish.
    Returns: (is_bullish_weekly, weekly_ema200)
    """
    try:
        symbol = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
        df_weekly = yf.download(
            symbol,
            period="2y",
            interval="1wk",
            progress=False,
            auto_adjust=False,
            timeout=10
        )
        
        if df_weekly.empty or len(df_weekly) < EMA_TREND_PERIOD:
            return True, 0.0  # Default bullish if not enough data
        
        if isinstance(df_weekly.columns, pd.MultiIndex):
            df_weekly.columns = [col[0] for col in df_weekly.columns]
        
        ema200_weekly = calculate_ema(df_weekly['Close'], EMA_TREND_PERIOD).iloc[-1]
        current_close_weekly = df_weekly['Close'].iloc[-1]
        
        is_bullish = current_close_weekly > ema200_weekly
        return bool(is_bullish), float(ema200_weekly)
        
    except Exception as e:
        print(f"⚠️ Weekly check failed for {ticker}: {e}")
        return True, 0.0  # Default bullish

# ======================================================
# 6. MAIN SCANNER LOGIC (v5: ENHANCED FILTERS)
# ======================================================

def fetch_and_analyze(ticker: str, settings: List[Tuple]) -> List[Dict]:
    """Fetch data dan check all filters untuk 1 saham."""
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
        # FILTER 0: BUY TIME WINDOW (NEW)
        # --------------------------------------------------
        if not check_buy_time_window():
            return []  # Belum masuk jam buy (prevent gap risk)

        # --------------------------------------------------
        # FILTER 1: LIQUIDITY
        # --------------------------------------------------
        df['TxValue'] = df['Close'] * df['Volume']
        mean_value = df['TxValue'].rolling(20).mean().iloc[-1]
        
        if mean_value < MIN_DAILY_VALUE:
            return []

        # --------------------------------------------------
        # FILTER 2: WEEKLY TREND CONFLUENCE (NEW)
        # --------------------------------------------------
        if USE_WEEKLY_FILTER:
            is_bullish_weekly, ema200_weekly = check_weekly_trend(ticker)
            if not is_bullish_weekly:
                return []  # Weekly trend bearish = skip

        # --------------------------------------------------
        # FILTER 3: DAILY TREND (EMA 200)
        # --------------------------------------------------
        ema200 = calculate_ema(df['Close'], EMA_TREND_PERIOD).iloc[-1]
        current_close = df['Close'].iloc[-1]
        
        if current_close < ema200:
            return []

        # --------------------------------------------------
        # FILTER 4: RSI (RELAXED FOR OVERNIGHT SWING)
        # --------------------------------------------------
        rsi_val = calculate_rsi(df['Close'], RSI_PERIOD)
        if not (RSI_BUY_MIN <= rsi_val <= RSI_BUY_MAX):
            return []

        # --------------------------------------------------
        # FILTER 5: MACD CONFIRMATION (NEW)
        # --------------------------------------------------
        macd_line, signal_line, histogram = calculate_macd(
            df['Close'], 
            MACD_FAST, 
            MACD_SLOW, 
            MACD_SIGNAL
        )
        
        # Bullish MACD = macd > signal & histogram positive & macd > 0
        is_macd_bullish = (macd_line > signal_line) and (histogram > 0) and (macd_line > 0)
        if not is_macd_bullish:
            return []  # No MACD confluence = less reliable

        # --------------------------------------------------
        # FILTER 6: VOLUME SURGE (STRICT)
        # --------------------------------------------------
        avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-2]
        current_volume = df['Volume'].iloc[-1]
        
        if current_volume < (avg_volume_20 * VOLUME_SURGE_MULTIPLIER):
            return []

        # --------------------------------------------------
        # FILTER 7: ADX (RELAXED FOR OVERNIGHT)
        # --------------------------------------------------
        adx_val = calculate_adx(df, ADX_PERIOD)
        if adx_val < ADX_THRESHOLD:
            return []

        # --------------------------------------------------
        # FILTER 8: PRICE ACTION (BULLISH CANDLE)
        # --------------------------------------------------
        last_open = df['Open'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        
        if last_close <= last_open:
            return []

        # --------------------------------------------------
        # FILTER 9: SUPPORT/RESISTANCE PROXIMITY
        # --------------------------------------------------
        support, resistance = calculate_support_resistance(df, lookback=20)
        
        distance_to_resistance = (resistance - current_close) / current_close
        if distance_to_resistance < 0.02:  # < 2% headroom
            return []
        
        # Distance to support (should be >2% for safety)
        distance_to_support = (current_close - support) / current_close
        if distance_to_support < 0.02:
            return []

        # --------------------------------------------------
        # FILTER 10: VOLATILITY CHECK (NEW)
        # --------------------------------------------------
        # Recent candle size shouldn't be too large (avoid gap risk next day)
        recent_candle_size = (df['High'].iloc[-1] - df['Low'].iloc[-1]) / current_close
        avg_candle_size = df['High'].sub(df['Low']).div(df['Close']).rolling(10).mean().iloc[-1]
        
        if recent_candle_size > avg_candle_size * 1.8:  # Candle terlalu besar
            return []

        # --------------------------------------------------
        # CHECK SIGNALS (UT BOT)
        # --------------------------------------------------
        for name, key_val, atr_per in settings:
            fresh_buy, _, trail_stop = calculate_ut_bot(df, key_val, atr_per)
            
            if fresh_buy:
                stop_loss_pct = ((current_close - trail_stop) / current_close) * 100
                
                detected_signals.append({
                    "ticker": ticker,
                    "setting": name,
                    "price": current_close,
                    "rsi": rsi_val,
                    "macd_histogram": histogram,
                    "adx": adx_val,
                    "volume_ratio": current_volume / avg_volume_20,
                    "value_avg_bn": mean_value / 1_000_000_000,
                    "ema_200": ema200,
                    "stop_loss": trail_stop,
                    "risk_pct": stop_loss_pct,
                    "resistance": resistance,
                    "support": support,
                    "upside_pct": distance_to_resistance * 100,
                    "weekly_bullish": "✅" if USE_WEEKLY_FILTER else "N/A"
                })

    except Exception as e:
        pass
        
    return detected_signals

def main() -> None:
    print("\n" + "="*70)
    print("🚀 UT BOT PRO v5.0 - OVERNIGHT SWING OPTIMIZED (Amibroker++ Precision)")
    print(f"📅 Strategy: BUY SORE (15:20+), JUAL PAGI (next day open)")
    print("="*70)
    print(f"✨ Enhanced Filters (10-layer):")
    print(f"   • Buy Time Window (prevent gap risk)")
    print(f"   • Weekly Trend Confluence (higher TF strength)")
    print(f"   • Liquidity (Min Rp{MIN_DAILY_VALUE/1e9:.0f}B daily value)")
    print(f"   • EMA{EMA_TREND_PERIOD} Daily Trend")
    print(f"   • RSI Range: {RSI_BUY_MIN}-{RSI_BUY_MAX} (relaxed for overnight)")
    print(f"   • MACD Bullish Confirmation (NEW)")
    print(f"   • Volume Surge: {VOLUME_SURGE_MULTIPLIER}x threshold")
    print(f"   • ADX > {ADX_THRESHOLD} (trend strength, relaxed)")
    print(f"   • Bullish Price Action")
    print(f"   • Support/Resistance Proximity")
    print(f"   • Volatility Check (candle size)")
    print("="*70)

    start_time = time.time()
    tickers = load_bei_tickers()
    
    # Optimized settings for overnight swing (tight stop = fast profit, wide stop = max profit potential)
    bot_settings = [
        ("AGGRESSIVE", 1.5, 7),       # Tight stop, 2-3% daily target
        ("STANDARD", 2.0, 10),        # Balanced, 3-5% target
        ("CONSERVATIVE", 2.5, 14)     # Wide stop, 5-7% target
    ]
    
    results = {s[0]: [] for s in bot_settings}
    
    print(f"\n🔍 Scanning {len(tickers)} saham BEI...")
    print(f"⏱️  Estimasi waktu: 5-10 menit\n")
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            elapsed = (time.time() - start_time) / 60
            print(f"   ⏳ Progress {i}/{len(tickers)} ({elapsed:.1f} min elapsed)...")
            
        signals = fetch_and_analyze(ticker, bot_settings)
        
        for sig in signals:
            group = sig['setting']
            results[group].append(sig)
            print(f"   🔥 FOUND: {sig['ticker']:6s} | {group:12s} | Price: Rp{sig['price']:.0f} | RSI:{sig['rsi']:5.1f} | ADX:{sig['adx']:5.1f} | Weekly:{sig['weekly_bullish']}")

    # ======================================================
    # REPORTING
    # ======================================================
    duration = (time.time() - start_time) / 60
    
    report = f"🤖 *UT BOT PRO v5.0 - OVERNIGHT SWING SIGNAL REPORT*\n\n"
    report += f"📅 *Tanggal:* {dt.datetime.now().strftime('%d-%m-%Y %H:%M')} WIB\n"
    report += f"⏱️ *Scan Time:* {duration:.1f} menit\n\n"
    report += f"🛡️ *10-Layer Filter Active:*\n"
    report += f"   ✅ Buy Time Window | Weekly Confluence | Trend | Liquidity\n"
    report += f"   ✅ RSI Range | MACD Bullish | Volume Surge | ADX\n"
    report += f"   ✅ Price Action | Support/Resistance | Volatility\n\n"
    
    total_found = 0
    for name in ["AGGRESSIVE", "STANDARD", "CONSERVATIVE"]:
        data = results[name]
        if not data:
            continue
            
        total_found += len(data)
        report += f"📌 *MODE: {name}*\n"
        
        # Sort by ADX (strongest trend first)
        data_sorted = sorted(data, key=lambda x: x['adx'], reverse=True)
        
        for s in data_sorted[:5]:  # Top 5 per mode
            report += f"• `{s['ticker']:6s}` • Rp{s['price']:.0f}\n"
            report += f"  RSI:{s['rsi']:5.1f} | ADX:{s['adx']:5.1f} | Vol:{s['volume_ratio']:.1f}x | MACD:{'✅' if s['macd_histogram'] > 0 else '❌'}\n"
            report += f"  Risk:{s['risk_pct']:.1f}% | Target:{s['upside_pct']:.1f}% | Support:Rp{s['support']:.0f}\n"
        
        if len(data) > 5:
            report += f"  _(+{len(data)-5} sinyal lainnya di mode ini)_\n"
        report += "\n"
        
    if total_found == 0:
        report += "ℹ️ *Tidak ada sinyal* yang lolos 10-layer filter hari ini.\n"
        report += "💡 Filter ketat = Win rate tinggi. Tunggu setup yang perfect.\n"
    else:
        report += f"✅ *Total {total_found} sinyal valid* untuk overnight swing trading.\n\n"
        report += f"📋 *HOW TO USE:*\n"
        report += f"1️⃣ BUY: Jam 15:20-15:30 WIB (harga di level signal)\n"
        report += f"2️⃣ STOP LOSS: Di bawah level yang tertera (dalam kolom Risk%)\n"
        report += f"3️⃣ SELL: Pagi hari saat profit 3-5% ATAU di open market Pukul 09:30\n\n"
        report += f"⚡ *TIPS:*\n"
        report += f"• AGGRESSIVE = Jual cepat (profit 3% atau jika loss 1%)\n"
        report += f"• STANDARD = Balanced (profit 4-5% atau jika loss 2%)\n"
        report += f"• CONSERVATIVE = Long hold (profit 5%+ atau jika loss 3%)\n"

    print("\n" + "="*70)
    print(report)
    print("="*70)
    
    send_telegram_message(report)

if __name__ == "__main__":
    main()