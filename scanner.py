# INSTRUKSI OPERASIONAL: THE RANGKUTI PROTOCOL (REFINED)
# SUBJECT: MODE B - SCALPING "BSJP" (High Volatility Focus)
# REFERENCE: Metodologi Indrawijaya Rangkuti (Place & Time Synchronized)

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import warnings
import os
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 1. CONFIG & TELEGRAM SETUP
# ======================================================
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    print("⚠️  config.py tidak ditemukan. Notifikasi Telegram hanya print layar.")

JAKARTA_TZ = dt.timezone(dt.timedelta(hours=7))

# --- RANGKUTI PARAMETERS (STRICT) ---
MIN_PRICE = 60
MIN_TURNOVER_BN = 5.0       # Likuiditas minimal 5 Milyar
RSI_SWEET_SPOT = 40.0       # [cite: 118] RSI > 40 adalah Sweet Spot awal tren
RSI_MAX_ENTRY = 75.0        # Hindari buying climax [cite: 15]

# Parameter Scalping / Volatilitas 
MIN_BETA_VOLATILITY = 3.0   # Persentase rata-rata range harian minimal (High Beta)
MIN_VOLUME_RATIO = 1.2      # Validasi "Mood" pasar

# ======================================================
# 2. DATA HELPERS
# ======================================================
def fix_yfinance_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty: return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in req_cols:
        if c not in data.columns: return pd.DataFrame()
    return data[req_cols].copy()

def fetch_data(ticker: str, period="6mo") -> pd.DataFrame:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") and not ticker.startswith("^") else ticker
        # Fetch data lebih panjang untuk perhitungan ATR & MA stabil
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        return fix_yfinance_data(df)
    except:
        return pd.DataFrame()

def load_tickers() -> List[str]:
    possible_paths = ["data.csv", "data/data.csv"]
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path:
        try:
            print(f"📂 Membaca watchlist dari: {file_path}")
            df = pd.read_csv(file_path, header=None)
            tickers = df.iloc[:, 0].dropna().astype(str).str.upper().tolist()
            if len(tickers) > 0 and tickers[0] in ["SYMBOL", "KODE", "TICKER"]:
                tickers = tickers[1:]
            return tickers
        except Exception:
            return []
    else:
        # Fallback List (Saham High Beta & Liquid)
        return ["BRIS", "ANTM", "MEDC", "PGEO", "ADRO", "BBRI", "BBCA", "GOTO", "TLKM", "ASII"]

def send_telegram(text: str):
    if not TELEGRAM_OK: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except:
        pass

# ======================================================
# 3. TECHNICAL INDICATORS (THE TOOLKIT)
# ======================================================
def calc_indicators(df: pd.DataFrame):
    # 1. RSI (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD (Trend & Trigger) [cite: 112]
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

    # 3. Bollinger Bands (Place & Volatility) [cite: 122]
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['SMA20'] - (df['STD20'] * 2)
    
    # Bandwidth untuk mendeteksi Squeeze vs Expansion 
    df['BB_Width'] = ((df['Upper_Band'] - df['Lower_Band']) / df['SMA20']) * 100

    # 4. Stochastic (Contextual) 
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(3).mean()

    # 5. Volume & Volatility (ATR)
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    # True Range for Beta/Volatility measurement
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    # Volatility Percentage (Beta Proxy)
    df['Daily_Vol_Pct'] = (df['TR'] / df['Close']) * 100
    df['Avg_Vol_Pct'] = df['Daily_Vol_Pct'].rolling(window=10).mean()

    return df

# ======================================================
# 4. MARKET MOOD ANALYSIS
# ======================================================
def check_market_condition():
    print("🌍 Menganalisis Market Mood (IHSG)...")
    df = fetch_data("^JKSE", period="10d")
    if df.empty: return "NEUTRAL"
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change_pct = ((last['Close'] - prev['Close']) / prev['Close']) * 100
    
    # [cite: 11] Market Crash Threshold
    if change_pct < -1.0: return "DANGER"
    if change_pct < 0: return "CAUTION"
    return "SAFE"

# ======================================================
# 5. THE RANGKUTI PROTOCOL LOGIC
# ======================================================
def analyze_bsjp(ticker: str, market_status: str) -> Optional[Dict]:
    try:
        df = fetch_data(ticker)
        if df.empty or len(df) < 50: return None
        
        df = calc_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 1. FILTER LIKUIDITAS (BASIC) ---
        if last['Close'] < MIN_PRICE: return None
        turnover_bn = (last['Close'] * last['Volume']) / 1_000_000_000
        # Di market DANGER, butuh likuiditas lebih besar agar tidak terjebak
        min_turnover = MIN_TURNOVER_BN * 1.5 if market_status == "DANGER" else MIN_TURNOVER_BN
        if turnover_bn < min_turnover: return None

        # --- 2. FILTER DIMENSI WAKTU & VOLATILITAS (TIME) ---
        #  Scalping butuh saham Beta tinggi, saham tidur tidak berguna.
        if last['Avg_Vol_Pct'] < MIN_BETA_VOLATILITY: return None
        
        # Cek Strong Close (BSJP): Harga penutupan dekat dengan High
        dist_from_high = (last['High'] - last['Close']) / last['High']
        if dist_from_high > 0.03: return None # Toleransi 3% dari pucuk

        # --- 3. FILTER DIMENSI MOMENTUM (THE INDRAWIJAYA RULE) ---
        #  Formula: MACD Positive + RSI > 40
        
        # Rule A: RSI Filter
        if last['RSI'] < RSI_SWEET_SPOT: return None # Belum masuk zona sweet spot
        if last['RSI'] > RSI_MAX_ENTRY: return None  # Terlalu panas (Risk Reversal)

        # Rule B: MACD Momentum
        # Kita cari yang MACD Line > Signal (Bullish) ATAU baru saja Golden Cross
        macd_bullish = last['MACD_Line'] > last['MACD_Signal']
        if not macd_bullish: return None

        # --- 4. FILTER DIMENSI MOOD (VOLUME) ---
        #  Volume never lies. Harus ada ledakan volume untuk scalping.
        if last['Vol_MA20'] == 0: return None
        vol_ratio = last['Volume'] / last['Vol_MA20']
        
        # Jika harga naik tapi volume drop (Divergence Bearish Exhaustion), skip
        # Kecuali ini scalping breakout, kita butuh volume besar.
        if vol_ratio < MIN_VOLUME_RATIO: return None

        # --- 5. FILTER DIMENSI TEMPAT (PLACE/STRUCTURE) ---
        #  Expansion vs Squeeze
        is_expanding = last['BB_Width'] > df['BB_Width'].iloc[-5:].mean()
        
        setup_type = ""
        score = 50 # Baseline Score

        # SETUP A: Expansion Breakout (High Risk High Reward)
        if last['Close'] > last['Upper_Band'] and vol_ratio > 2.0 and is_expanding:
            setup_type = "🚀 Expansion Breakout"
            score += 30
        
        # SETUP B: Support Rebound (Range Scalping) 
        # Harga mantul dari support/MA20 dengan volume
        elif last['Close'] > last['SMA20'] and prev['Close'] < prev['SMA20']:
             setup_type = "🛡️ Mid-Band Rebound"
             score += 20
        
        # SETUP C: Stochastic Momentum (Trend Following) 
        # Stochastic naik tapi belum overbought parah, atau 'menempel' di atas (Super Bull)
        elif last['%K'] > last['%D'] and last['%K'] < 80:
             setup_type = "📈 Momentum Push"
             score += 10
        
        else:
            # Jika tidak memenuhi setup spesifik, cek ulang kekuatan candle
            if last['Close'] > prev['Close'] and vol_ratio > 1.5:
                setup_type = "⚡ Volume Spike"
                score += 10
            else:
                return None

        # [cite: 21] DANGER Mode Adjustment
        if market_status == "DANGER":
            score -= 20 # Persulit lolos screening
            if setup_type == "🚀 Expansion Breakout": return None # Hindari breakout saat market crash

        if score < 60: return None

        # Hitung Stop Loss Struktural (ATR Based) [cite: 175]
        # Stop loss bukan nominal, tapi berdasarkan volatilitas
        stop_loss_price = int(last['Close'] - (1.5 * last['ATR']))

        return {
            "ticker": ticker,
            "close": int(last['Close']),
            "change": round(((last['Close'] - prev['Close']) / prev['Close']) * 100, 2),
            "vol_ratio": round(vol_ratio, 1),
            "rsi": int(last['RSI']),
            "atr_sl": stop_loss_price,
            "beta": round(last['Avg_Vol_Pct'], 1),
            "setup": setup_type,
            "score": score
        }

    except Exception as e:
        return None

# ======================================================
# 6. MAIN EXECUTION
# ======================================================
def main():
    print("="*70)
    print("🦅 THE RANGKUTI PROTOCOL: BSJP (SCALPING MODE)")
    print("   Metodologi: Place & Time Synchronized | Logic: RSI 40 + High Beta")
    print("="*70)
    
    market_status = check_market_condition()
    print(f"📊 Market Mood: {market_status}\n")
    
    tickers = load_tickers()
    results = []
    
    print(f"🔍 Scanning {len(tickers)} saham...")
    for i, ticker in enumerate(tickers):
        print(f"   Scanning {ticker}...", end="\r")
        res = analyze_bsjp(ticker, market_status)
        if res:
            results.append(res)
    
    # Sort by Score (Priority)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # --- MESSAGE BUILDER ---
    msg = f"🦅 *THE RANGKUTI PROTOCOL: BSJP*\n"
    msg += f"📅 {dt.datetime.now(JAKARTA_TZ).strftime('%d/%m %H:%M WIB')}\n"
    msg += f"📊 Market Mood: *{market_status}*\n"
    msg += "⚙️ _Filter: RSI>40, High Beta, Vol Explosion_\n\n"
    
    if not results:
        msg += "⛔ *NO SIGNALS DETECTED*\n"
        msg += "_Pasar tidak sinkron (Time/Place/Mood mismatch)._\n"
        msg += "_Cash is King. Wait & See._"
    else:
        msg += "🎯 *TOP WATCHLIST (High Probability):*\n"
        for r in results[:5]: # Top 5 only
            icon = "🔥" if r['score'] >= 80 else "⚡"
            msg += f"{icon} *{r['ticker']}* (+{r['change']}%)\n"
            msg += f"   Price: {r['close']} | SL Struktural: {r['atr_sl']}\n"
            msg += f"   Vol: {r['vol_ratio']}x | RSI: {r['rsi']} | Beta: {r['beta']}%\n"
            msg += f"   🔮 {r['setup']}\n\n"
            
    print("\n" + msg)
    send_telegram(msg)

if __name__ == "__main__":
    main()