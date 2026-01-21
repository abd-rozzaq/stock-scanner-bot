# scanner_v6_1_corrected.py
# UT BOT SCANNER v6.1 (CORRECTED - FAITHFUL TO PANDUAN)
# STRATEGI: "Akumulasi Pelan Naik Konstan" - STRICT ADHERENCE
# FOKUS: Early Entry pada QUIET ACCUMULATION tanpa relax kriteria

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import time
import warnings
from typing import Dict, Optional, Tuple, List

warnings.filterwarnings("ignore")

# ======================================================
# 1. TELEGRAM CONFIG
# ======================================================
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except:
    TELEGRAM_OK = False
    print("⚠️  config.py belum lengkap - Telegram skip")

# ======================================================
# 2. GLOBAL SETTINGS (STRICT - SESUAI PANDUAN)
# ======================================================
JAKARTA_TZ = dt.timezone(dt.timedelta(hours=7))

# === QUIET ACCUMULATION FILTERS (STRICT!) ===
MAX_DAILY_GAIN = 3.0                # STRICT: < 3% (jangan relax!)
MAX_ATR_PERCENT = 3.0               # STRICT: < 3% (tenang, tidak liar)
MIN_HAJAR_KANAN_PCT = 50.0          # > 50% pembeli dominan
MIN_MONEY_FLOW_SINGLE = 100_000_000 # >= 100jt per hari

# Volume Pattern Rules (STRICT!)
MAX_VOL_RATIO = 2.0                 # STRICT: < 2x MA20 (kunci "senyap")
MIN_VOL_RATIO = 1.0                 # > 1x MA20 (mulai ramai)

# Trend & Momentum Thresholds
EMA5_PERIOD = 5
EMA10_PERIOD = 10
EMA50_PERIOD = 50

# Liquidity
MIN_TURNOVER = 1.0                  # >= 1 Milyar

# ======================================================
# 3. DATA CLEANUP & FETCH
# ======================================================
def fix_yfinance_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fix MultiIndex columns dari yfinance"""
    if data.empty:
        return data
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in req_cols:
        if c not in data.columns:
            return pd.DataFrame()
    
    data = data[req_cols].copy()
    data["Close"] = data["Close"].fillna(method="ffill")
    data["Volume"] = data["Volume"].fillna(0)
    
    return data

def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch data dari yfinance dengan retry logic"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            df = yf.download(
                f"{ticker}.JK",
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False
            )
            return fix_yfinance_data(df)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                return pd.DataFrame()

# ======================================================
# 4. CORE INDICATORS (AMIBROKER PRECISION)
# ======================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA dengan proper NaN handling"""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def calculate_atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Calculate ATR% dengan Amibroker precision"""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period, min_periods=period).mean()
    atr_pct = (atr / close) * 100
    
    return float(atr_pct.dropna().iloc[-1]) if len(atr_pct.dropna()) > 0 else 0.0

def calculate_daily_gain(df: pd.DataFrame) -> float:
    """Hitung % gain hari ini (Close vs Open) - SESUAI PANDUAN"""
    if len(df) < 1:
        return 0.0
    
    open_price = df["Open"].iloc[-1]
    close_price = df["Close"].iloc[-1]
    
    if open_price == 0:
        return 0.0
    
    gain = ((close_price - open_price) / open_price) * 100
    return max(gain, 0.0)

def calculate_hajar_kanan_improved(df: pd.DataFrame, lookback: int = 5) -> float:
    """
    Hitung % pembeli HAJAR KANAN (beli di harga offer)
    
    Gunakan: Close vs Open sebagai proxy
    - Close > Open = Pembeli menang (HAJAR KANAN)
    - Close < Open = Penjual menang
    
    Return: 0-100 (persentase bullish candles)
    """
    if len(df) < lookback:
        return 50.0
    
    df_recent = df.tail(lookback).copy()
    bullish_bars = (df_recent["Close"] > df_recent["Open"]).sum()
    total_bars = len(df_recent)
    
    haka_pct = (bullish_bars / total_bars) * 100
    return float(haka_pct)

def calculate_money_flow_single(df: pd.DataFrame) -> float:
    """
    Hitung Money Flow (Arus Uang) untuk hari ini
    Money Flow = Close * Volume
    
    Return: dalam Rupiah
    """
    if len(df) < 1:
        return 0.0
    
    last_close = float(df["Close"].iloc[-1])
    last_vol = float(df["Volume"].iloc[-1])
    
    money_flow = last_close * last_vol
    return money_flow

def calculate_volume_pattern(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Hitung Volume Pattern untuk deteksi Quiet Accumulation
    Return: (vol_ratio, vol_ma20)
    """
    if len(df) < 20:
        return 1.0, 0.0
    
    last_vol = float(df["Volume"].iloc[-1])
    vol_ma20 = float(df["Volume"].rolling(20, min_periods=20).mean().iloc[-1])
    
    if vol_ma20 == 0:
        return 1.0, 0.0
    
    vol_ratio = last_vol / vol_ma20
    return vol_ratio, vol_ma20

# ======================================================
# 5. FRESH SIGNAL DETECTION (KEY IMPROVEMENT!)
# ======================================================

def check_conditions_history(df: pd.DataFrame, lookback_days: int = 3) -> List[bool]:
    """
    Check apakah kondisi quiet accumulation muncul dalam N hari terakhir
    
    Return: List[bool] untuk setiap hari (paling baru di akhir list)
    Logic: Jika hari ini TRUE tapi kemarin FALSE = FRESH SIGNAL
    """
    
    conditions_history = []
    
    # Ambil data dari lookback_days sebelumnya hingga hari ini
    start_idx = max(0, len(df) - lookback_days)
    
    for i in range(start_idx, len(df)):
        subset = df.iloc[:i+1]
        
        if len(subset) < 50:
            conditions_history.append(False)
            continue
        
        try:
            ema5 = calculate_ema(subset["Close"], EMA5_PERIOD)
            ema10 = calculate_ema(subset["Close"], EMA10_PERIOD)
            ema50 = calculate_ema(subset["Close"], EMA50_PERIOD)
            
            price = float(subset["Close"].iloc[-1])
            gain = calculate_daily_gain(subset)
            atr_pct = calculate_atr_percent(subset["High"], subset["Low"], subset["Close"])
            vol_ratio, _ = calculate_volume_pattern(subset)
            
            # STRICT CONDITIONS (sesuai panduan)
            cond1 = price > ema5.iloc[-1]
            cond2 = ema10.iloc[-1] > ema50.iloc[-1]
            cond3 = gain < MAX_DAILY_GAIN  # < 3%
            cond4 = atr_pct < MAX_ATR_PERCENT  # < 3%
            cond5 = MIN_VOL_RATIO <= vol_ratio < MAX_VOL_RATIO  # 1-2x
            
            all_conditions = cond1 and cond2 and cond3 and cond4 and cond5
            conditions_history.append(all_conditions)
        
        except:
            conditions_history.append(False)
    
    return conditions_history

def detect_fresh_signal(df: pd.DataFrame) -> bool:
    """
    Deteksi FRESH SIGNAL:
    - Hari ini semua kondisi TRUE
    - Minimal 1 hari sebelumnya ada yang FALSE
    
    Tujuan: Tangkap saham yang BARU memasuki quiet accumulation zone
    (undervalued) bukan yang sudah lama di sana
    """
    
    if len(df) < 3:
        return False
    
    conditions = check_conditions_history(df, lookback_days=3)
    
    if len(conditions) < 2:
        return False
    
    # Hari ini (paling akhir)
    today_qualified = conditions[-1]
    
    # Sebelumnya
    prev_qualified = conditions[:-1]
    
    # Fresh = hari ini TRUE, minimal 1 hari sebelumnya FALSE
    is_fresh = today_qualified and not all(prev_qualified)
    
    return is_fresh

# ======================================================
# 6. MAIN FILTER FUNCTION (STRICT ADHERENCE)
# ======================================================

def check_quiet_accumulation(df: pd.DataFrame) -> Optional[Dict]:
    """
    Cek apakah saham sedang dalam pattern "Akumulasi Pelan Naik Konstan"
    TETAP STRICT sesuai panduan original
    
    Return: dict dengan semua metrics atau None
    """
    
    if len(df) < 50:
        return None
    
    try:
        # === RULE 1: Price > EMA5 ===
        ema5 = calculate_ema(df["Close"], EMA5_PERIOD)
        ema10 = calculate_ema(df["Close"], EMA10_PERIOD)
        ema50 = calculate_ema(df["Close"], EMA50_PERIOD)
        
        price = float(df["Close"].iloc[-1])
        
        if price <= ema5.iloc[-1]:
            return None
        
        # === RULE 2: EMA10 > EMA50 ===
        if ema10.iloc[-1] <= ema50.iloc[-1]:
            return None
        
        # === RULE 3: Daily Gain < 3% (STRICT!) ===
        gain = calculate_daily_gain(df)
        if gain >= MAX_DAILY_GAIN:  # < 3% HARUS!
            return None
        
        # === RULE 4: ATR% < 3% (STRICT!) ===
        atr_pct = calculate_atr_percent(df["High"], df["Low"], df["Close"])
        if atr_pct >= MAX_ATR_PERCENT:  # < 3% HARUS!
            return None
        
        # === RULE 5: Volume > MA20 (ramai mulai) ===
        vol_ratio, vol_ma20 = calculate_volume_pattern(df)
        if vol_ratio <= MIN_VOL_RATIO:
            return None
        
        # === RULE 6: Volume < 2*MA20 (SENYAP - STRICT!) ===
        if vol_ratio >= MAX_VOL_RATIO:  # < 2x HARUS!
            return None
        
        # === RULE 7: HAKA > 50% ===
        haka = calculate_hajar_kanan_improved(df, lookback=5)
        if haka < MIN_HAJAR_KANAN_PCT:
            return None
        
        # === RULE 8: Money Flow >= 100jt ===
        mf = calculate_money_flow_single(df)
        if mf < MIN_MONEY_FLOW_SINGLE:
            return None
        
        # === RULE 9: Turnover >= 1 Milyar ===
        last_vol = float(df["Volume"].iloc[-1])
        turnover_bn = (price * last_vol) / 1_000_000_000
        if turnover_bn < MIN_TURNOVER:
            return None
        
        # === BONUS: Fresh Signal Detection ===
        is_fresh = detect_fresh_signal(df)
        
        # ===== ALL CHECKS PASSED =====
        return {
            "price": price,
            "gain": gain,
            "atr_pct": atr_pct,
            "vol_ratio": vol_ratio,
            "vol_ma20": vol_ma20,
            "haka": haka,
            "mf": mf / 1_000_000,  # Convert to Juta
            "turnover": turnover_bn,
            "ema5": float(ema5.iloc[-1]),
            "ema10": float(ema10.iloc[-1]),
            "ema50": float(ema50.iloc[-1]),
            "is_fresh": is_fresh
        }
        
    except Exception as e:
        return None

# ======================================================
# 7. SCORING SYSTEM (Konviksi Akumulasi)
# ======================================================

def calculate_accumulation_score(metrics: Dict) -> float:
    """
    Hitung score 0-100 untuk setiap saham
    Fokus: Seberapa kuat pattern Quiet Accumulation-nya
    """
    
    score = 0.0
    
    # 1. HAKA Score (Dominasi Pembeli) - Max 30
    haka = metrics["haka"]
    if haka >= 70:
        score += 30
    elif haka >= 60:
        score += 25
    elif haka >= 50:
        score += 20
    
    # 2. Money Flow Score - Max 30
    mf = metrics["mf"]
    if mf >= 300:
        score += 30
    elif mf >= 200:
        score += 25
    elif mf >= 100:
        score += 20
    
    # 3. Volume Pattern Score (Quiet Confirmation) - Max 25
    # Sweet spot: 1.3-1.8x (aktif tapi tetap senyap)
    vol_ratio = metrics["vol_ratio"]
    if 1.3 <= vol_ratio < 1.8:
        score += 25  # Best quiet pattern
    elif 1.0 <= vol_ratio < 1.3:
        score += 20
    elif 1.8 <= vol_ratio < 2.0:
        score += 20
    
    # 4. Stability Score (ATR%) - Max 15
    atr = metrics["atr_pct"]
    if atr < 1.5:
        score += 15  # Super tenang
    elif atr < 2.5:
        score += 10
    else:
        score += 5
    
    # 5. Fresh Signal Bonus - Max 5
    if metrics["is_fresh"]:
        score += 5  # Bonus untuk sinyal baru
    
    return float(np.clip(score, 0, 100))

# ======================================================
# 8. SINGLE STOCK ANALYSIS
# ======================================================

def analyze_stock(ticker: str) -> Optional[Dict]:
    """Analisa satu saham untuk pattern Quiet Accumulation"""
    try:
        df = fetch_stock_data(ticker, period="6mo")
        
        if df.empty or len(df) < 50:
            return None
        
        # Check if data updated today
        last_date = df.index[-1].date()
        today = dt.datetime.now(JAKARTA_TZ).date()
        if last_date != today:
            return None
        
        # Check quiet accumulation pattern
        metrics = check_quiet_accumulation(df)
        
        if not metrics:
            return None
        
        # Calculate conviction score
        score = calculate_accumulation_score(metrics)
        
        # Filter: Minimal score 60
        if score < 60:
            return None
        
        return {
            "ticker": ticker,
            "price": metrics["price"],
            "score": score,
            "gain": metrics["gain"],
            "atr_pct": metrics["atr_pct"],
            "vol_ratio": metrics["vol_ratio"],
            "haka": metrics["haka"],
            "mf": metrics["mf"],
            "turnover": metrics["turnover"],
            "is_fresh": metrics["is_fresh"]
        }
        
    except Exception as e:
        return None

# ======================================================
# 9. LOAD TICKERS & TELEGRAM
# ======================================================

def load_tickers() -> List[str]:
    """Load daftar saham BEI dari data/data.csv"""
    try:
        df = pd.read_csv("data/data.csv", header=None, names=["Symbol", "Name"])
        return df["Symbol"].dropna().tolist()
    except:
        print("❌ data/data.csv tidak ditemukan")
        return []

def send_telegram(text: str) -> None:
    """Kirim pesan ke Telegram Bot"""
    if not TELEGRAM_OK:
        print(text)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10
        )
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# ======================================================
# 10. MAIN EXECUTION
# ======================================================

def main() -> None:
    print("="*70)
    print("🔥 UT BOT v6.1 - QUIET ACCUMULATION SCANNER (CORRECTED)")
    print("   Strategi: Akumulasi Pelan Naik Konstan")
    print("   Filosofi: TENANG TAPI PASTI - STRICTLY ADHERENT")
    print("="*70)
    
    tickers = load_tickers()
    if not tickers:
        return
    
    results = []
    print(f"\n📊 Scanning {len(tickers)} saham BEI...")
    start_time = time.time()
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(tickers)} ({ticker})")
        
        result = analyze_stock(ticker)
        if result:
            results.append(result)
        
        # Rate limit
        if i % 10 == 0 and i > 0:
            time.sleep(0.1)
    
    # Sort by score descending (Fresh signals di top)
    results.sort(key=lambda x: (-x["score"], -x["is_fresh"]))
    
    # Build Telegram message
    now_wib = dt.datetime.now(JAKARTA_TZ).strftime("%d/%m %H:%M WIB")
    
    msg = f"🔥 QUIET ACCUMULATION SCANNER v6.1 (STRICT ADHERENCE)\n"
    msg += f"📅 {now_wib}\n"
    msg += f"🔎 Scanned: {len(tickers)} | Found: {len(results)} Quiet Accum\n\n"
    
    if results:
        msg += "💰 TOP PICKS (AKUMULASI PELAN NAIK KONSTAN):\n"
        msg += "="*70 + "\n"
        
        for rank, r in enumerate(results[:10], 1):
            # Star rating
            if r["score"] >= 85:
                stars = "⭐⭐⭐⭐⭐"
            elif r["score"] >= 75:
                stars = "⭐⭐⭐⭐"
            elif r["score"] >= 70:
                stars = "⭐⭐⭐"
            else:
                stars = "⭐⭐"
            
            fresh_badge = "🆕 FRESH!" if r["is_fresh"] else ""
            
            msg += f"{rank}. {r['ticker']} {stars} {fresh_badge}\n"
            msg += f"   Harga: {r['price']:.0f} | Score: {r['score']:.1f}/100\n"
            msg += f"   Naik: +{r['gain']:.2f}% | ATR: {r['atr_pct']:.2f}%\n"
            msg += f"   Vol: {r['vol_ratio']:.2f}x | HAKA: {r['haka']:.0f}%\n"
            msg += f"   MF: {r['mf']:.0f}Jt | Turnover: {r['turnover']:.1f}B\n"
            msg += "\n"
        
        msg += "="*70 + "\n\n"
    else:
        msg += "💤 Tidak ada saham dengan pattern Quiet Accumulation hari ini.\n\n"
    
    msg += "📋 KRITERIA FILTER (SESUAI PANDUAN):\n"
    msg += "✓ Price > EMA5 (Tren jangka sangat pendek naik)\n"
    msg += "✓ EMA10 > EMA50 (Tren menengah sehat, uptrend)\n"
    msg += "✓ Gain < 3% (Naiknya santai, belum terbang)\n"
    msg += "✓ ATR% < 3% (Tenang, tidak liar)\n"
    msg += "✓ Vol > MA20 (Minat pasar mulai meningkat)\n"
    msg += "✓ Vol < 2x MA20 (KUNCI SENYAP - Tidak terlalu heboh)\n"
    msg += "✓ HAKA > 50% (Pembeli sedikit dominan)\n"
    msg += "✓ Money Flow >= 100jt (Uang nominal besar terlibat)\n\n"
    msg += "💡 Score: 60-70=OK | 70-80=BAGUS | 80-90=MANTAP | 90+=JACKPOT\n"
    msg += "🆕 Fresh = Kondisi BARU tercipta (undervalued entry!)\n"
    
    duration = time.time() - start_time
    print(f"\n✅ Scan selesai dalam {duration:.2f} detik")
    print(f"📊 Total saham qualified: {len(results)}")
    
    # Send to Telegram
    send_telegram(msg)

if __name__ == "__main__":
    main()
