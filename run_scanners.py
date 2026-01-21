# scanner_pro_v5_ultimate.py - UT BOT v5.0 (HIGH WIN RATE EDITION)
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

# Strategy Settings
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 50
RSI_BUY_MAX = 70  # NEW: Hindari overbought
EMA_TREND_PERIOD = 200
MIN_DAILY_VALUE = 1_000_000_000

# Enhanced Filter Thresholds
MIN_SIGNAL_SCORE = 75          # Dinaikkan dari 70
MIN_VOLUME_RATIO = 1.3
MIN_BODY_RATIO = 0.55          # Sedikit dilonggarkan untuk catch more setup
MIN_EMA_DISTANCE = 0.5
MAX_EMA_DISTANCE = 4.0         # Dikurangi dari 5% (lebih ketat)
RSI_MOMENTUM_THRESHOLD = 2

# NEW: Advanced Filters
MIN_ADX = 20                   # ADX >= 20 untuk trend yang jelas
MIN_RELATIVE_STRENGTH = 0.5    # Saham harus outperform IHSG minimal 0.5%
MAX_GAP_YESTERDAY = 3.0        # Max gap kemarin 3% (risk control)
MIN_RR_RATIO = 2.0             # Minimum Risk/Reward 2:1
SUPPORT_PROXIMITY_PCT = 2.0    # Entry max 2% dari support

# Market Regime (IHSG)
INDEX_SYMBOL = "^JKSE"
MIN_INDEX_EMA_DISTANCE = -2.0  # IHSG minimal -2% dari EMA50 (not too bearish)

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
# 4. MARKET REGIME CHECKER (NEW)
# ======================================================
def check_market_regime() -> Tuple[bool, float]:
    """
    Cek kondisi IHSG: apakah market dalam kondisi healthy untuk long position.
    Returns: (is_healthy, ema_distance_pct)
    """
    try:
        df_index = yf.download(INDEX_SYMBOL, period="3mo", interval="1d", progress=False, timeout=10)
        
        if df_index.empty or len(df_index) < 50:
            return True, 0.0  # Default allow jika data tidak tersedia
        
        if isinstance(df_index.columns, pd.MultiIndex):
            df_index.columns = [col[0] for col in df_index.columns]
        
        close_price = df_index['Close'].iloc[-1]
        ema50 = df_index['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        ema_dist = ((close_price - ema50) / ema50) * 100
        
        # Market dianggap healthy jika IHSG di atas atau tidak terlalu jauh di bawah EMA50
        is_healthy = ema_dist >= MIN_INDEX_EMA_DISTANCE
        
        return is_healthy, ema_dist
        
    except Exception as e:
        print(f"⚠️  Gagal cek market regime: {e}")
        return True, 0.0  # Default allow

# ======================================================
# 5. TECHNICAL ANALYSIS ENGINE (ENHANCED)
# ======================================================
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Menghitung Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Menghitung RSI full series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Menghitung ATR sebagai Series."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Menghitung ADX (Average Directional Index) untuk measure trend strength.
    Returns: ADX value (0-100)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # ATR
    atr = calculate_atr(df, period)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(period).sum()
    
    # DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # ADX
    adx = dx.rolling(period).mean()
    
    return adx.iloc[-1] if len(adx) > 0 else 0.0

def calculate_vwap(df: pd.DataFrame) -> float:
    """Menghitung VWAP untuk hari ini (approx dengan historical)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).sum() / df['Volume'].sum()
    return vwap

def detect_support_level(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Deteksi support level terdekat menggunakan swing lows.
    Returns: Support price level
    """
    lows = df['Low'].tail(lookback)
    
    # Cari lowest low dalam lookback period
    support = lows.min()
    
    # Alternatif: gunakan percentile untuk lebih smooth
    support_alt = lows.quantile(0.1)  # Bottom 10% dari lows
    
    return max(support, support_alt)

def check_price_structure(df: pd.DataFrame) -> bool:
    """
    Cek apakah price structure membentuk higher highs dan higher lows (uptrend).
    Returns: True jika struktur bullish
    """
    if len(df) < 10:
        return False
    
    # Bandingkan 5 candle terakhir dengan 5 candle sebelumnya
    recent_high = df['High'].tail(5).max()
    prev_high = df['High'].iloc[-10:-5].max()
    
    recent_low = df['Low'].tail(5).min()
    prev_low = df['Low'].iloc[-10:-5].min()
    
    higher_high = recent_high > prev_high
    higher_low = recent_low > prev_low
    
    return higher_high and higher_low

def calculate_relative_strength(df_stock: pd.DataFrame, df_index: pd.DataFrame) -> float:
    """
    Menghitung relative strength: stock performance vs IHSG (5 days).
    Returns: RS% (positive = outperform)
    """
    if len(df_stock) < 5 or len(df_index) < 5:
        return 0.0
    
    stock_change = ((df_stock['Close'].iloc[-1] / df_stock['Close'].iloc[-6]) - 1) * 100
    index_change = ((df_index['Close'].iloc[-1] / df_index['Close'].iloc[-6]) - 1) * 100
    
    return stock_change - index_change

def check_gap_risk(df: pd.DataFrame) -> Tuple[bool, float]:
    """
    Cek apakah kemarin ada gap besar (risk overnight gap continuation).
    Returns: (is_safe, gap_pct)
    """
    if len(df) < 2:
        return True, 0.0
    
    yesterday_close = df['Close'].iloc[-2]
    today_open = df['Open'].iloc[-1]
    
    gap_pct = abs(((today_open / yesterday_close) - 1) * 100)
    
    is_safe = gap_pct <= MAX_GAP_YESTERDAY
    
    return is_safe, gap_pct

def calculate_rr_ratio(current_price: float, support: float, atr: float) -> float:
    """
    Menghitung Risk/Reward Ratio.
    Risk = current_price - support
    Reward = 2 * ATR (target)
    """
    risk = current_price - support
    
    if risk <= 0:
        return 0.0
    
    reward = 2.0 * atr  # Target 2x ATR
    
    rr = reward / risk
    
    return rr

def calculate_ut_bot(
    df: pd.DataFrame, 
    key_value: float = 2.0, 
    atr_period: int = 10
) -> Tuple[bool, bool]:
    """
    Menghitung sinyal UT Bot menggunakan Numpy.
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
# 6. ENHANCED SIGNAL QUALITY SCORING
# ======================================================
def calculate_signal_quality_score(
    df: pd.DataFrame, 
    df_index: pd.DataFrame,
    rsi_current: float, 
    rsi_prev: float,
    adx: float,
    relative_strength: float,
    rr_ratio: float,
    support_distance_pct: float
) -> float:
    """
    Menghitung skor kualitas sinyal (0-100) dengan faktor tambahan.
    """
    score = 0.0
    
    # 1. VOLUME SURGE (20 poin max) - dikurangi weight
    vol_20_avg = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = df['Volume'].iloc[-1] / vol_20_avg if vol_20_avg > 0 else 1.0
    vol_score = min(20, (vol_ratio - 1.0) * 15)
    score += max(0, vol_score)
    
    # 2. PRICE ACTION QUALITY (20 poin max) - dikurangi weight
    open_price = df['Open'].iloc[-1]
    close_price = df['Close'].iloc[-1]
    high_price = df['High'].iloc[-1]
    low_price = df['Low'].iloc[-1]
    
    body = close_price - open_price
    total_range = high_price - low_price
    
    if total_range > 0:
        body_ratio = body / total_range
        candle_score = min(20, max(0, (body_ratio - 0.3) * 40))
    else:
        candle_score = 0
    score += candle_score
    
    # 3. RSI MOMENTUM (15 poin max) - dikurangi weight
    rsi_momentum = rsi_current - rsi_prev
    momentum_score = min(15, max(0, rsi_momentum * 3))
    score += momentum_score
    
    # 4. ADX TREND STRENGTH (15 poin max) - NEW
    if adx >= 25:
        adx_score = 15
    elif adx >= 20:
        adx_score = 10
    elif adx >= 15:
        adx_score = 5
    else:
        adx_score = 0
    score += adx_score
    
    # 5. RELATIVE STRENGTH (15 poin max) - NEW
    if relative_strength >= 2.0:
        rs_score = 15
    elif relative_strength >= 1.0:
        rs_score = 12
    elif relative_strength >= 0.5:
        rs_score = 8
    else:
        rs_score = 0
    score += rs_score
    
    # 6. RISK/REWARD RATIO (10 poin max) - NEW
    if rr_ratio >= 3.0:
        rr_score = 10
    elif rr_ratio >= 2.0:
        rr_score = 7
    elif rr_ratio >= 1.5:
        rr_score = 3
    else:
        rr_score = 0
    score += rr_score
    
    # 7. SUPPORT PROXIMITY (5 poin max) - NEW
    if support_distance_pct <= 1.0:
        support_score = 5
    elif support_distance_pct <= 2.0:
        support_score = 3
    else:
        support_score = 0
    score += support_score
    
    return float(score)

# ======================================================
# 7. MAIN SCANNER LOGIC (ENHANCED)
# ======================================================
def fetch_and_analyze(
    ticker: str, 
    settings: List[Tuple], 
    df_index: pd.DataFrame,
    market_healthy: bool
) -> List[Dict]:
    """Mengambil data 1 saham dan mengecek dengan ALL enhanced filters."""
    detected_signals = []
    
    # Skip jika market tidak healthy
    if not market_healthy:
        return []
    
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

        # --------------------------------------------------
        # FILTER 2B: EMA DISTANCE
        # --------------------------------------------------
        ema_distance_pct = ((current_close - ema200) / ema200) * 100
        if ema_distance_pct < MIN_EMA_DISTANCE or ema_distance_pct > MAX_EMA_DISTANCE:
            return []

        # --------------------------------------------------
        # FILTER 3: RSI (with upper bound)
        # --------------------------------------------------
        rsi_series = calculate_rsi(df['Close'], RSI_PERIOD)
        rsi_current = rsi_series.iloc[-1]
        rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else rsi_current
        
        if rsi_current <= RSI_BUY_THRESHOLD or rsi_current >= RSI_BUY_MAX:
            return []

        # --------------------------------------------------
        # FILTER 4: RSI MOMENTUM
        # --------------------------------------------------
        rsi_momentum = rsi_current - rsi_prev
        if rsi_momentum < RSI_MOMENTUM_THRESHOLD:
            return []

        # --------------------------------------------------
        # FILTER 5: VOLUME SURGE
        # --------------------------------------------------
        vol_20_avg = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df['Volume'].iloc[-1] / vol_20_avg if vol_20_avg > 0 else 0
        
        if vol_ratio < MIN_VOLUME_RATIO:
            return []

        # --------------------------------------------------
        # FILTER 6: PRICE ACTION QUALITY
        # --------------------------------------------------
        open_price = df['Open'].iloc[-1]
        close_price = df['Close'].iloc[-1]
        high_price = df['High'].iloc[-1]
        low_price = df['Low'].iloc[-1]
        
        body = close_price - open_price
        total_range = high_price - low_price
        
        if total_range > 0:
            body_ratio = body / total_range
            if body_ratio < MIN_BODY_RATIO:
                return []
        else:
            return []

        # --------------------------------------------------
        # FILTER 7: ADX TREND STRENGTH (NEW)
        # --------------------------------------------------
        adx = calculate_adx(df, 14)
        if adx < MIN_ADX:
            return []

        # --------------------------------------------------
        # FILTER 8: RELATIVE STRENGTH (NEW)
        # --------------------------------------------------
        relative_strength = calculate_relative_strength(df, df_index)
        if relative_strength < MIN_RELATIVE_STRENGTH:
            return []

        # --------------------------------------------------
        # FILTER 9: GAP RISK (NEW)
        # --------------------------------------------------
        is_gap_safe, gap_pct = check_gap_risk(df)
        if not is_gap_safe:
            return []

        # --------------------------------------------------
        # FILTER 10: PRICE STRUCTURE (NEW)
        # --------------------------------------------------
        has_bullish_structure = check_price_structure(df)
        if not has_bullish_structure:
            return []

        # --------------------------------------------------
        # FILTER 11: SUPPORT & R/R RATIO (NEW)
        # --------------------------------------------------
        support_level = detect_support_level(df, 20)
        support_distance_pct = ((current_close - support_level) / support_level) * 100
        
        # Entry harus dekat support (max SUPPORT_PROXIMITY_PCT%)
        if support_distance_pct > SUPPORT_PROXIMITY_PCT:
            return []
        
        atr_14 = calculate_atr(df, 14).iloc[-1]
        rr_ratio = calculate_rr_ratio(current_close, support_level, atr_14)
        
        if rr_ratio < MIN_RR_RATIO:
            return []

        # --------------------------------------------------
        # CHECK SIGNALS (UT BOT)
        # --------------------------------------------------
        for name, key_val, atr_per in settings:
            fresh_buy, _ = calculate_ut_bot(df, key_val, atr_per)
            
            if fresh_buy:
                # Calculate enhanced signal quality score
                quality_score = calculate_signal_quality_score(
                    df, df_index, rsi_current, rsi_prev, adx, 
                    relative_strength, rr_ratio, support_distance_pct
                )
                
                # Filter by score threshold
                if quality_score >= MIN_SIGNAL_SCORE:
                    # Calculate target & stop loss
                    stop_loss = support_level
                    target = current_close + (2 * atr_14)
                    
                    detected_signals.append({
                        "ticker": ticker,
                        "setting": name,
                        "price": current_close,
                        "stop_loss": stop_loss,
                        "target": target,
                        "rsi": rsi_current,
                        "rsi_momentum": rsi_momentum,
                        "adx": adx,
                        "relative_strength": relative_strength,
                        "volume_ratio": vol_ratio,
                        "body_ratio": body_ratio,
                        "rr_ratio": rr_ratio,
                        "support_dist": support_distance_pct,
                        "value_avg_bn": mean_value / 1_000_000_000,
                        "ema_200": ema200,
                        "distance_from_ema": ema_distance_pct,
                        "quality_score": quality_score
                    })
                    
    except Exception as e:
        pass
        
    return detected_signals

def main():
    print("\n" + "="*80)
    print("🚀 UT BOT ULTIMATE v5.0 - HIGH WIN RATE OVERNIGHT SWING SCANNER")
    print(f"⚙️  Multi-Layer Filters: Market Regime | Trend | RSI | Volume | ADX | RS")
    print(f"          Structure | Support | R/R | Gap Risk | Quality Score")
    print("="*80)

    start_time = time.time()
    
    # Check Market Regime
    print("\n🔍 Checking Market Regime (IHSG)...")
    market_healthy, ihsg_ema_dist = check_market_regime()
    
    if market_healthy:
        print(f"✅ Market Healthy: IHSG {ihsg_ema_dist:+.2f}% from EMA50")
    else:
        print(f"❌ Market NOT Healthy: IHSG {ihsg_ema_dist:+.2f}% from EMA50")
        print("⚠️  Bearish market regime - No trades recommended today.")
        send_telegram_message(
            f"🛑 *MARKET REGIME CHECK FAILED*\n"
            f"IHSG: {ihsg_ema_dist:+.2f}% from EMA50\n"
            f"Market terlalu bearish - Tidak ada scan hari ini."
        )
        return
    
    # Load IHSG data for relative strength calc
    try:
        df_index = yf.download(INDEX_SYMBOL, period="3mo", interval="1d", progress=False, timeout=10)
        if isinstance(df_index.columns, pd.MultiIndex):
            df_index.columns = [col[0] for col in df_index.columns]
    except:
        df_index = pd.DataFrame()
    
    tickers = load_bei_tickers()
    
    bot_settings = [
        ("AGRESIF", 1.0, 10),
        ("STANDAR", 2.0, 10),
        ("KONSERVATIF", 3.0, 14)
    ]
    
    results = {s[0]: [] for s in bot_settings}
    
    print(f"\n🔍 Scanning {len(tickers)} saham... (Estimasi 5-10 menit)")
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"   Processed {i}/{len(tickers)}...")
            
        signals = fetch_and_analyze(ticker, bot_settings, df_index, market_healthy)
        
        for sig in signals:
            group = sig['setting']
            results[group].append(sig)
            print(f"   🔥 FOUND: {sig['ticker']} ({group}) | "
                  f"RSI: {sig['rsi']:.1f} | ADX: {sig['adx']:.0f} | "
                  f"RS: {sig['relative_strength']:.1f}% | Score: {sig['quality_score']:.0f}")

    # ======================================================
    # REPORTING
    # ======================================================
    duration = (time.time() - start_time) / 60
    
    report = f"🤖 *UT BOT ULTIMATE v5.0 - ENHANCED OVERNIGHT SIGNALS*\n"
    report += f"📅 {dt.datetime.now().strftime('%d-%m-%Y %H:%M')} WIB\n"
    report += f"⏱️ Scan Time: {duration:.1f} min\n"
    report += f"📊 IHSG Status: {ihsg_ema_dist:+.2f}% from EMA50 ✅\n\n"
    report += f"🛡️ *10-Layer Filter System*:\n"
    report += f"   1. Market Regime (IHSG healthy)\n"
    report += f"   2. EMA200 Trend ({MIN_EMA_DISTANCE}-{MAX_EMA_DISTANCE}%)\n"
    report += f"   3. Liquidity >1M Rp\n"
    report += f"   4. RSI ({RSI_BUY_THRESHOLD}-{RSI_BUY_MAX}) + Momentum +{RSI_MOMENTUM_THRESHOLD}\n"
    report += f"   5. Volume Surge >x{MIN_VOLUME_RATIO}\n"
    report += f"   6. Bullish Candle >{int(MIN_BODY_RATIO*100)}%\n"
    report += f"   7. ADX Trend >{MIN_ADX}\n"
    report += f"   8. Relative Strength vs IHSG\n"
    report += f"   9. Price Structure (HH/HL)\n"
    report += f"   10. Support + R/R >{MIN_RR_RATIO}:1\n\n"
    
    total_found = 0
    for name in ["AGRESIF", "STANDAR", "KONSERVATIF"]:
        data = results[name]
        if not data:
            continue
            
        # Sort by quality score
        data_sorted = sorted(data, key=lambda x: x['quality_score'], reverse=True)
        
        total_found += len(data_sorted)
        report += f"📌 *MODE {name}* ({len(data_sorted)} signals):\n"
        for s in data_sorted[:10]:  # Top 10 saja
            report += (f"• `{s['ticker']}` @ {s['price']:.0f}\n"
                      f"  SL: {s['stop_loss']:.0f} | Target: {s['target']:.0f} (R/R {s['rr_ratio']:.1f}:1)\n"
                      f"  RSI {s['rsi']:.0f} | ADX {s['adx']:.0f} | "
                      f"RS +{s['relative_strength']:.1f}% | Score {s['quality_score']:.0f}\n\n")
        report += "\n"
        
    if total_found == 0:
        report += "ℹ️ Tidak ada setup berkualitas tinggi hari ini.\n"
        report += "Sabar menunggu setup terbaik = Kunci long-term success."
    else:
        report += f"✅ Total {total_found} sinyal premium quality.\n"
        report += f"🎯 Trade Plan: Beli 15:15-15:25 WIB, Jual esok pagi 09:05-09:30 WIB"

    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    send_telegram_message(report)

if __name__ == "__main__":
    main()
