# improved_scanner_v7.py - UT BOT SCANNER v7.0 (WIN RATE OPTIMIZED)
# STRATEGY: "Akumulasi Pelan Naik Konstan" (Smart Money Footprint)
# ENHANCEMENTS v7:
#   [+] RSI Filter (hindari overbought & oversold ekstrem)
#   [+] Higher Low Structure (konfirmasi struktur harga naik)
#   [+] Consolidation Tightness (pastikan bukan distribusi)
#   [+] VWAP Proximity Filter (harga dekat/di atas VWAP = sehat)
#   [+] Relative Strength vs IHSG (stock lebih kuat dari pasar)
#   [+] Candle Quality Filter (candle bullish, bukan pin bar/shooting star)
#   [+] Score System lebih granular (0-100, lebih diskriminatif)
#   [+] Smarter Freshness Logic (crossover EMA & vol surge terpisah)
#   [+] Gain filter diperlonggar untuk menghindari false negative
#   [+] EMA perfect alignment bonus (EMA5 > EMA10 > EMA20 > EMA50)
# ====================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import time
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# ======================================================
# 1. CONFIGURATION
# ======================================================
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except ImportError:
    TELEGRAM_OK = False
    print("⚠️  config.py tidak ditemukan. Telegram notifikasi dinonaktifkan.")

JAKARTA_TZ   = dt.timezone(dt.timedelta(hours=7))
LOOKBACK_PERIOD = "8mo"   # Diperpanjang agar RSI & Higher Low lebih akurat
IHSG_TICKER  = "^JKSE"   # Benchmark IHSG untuk Relative Strength

# ── STRATEGY PARAMETERS ─────────────────────────────────────────────
PARAMS = {
    # Trend
    "EMA_SHORT":  5,
    "EMA_MID":    10,
    "EMA_LONG":   50,
    "EMA_EXTRA":  20,        # Tambahan EMA20 untuk perfect-alignment bonus

    # Gain & Volatility
    "MAX_DAILY_GAIN":   3.5,   # % — sedikit dilonggarkan agar tidak miss saham bagus
    "MIN_DAILY_GAIN":  -1.0,   # % — buang hari merah tajam
    "MAX_ATR_PERCENT":  3.0,   # %
    "MIN_ATR_PERCENT":  0.3,   # % — hindari saham yang benar-benar stagnan (stuck)

    # Volume
    "VOL_RATIO_MIN":   1.0,    # Volume > MA20
    "VOL_RATIO_MAX":   2.5,    # Tidak euforia

    # Buying Pressure (HAKA proxy)
    "MIN_BUYING_PRESSURE": 58.0,   # % (sedikit dilonggarkan, terkompensasi filter baru)

    # Likuiditas
    "MIN_TURNOVER_IDR":  100_000_000,   # 100 Juta

    # ── FILTER BARU v7 ──
    "RSI_PERIOD":       14,
    "RSI_MIN":          40.0,    # Tidak oversold kronis
    "RSI_MAX":          70.0,    # Tidak overbought (belum terbang)

    "HIGHER_LOW_BARS":  10,      # Cek Higher Low dalam N bar terakhir
    "CONSOL_BARS":      10,      # Window konsolidasi untuk tightness check
    "CONSOL_TIGHT_PCT":  8.0,    # Max % range selama konsolidasi (High - Low)

    "VWAP_PERIOD":       20,     # Periode rolling VWAP
    "VWAP_MAX_ABOVE":    3.0,    # % maks di atas VWAP (tidak terlalu jauh)

    "RS_PERIOD":         20,     # Periode RS vs IHSG
}

# ======================================================
# 2. DATA UTILITIES
# ======================================================
def fetch_stock_data(ticker: str, period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    """Fetch OHLCV dari Yahoo Finance."""
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            progress=False, auto_adjust=False,
            multi_level_index=False
        )
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        df.columns = [c.capitalize() for c in df.columns]
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        return df[required].copy()
    except Exception:
        return pd.DataFrame()


def fetch_ihsg(period: str = LOOKBACK_PERIOD) -> pd.Series:
    """Fetch harga penutupan IHSG sebagai benchmark."""
    try:
        df = yf.download(IHSG_TICKER, period=period, interval="1d",
                         progress=False, auto_adjust=True,
                         multi_level_index=False)
        df.columns = [c.capitalize() for c in df.columns]
        return df['Close'] if 'Close' in df.columns else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


# ======================================================
# 3. TECHNICAL INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1/period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_vwap_rolling(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP sederhana: sum(typical_price * vol) / sum(vol) dalam window."""
    tp  = (df['High'] + df['Low'] + df['Close']) / 3
    num = (tp * df['Volume']).rolling(period).sum()
    den = df['Volume'].rolling(period).sum()
    return num / den


def calculate_relative_strength(stock_close: pd.Series,
                                 ihsg_close: pd.Series,
                                 period: int = 20) -> pd.Series:
    """RS = (pct_change stock N hari) / (pct_change IHSG N hari). > 1 = outperform."""
    stock_ret = stock_close.pct_change(period)
    ihsg_ret  = ihsg_close.reindex(stock_close.index, method='ffill').pct_change(period)
    # Hindari divide-by-zero
    return stock_ret / ihsg_ret.replace(0, np.nan)


def check_higher_low(df: pd.DataFrame, n_bars: int = 10) -> bool:
    """
    Cek apakah low dalam n_bars terakhir membentuk pola Higher Low.
    Minimal ada 1 Higher Low yang terkonfirmasi (low[-1] > low[-n_bars]).
    """
    if len(df) < n_bars + 1:
        return False
    lows = df['Low'].iloc[-(n_bars+1):].values
    # Sederhana: Low terakhir lebih tinggi dari Low paling kiri window
    return float(lows[-1]) > float(lows[0])


def calculate_indicators(df: pd.DataFrame,
                          ihsg_close: pd.Series) -> pd.DataFrame:
    """Hitung semua indikator teknikal."""

    # ── EMAs ──────────────────────────────────────────────────────────
    df['EMA5']  = df['Close'].ewm(span=PARAMS['EMA_SHORT'], adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=PARAMS['EMA_MID'],   adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=PARAMS['EMA_EXTRA'], adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=PARAMS['EMA_LONG'],  adjust=False).mean()

    # ── Gain ──────────────────────────────────────────────────────────
    df['PrevClose'] = df['Close'].shift(1)
    df['Gain']      = ((df['Close'] - df['PrevClose']) / df['PrevClose']) * 100

    # ── ATR % ────────────────────────────────────────────────────────
    h_l  = df['High'] - df['Low']
    h_pc = (df['High'] - df['PrevClose']).abs()
    l_pc = (df['Low']  - df['PrevClose']).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr  = tr.rolling(14).mean()
    df['ATRP'] = (atr / df['Close']) * 100

    # ── Volume ───────────────────────────────────────────────────────
    df['VolMA20']  = df['Volume'].rolling(20).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA20']

    # ── Buying Pressure (HAKA proxy) ─────────────────────────────────
    rng = df['High'] - df['Low']
    raw_haka = np.where(rng > 0, (df['Close'] - df['Low']) / rng, 0.5)
    haka_s   = pd.Series(raw_haka, index=df.index)
    df['BuyPressMA5'] = haka_s.rolling(5).mean() * 100
    df['BuyPressMA5'] = df['BuyPressMA5'].bfill()

    # ── Turnover ─────────────────────────────────────────────────────
    df['Turnover'] = df['Close'] * df['Volume']

    # ── RSI ──────────────────────────────────────────────────────────
    df['RSI'] = calculate_rsi(df['Close'], period=PARAMS['RSI_PERIOD'])

    # ── Rolling VWAP ─────────────────────────────────────────────────
    df['VWAP'] = calculate_vwap_rolling(df, period=PARAMS['VWAP_PERIOD'])

    # ── Relative Strength vs IHSG ────────────────────────────────────
    if not ihsg_close.empty:
        df['RS_IHSG'] = calculate_relative_strength(
            df['Close'], ihsg_close, period=PARAMS['RS_PERIOD'])
    else:
        df['RS_IHSG'] = np.nan

    # ── Candle Body Ratio (untuk filter candle sehat) ────────────────
    body      = (df['Close'] - df['Open']).abs()
    full_rng  = (df['High'] - df['Low']).replace(0, np.nan)
    df['BodyRatio'] = body / full_rng    # > 0.4 = candle berisi, bukan doji/pin

    # ── Upper Shadow Ratio (deteksi shooting star / distribusi) ──────
    upper_shadow     = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['UpperShadow'] = upper_shadow / full_rng   # < 0.35 = tidak ada distribusi atas

    return df


# ======================================================
# 4. SIGNAL LOGIC
# ======================================================
def check_strategy(df: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """
    Evaluasi candle terakhir terhadap semua kriteria.
    Return Dict jika valid, None jika tidak lolos.
    """
    if len(df) < 60:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # ── FILTER WAJIB (HARD RULES) ────────────────────────────────────

    # 1. Trend EMA
    if curr['Close']  <= curr['EMA5']:  return None   # Short trend harus UP
    if curr['EMA10']  <= curr['EMA50']: return None   # Mid trend harus UP

    # 2. Gain harian
    if curr['Gain']   >  PARAMS['MAX_DAILY_GAIN']:   return None   # Terlalu spike
    if curr['Gain']   <  PARAMS['MIN_DAILY_GAIN']:   return None   # Hari merah tajam

    # 3. Volatilitas (ATR%)
    if curr['ATRP']   >  PARAMS['MAX_ATR_PERCENT']:  return None   # Terlalu liar
    if curr['ATRP']   <  PARAMS['MIN_ATR_PERCENT']:  return None   # Saham zombie/stuck

    # 4. Volume
    if curr['VolRatio'] < PARAMS['VOL_RATIO_MIN']:   return None   # Sepi
    if curr['VolRatio'] > PARAMS['VOL_RATIO_MAX']:   return None   # Euforia

    # 5. Buying Pressure
    if curr['BuyPressMA5'] < PARAMS['MIN_BUYING_PRESSURE']: return None

    # 6. Likuiditas
    if curr['Turnover'] < PARAMS['MIN_TURNOVER_IDR']: return None

    # ── FILTER BARU v7 ───────────────────────────────────────────────

    # 7. RSI — tidak overbought, tidak oversold kronis
    if not np.isnan(curr['RSI']):
        if curr['RSI'] > PARAMS['RSI_MAX']: return None   # Sudah mahal/overbought
        if curr['RSI'] < PARAMS['RSI_MIN']: return None   # Downtrend tersembunyi

    # 8. VWAP — harga tidak terlalu jauh di atas VWAP (hindari yang sudah extended)
    if not np.isnan(curr['VWAP']) and curr['VWAP'] > 0:
        vwap_pct = ((curr['Close'] - curr['VWAP']) / curr['VWAP']) * 100
        if vwap_pct > PARAMS['VWAP_MAX_ABOVE']:
            return None   # Harga sudah terlalu jauh di atas VWAP
        if vwap_pct < -2.0:
            return None   # Harga di bawah VWAP, beli dengan hati-hati

    # 9. Higher Low structure (konfirmasi akumulasi nyata)
    if not check_higher_low(df, n_bars=PARAMS['HIGHER_LOW_BARS']):
        return None

    # 10. Konsolidasi Tight (range High-Low dalam window tidak lebih dari X%)
    recent_window = df.iloc[-PARAMS['CONSOL_BARS']:]
    consol_high   = recent_window['High'].max()
    consol_low    = recent_window['Low'].min()
    if consol_low > 0:
        consol_range_pct = ((consol_high - consol_low) / consol_low) * 100
        if consol_range_pct > PARAMS['CONSOL_TIGHT_PCT']:
            return None   # Range terlalu lebar — bukan konsolidasi, mungkin distribusi

    # 11. Candle body harus berisi (bukan doji atau pin bar murni)
    if curr['BodyRatio'] < 0.30:
        return None   # Candle tidak berisi = tidak ada commitment bullish

    # 12. Upper shadow tidak boleh dominan (tanda distribusi/penolakan)
    if curr['UpperShadow'] > 0.40:
        return None   # Upper shadow terlalu panjang = ada penjual kuat di atas

    # ── FRESHNESS CHECK ──────────────────────────────────────────────
    is_fresh_ema_cross  = (prev['Close'] <= prev['EMA5']) and (curr['Close'] > curr['EMA5'])
    is_fresh_vol_surge  = (prev['VolRatio'] < 1.0) and (curr['VolRatio'] >= 1.0)
    is_fresh_rsi_entry  = not np.isnan(prev['RSI']) and (prev['RSI'] < 50) and (curr['RSI'] >= 50)

    fresh_count = sum([is_fresh_ema_cross, is_fresh_vol_surge, is_fresh_rsi_entry])

    if fresh_count >= 2:
        status_label = "FRESH 🟢🟢"   # Multi-konfirmasi fresh
    elif fresh_count == 1:
        status_label = "FRESH 🟢"
    else:
        status_label = "ACCUM 🟡"

    # ── SCORE ────────────────────────────────────────────────────────
    score = calculate_score(curr, is_fresh_ema_cross or is_fresh_vol_surge)

    # ── Relative Strength vs IHSG ────────────────────────────────────
    rs_val  = curr['RS_IHSG'] if not np.isnan(curr['RS_IHSG']) else None
    rs_label = f"{rs_val:.2f}x" if rs_val else "N/A"

    return {
        "ticker":       ticker,
        "price":        float(curr['Close']),
        "gain":         float(curr['Gain']),
        "vol_ratio":    float(curr['VolRatio']),
        "haka_proxy":   float(curr['BuyPressMA5']),
        "rsi":          float(curr['RSI']) if not np.isnan(curr['RSI']) else 0,
        "atrp":         float(curr['ATRP']),
        "rs_ihsg":      rs_val,
        "rs_label":     rs_label,
        "turnover_b":   float(curr['Turnover']) / 1_000_000_000,
        "status":       status_label,
        "score":        score,
        "consol_range": consol_range_pct,
    }


def calculate_score(row, is_fresh: bool) -> int:
    """Skor kualitas setup (0-100). Lebih granular dari v6."""
    score = 50   # Base score (lebih ketat dari v6's 60)

    # Fresh signal bonus
    if is_fresh:
        score += 10

    # RSI ideal zone (50–62 = early momentum, belum overbought)
    rsi = row['RSI'] if not np.isnan(row['RSI']) else 0
    if 50 <= rsi <= 62:
        score += 8
    elif 62 < rsi <= 68:
        score += 4   # Masih ok tapi mulai panas

    # Buying pressure
    if row['BuyPressMA5'] >= 70:
        score += 8
    elif row['BuyPressMA5'] >= 65:
        score += 4

    # ATRP — makin tenang makin bagus
    if row['ATRP'] < 1.5:
        score += 8
    elif row['ATRP'] < 2.0:
        score += 4

    # Volume — ideal zone (tidak terlalu ramai, tidak terlalu sepi)
    if 1.1 <= row['VolRatio'] <= 1.8:
        score += 8
    elif row['VolRatio'] <= 2.0:
        score += 4

    # EMA perfect alignment (EMA5 > EMA10 > EMA20 > EMA50)
    ema_ok = (row['EMA5'] > row['EMA10'] > row['EMA20'] > row['EMA50'])
    if ema_ok:
        score += 8

    # Candle body kuat
    if row['BodyRatio'] >= 0.60:
        score += 5

    # Relative Strength vs IHSG
    if not np.isnan(row['RS_IHSG']) and row['RS_IHSG'] > 1.2:
        score += 5   # Outperform IHSG = smart money pilih saham ini

    return min(score, 100)


# ======================================================
# 5. TELEGRAM REPORT
# ======================================================
def send_telegram_report(results: List[Dict]):
    if not TELEGRAM_OK:
        return

    now_str = dt.datetime.now(JAKARTA_TZ).strftime('%d-%m-%Y %H:%M WIB')
    msg = (f"🔍 <b>QUIET ACCUMULATION SCANNER v7</b>\n"
           f"📅 {now_str}\n"
           f"Strategi: Akumulasi Pelan, Naik Konstan\n\n")

    if not results:
        msg += ("💤 <b>HASIL SCAN: NIHIL</b>\n"
                "Tidak ada saham yang lolos semua filter hari ini.\n"
                "Pasar mungkin terlalu volatil atau distribusi.\n")
    else:
        fresh_signals = [r for r in results if "FRESH" in r['status']]
        accum_signals = [r for r in results if "ACCUM" in r['status']]

        if fresh_signals:
            msg += "🚀 <b>FRESH SIGNALS (Baru Mulai):</b>\n"
            for r in fresh_signals[:5]:
                rs_str = f"RS:{r['rs_label']}" if r['rs_label'] != "N/A" else ""
                msg += (f"• <b>{r['ticker']}</b> [Skor:{r['score']}]\n"
                        f"  P:{r['price']:.0f} | G:{r['gain']:.1f}% | "
                        f"RSI:{r['rsi']:.0f} | HAKA:{r['haka_proxy']:.0f}% | "
                        f"Vol:{r['vol_ratio']:.1f}x | {rs_str}\n")
            msg += "\n"

        if accum_signals:
            msg += "⏳ <b>ONGOING ACCUMULATION:</b>\n"
            for r in accum_signals[:8]:
                rs_str = f"RS:{r['rs_label']}" if r['rs_label'] != "N/A" else ""
                msg += (f"• {r['ticker']} [Skor:{r['score']}]\n"
                        f"  P:{r['price']:.0f} | G:{r['gain']:.1f}% | "
                        f"RSI:{r['rsi']:.0f} | HAKA:{r['haka_proxy']:.0f}% | "
                        f"Vol:{r['vol_ratio']:.1f}x | {rs_str}\n")

        msg += ("\n<i>Note: FRESH = EMA5 cross/Vol surge/RSI cross 50 hari ini.\n"
                "RS > 1 = outperform IHSG.</i>")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
        print("✅ Telegram sent.")
    except Exception as e:
        print(f"❌ Telegram failed: {e}")


# ======================================================
# 6. MAIN
# ======================================================
def main():
    print("=" * 65)
    print("🚀 SCANNER v7: Quiet Accumulation — Win Rate Optimized")
    print("=" * 65)

    # ── Load Tickers ──────────────────────────────────────────────
    try:
        tickers_df = pd.read_csv("data/data.csv", header=None)
        all_tickers = tickers_df.iloc[:, 0].tolist()
    except Exception:
        print("⚠️ data/data.csv tidak ditemukan. Pakai sample tickers.")
        all_tickers = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII",
                       "UNTR", "ICBP", "KLBF", "EXCL", "MDKA"]

    # ── Fetch IHSG sekali (benchmark) ────────────────────────────
    print("📊 Fetching IHSG benchmark...")
    ihsg_close = fetch_ihsg()
    if ihsg_close.empty:
        print("⚠️  IHSG tidak bisa di-fetch. RS vs IHSG dinonaktifkan.")

    results    = []
    start_time = time.time()
    found_count = 0

    for i, ticker in enumerate(all_tickers):
        print(f"\rScanning {i+1}/{len(all_tickers)}: {ticker:<8} | Found: {found_count}",
              end="", flush=True)

        df = fetch_stock_data(ticker)
        if df.empty:
            continue

        df  = calculate_indicators(df, ihsg_close)
        res = check_strategy(df, ticker)

        if res:
            results.append(res)
            found_count += 1

    elapsed = time.time() - start_time
    print(f"\n\n✅ Scan selesai dalam {elapsed:.1f}s")
    print(f"   Saham terpilih : {len(results)} dari {len(all_tickers)}")

    # ── Sort: Fresh multi-konfirmasi dulu, lalu score ─────────────
    priority = {"FRESH 🟢🟢": 0, "FRESH 🟢": 1, "ACCUM 🟡": 2}
    results.sort(key=lambda x: (priority.get(x['status'], 9), -x['score']))

    # ── Console Output ────────────────────────────────────────────
    header = (f"\n{'TICKER':<8} | {'STATUS':<12} | {'SCORE':<5} | {'PRICE':<8} | "
              f"{'GAIN%':<6} | {'RSI':<5} | {'HAKA%':<6} | {'VOL(x)':<6} | "
              f"{'ATRP%':<6} | {'RS-IHSG'}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['ticker']:<8} | {r['status']:<12} | {r['score']:<5} | "
              f"{r['price']:<8.0f} | {r['gain']:<6.1f} | {r['rsi']:<5.0f} | "
              f"{r['haka_proxy']:<6.0f} | {r['vol_ratio']:<6.1f} | "
              f"{r['atrp']:<6.2f} | {r['rs_label']}")

    # ── Send Telegram ─────────────────────────────────────────────
    send_telegram_report(results)


if __name__ == "__main__":
    main()