#!/usr/bin/env python3
# ================================================================
# scanner_dual_session_v5.py — Dual-Session Screener v5.0
# Menggabungkan keunggulan v3.2 (intraday real cutoff) dan
# v4.0 (kode bersih, ADX epsilon, filter gabungan efisien).
#
# Cara pakai:
#   python scanner_dual_session_v5.py session1   <- jam ~12:00 WIB
#   python scanner_dual_session_v5.py session2   <- jam ~15:30 WIB
# ================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings, os, logging, json, sys
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING
# ======================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"scanner_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DSv5")

VERSION = "5.0.0"

# ======================================================
# 1. KONSTANTA FILTER
# ======================================================
RSI_MAX            = 65
TREND_MARGIN       = 0.015      # harga > MA * (1 + 1.5%)
VOL_SPIKE          = 1.8        # volume > 1.8x MA20
ADX_MIN            = 25
MIN_PRICE_CHANGE   = 2.0        # kenaikan harian minimal 2%
CLOSE_HIGH_RATIO   = 0.96       # close >= 96% dari high
MIN_VALUE_IDR      = 200_000_000        # 200 juta
MIN_VALUE_MA20_IDR = 1_000_000_000      # 1 miliar

TP_PCT = 0.06
CL_PCT = 0.05

# Data fetch — intraday sebagai primary, daily sebagai fallback
PERIOD           = "60d"
INTERVAL_INTRA   = "1h"
INTERVAL_DAILY   = "1d"

# Cutoff UTC (WIB = UTC+7)
# Sesi 1: 12:00 WIB = 05:00 UTC
# Sesi 2: 15:45 WIB = 08:45 UTC
SESSION_CUTOFF = {
    "session1": dt.time(5, 0),
    "session2": dt.time(8, 45),
}

CACHE_FILE = "session1_results.json"

# ======================================================
# 2. TELEGRAM (opsional)
# ======================================================
TELEGRAM_OK        = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID   = ""
_req = None

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        import requests as _req
        TELEGRAM_OK = True
except Exception:
    pass

def send_telegram(text: str) -> bool:
    """Kirim pesan ke Telegram, chunked per baris agar tidak putus di tengah kata."""
    if not TELEGRAM_OK:
        return False
    MAX = 4096
    lines, chunk = text.split('\n'), ""
    chunks = []
    for line in lines:
        if len(chunk) + len(line) + 1 <= MAX:
            chunk = (chunk + '\n' + line).lstrip('\n')
        else:
            if chunk:
                chunks.append(chunk)
            chunk = line
    if chunk:
        chunks.append(chunk)

    ok = True
    for c in chunks:
        try:
            r = _req.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": c, "parse_mode": "HTML"},
                timeout=10
            )
            if r.status_code != 200:
                logger.error(f"Telegram error: {r.text}")
                ok = False
        except Exception as e:
            logger.error(f"Telegram exception: {e}")
            ok = False
    return ok

# ======================================================
# 3. DATA FETCHING
# Coba intraday 1h dulu; jika kosong/tidak cukup, fallback ke daily.
# Pada mode intraday, filter bar sesuai cutoff UTC.
# ======================================================
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi kolom MultiIndex dan pastikan kolom wajib ada."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    return df[required].dropna()

def fetch_intraday(ticker: str, cutoff: dt.time) -> pd.DataFrame:
    """Ambil data 1h lalu potong sampai cutoff UTC."""
    symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
    try:
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL_INTRA,
                         progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = _normalize_df(df)
        if df.empty:
            return pd.DataFrame()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Filter hanya bar sampai cutoff UTC pada hari ini
        today = dt.date.today()
        df = df[(df.index.date == today) | (df.index.date < today)]
        df = df[df.index.time <= cutoff]
        return df
    except Exception as e:
        logger.debug(f"Intraday fetch error {ticker}: {e}")
        return pd.DataFrame()

def fetch_daily(ticker: str) -> pd.DataFrame:
    symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
    try:
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL_DAILY,
                         progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        return _normalize_df(df)
    except Exception as e:
        logger.debug(f"Daily fetch error {ticker}: {e}")
        return pd.DataFrame()

def fetch_data(ticker: str, cutoff: dt.time) -> tuple[pd.DataFrame, str]:
    """
    Kembalikan (DataFrame, source) di mana source = 'intraday' atau 'daily'.
    Fallback ke daily jika intraday tidak cukup data (< 50 bar).
    """
    df = fetch_intraday(ticker, cutoff)
    if len(df) >= 50:
        return df, "intraday"
    df_daily = fetch_daily(ticker)
    if not df_daily.empty:
        return df_daily, "daily"
    return pd.DataFrame(), "none"

# ======================================================
# 4. INDIKATOR TEKNIKAL
# ======================================================
def calc_rsi(close: pd.Series, period: int = 14) -> float:
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        last_loss = avg_loss.iloc[-1]
        rs = avg_gain.iloc[-1] / last_loss if last_loss != 0 else 100.0
        return round(100 - 100 / (1 + rs), 1)
    except Exception:
        return -1.0

def calc_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """ADX dengan epsilon di denominator untuk menghindari division by zero."""
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        up   = high.diff()
        down = -low.diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        plus_di  = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr + 1e-9))
        dx  = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
        adx = dx.rolling(period).mean().iloc[-1]
        return round(float(adx), 1) if not np.isnan(adx) else None
    except Exception:
        return None

def check_macd_bullish(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    try:
        close    = df['Close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd     = ema_fast - ema_slow
        sig_line = macd.ewm(span=signal, adjust=False).mean()
        return bool(macd.iloc[-1] > sig_line.iloc[-1])
    except Exception:
        return False

# ======================================================
# 5. FILTER — urutan dari yang paling cepat di-reject
# ======================================================
def pass_all_filters(df: pd.DataFrame) -> bool:
    if len(df) < 50:
        return False

    curr  = df.iloc[-1]
    prev  = df.iloc[-2]
    close = float(curr['Close'])
    high  = float(curr['High'])
    vol   = float(curr['Volume'])

    if close <= 0 or vol <= 0:
        return False

    # --- [FAST-FAIL] ADX dulu — filter paling diskriminatif ---
    adx = calc_adx(df)
    if adx is None or adx < ADX_MIN:
        return False

    # --- [FAST-FAIL] MACD bullish ---
    if not check_macd_bullish(df):
        return False

    # --- Filter 1: BSJP dasar ---
    pct_change = (close - float(prev['Close'])) / float(prev['Close']) * 100
    if pct_change < MIN_PRICE_CHANGE:
        return False
    rsi = calc_rsi(df['Close'])
    if rsi == -1 or rsi >= RSI_MAX:
        return False
    if close < high * CLOSE_HIGH_RATIO:
        return False
    if vol < df['Volume'].rolling(5).mean().iloc[-1]:
        return False

    # --- Filter 2: Trend & Volume ---
    ma20     = df['Close'].rolling(20).mean()
    ma50     = df['Close'].rolling(50).mean()
    vol_ma20 = df['Volume'].rolling(20).mean()
    if close <= ma20.iloc[-1] * (1 + TREND_MARGIN):
        return False
    if close <= ma50.iloc[-1] * (1 + TREND_MARGIN):
        return False
    if vol < vol_ma20.iloc[-1] * VOL_SPIKE:
        return False

    # --- Filter 3: Likuiditas ---
    curr_value = close * vol
    avg_value  = (ma20 * vol_ma20).iloc[-1]
    if curr_value < MIN_VALUE_IDR:
        return False
    if curr_value <= avg_value:
        return False
    if avg_value < MIN_VALUE_MA20_IDR:
        return False

    # --- Filter 4: Bandarmologi proxy ---
    df2 = df[['Open', 'Close', 'Volume']].copy()
    df2['Bandar']     = np.where(df2['Close'] > df2['Open'],
                                  df2['Close'] * df2['Volume'] * 0.5,
                                  -df2['Close'] * df2['Volume'] * 0.5)
    df2['Bandar_MA20'] = df2['Bandar'].rolling(20).mean()
    df2['Bandar_MA10'] = df2['Bandar'].rolling(10).mean()
    b_curr = df2['Bandar'].iloc[-1]
    b_prev = df2['Bandar'].iloc[-2]
    b_ma20 = df2['Bandar_MA20'].iloc[-1]
    b_ma10 = df2['Bandar_MA10'].iloc[-1]
    if b_curr <= b_ma20:
        return False
    if b_prev > b_curr:
        return False
    if b_ma10 <= b_ma20:
        return False

    return True

# ======================================================
# 6. ANALISIS PER SAHAM
# ======================================================
def analyze_ticker(ticker: str, cutoff: dt.time) -> Optional[Dict]:
    df, source = fetch_data(ticker, cutoff)
    if df.empty or len(df) < 50:
        return None
    if not pass_all_filters(df):
        return None
    curr  = df.iloc[-1]
    entry = int(curr['Close'])
    adx   = calc_adx(df)
    return {
        "ticker":  ticker,
        "close":   entry,
        "rsi":     calc_rsi(df['Close']),
        "value_b": round(float(curr['Close']) * float(curr['Volume']) / 1e9, 2),
        "adx":     adx,
        "entry":   entry,
        "tp":      int(entry * (1 + TP_PCT)),
        "cl":      int(entry * (1 - CL_PCT)),
        "source":  source,          # intraday / daily (transparansi)
    }

# ======================================================
# 7. MANAJEMEN DUAL SESSION
# ======================================================
def save_session1(results: List[Dict]):
    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)

def load_session1() -> List[Dict]:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def run_session(label: str, tickers: List[str], cutoff: dt.time) -> List[Dict]:
    logger.info(f"=== Memulai {label} (cutoff UTC {cutoff}) ===")
    matches = []
    total   = len(tickers)
    for i, ticker in enumerate(tickers):
        print(f"\r  [{label}] {(i+1)/total*100:5.1f}% — {ticker:<6}", end="", flush=True)
        try:
            res = analyze_ticker(ticker, cutoff)
            if res:
                logger.info(f"  ✅ {ticker} lolos {label} "
                            f"(source={res['source']}, RSI={res['rsi']}, ADX={res['adx']})")
                matches.append(res)
        except KeyboardInterrupt:
            print()
            logger.info("Dibatalkan oleh pengguna.")
            break
        except Exception as e:
            logger.debug(f"Error {ticker}: {e}")
    print()
    logger.info(f"{label} selesai: {len(matches)} kandidat dari {total} saham")
    return matches

# ======================================================
# 8. MAIN
# ======================================================
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("session1", "session2"):
        print("Gunakan: python scanner_dual_session_v5.py [session1|session2]")
        sys.exit(1)
    mode = sys.argv[1]

    # --- Load tickers dari CSV ---
    csv_path = "data.csv" if os.path.exists("data.csv") else os.path.join("data", "data.csv")
    try:
        df_csv = pd.read_csv(csv_path)
        col     = next((c for c in ('Ticker','ticker','Kode','kode') if c in df_csv.columns),
                       df_csv.columns[0])
        tickers = sorted({str(t).strip().upper() for t in df_csv[col].dropna()
                          if len(str(t).strip()) >= 4})
        logger.info(f"Total ticker: {len(tickers)}")
    except Exception:
        logger.error("File data.csv tidak ditemukan atau format salah.")
        sys.exit(1)

    cutoff = SESSION_CUTOFF[mode]

    if mode == "session1":
        matches = run_session("Sesi 1 (Siang ~12:00 WIB)", tickers, cutoff)
        if matches:
            save_session1(matches)
            logger.info(f"✅ {len(matches)} kandidat disimpan ke {CACHE_FILE}")
            send_telegram(
                f"<b>Sesi 1 selesai</b>\n"
                f"{len(matches)} kandidat akan dikonfirmasi di Sesi 2.\n"
                + "\n".join(f"• {m['ticker']} (RSI {m['rsi']} | ADX {m['adx']})" for m in matches)
            )
        else:
            logger.info("Tidak ada kandidat di Sesi 1.")
            send_telegram("ℹ️ Sesi 1: Tidak ada kandidat ditemukan hari ini.")

    else:  # session2
        matches_s2 = run_session("Sesi 2 (Sore ~15:45 WIB)", tickers, cutoff)
        matches_s1 = load_session1()

        if not matches_s1:
            logger.warning("⚠️ Data Sesi 1 tidak ditemukan. Jalankan session1 lebih dulu.")
            send_telegram("⚠️ Dual Session v5: Data Sesi 1 tidak ditemukan.")
            return

        t1        = {s['ticker'] for s in matches_s1}
        confirmed = [s for s in matches_s2 if s['ticker'] in t1]

        if confirmed:
            header = (
                f"<b>Deepseek Screener - 🔔 Dual Session Signal v{VERSION}</b>\n"
                f"{len(confirmed)} saham lolos konfirmasi ganda:\n\n"
            )
            body = "\n".join(
                f"• <b>{c['ticker']}</b> @ {c['entry']:,}\n"
                f"  TP {c['tp']:,} (+{TP_PCT*100:.0f}%) | CL {c['cl']:,} (-{CL_PCT*100:.0f}%)\n"
                f"  RSI {c['rsi']} | ADX {c['adx']} | Val {c['value_b']:.2f}B | [{c['source']}]"
                for c in confirmed
            )
            msg = header + body
            logger.info(msg)
            send_telegram(msg)
            try:
                os.remove(CACHE_FILE)
            except Exception:
                pass
        else:
            logger.info("Tidak ada saham yang lolos dual-session hari ini.")
            send_telegram(f"ℹ️ Dual Session v{VERSION}: Tidak ada sinyal trading hari ini.")

if __name__ == "__main__":
    main()