# ================================================================
# scanner_bsjp_v4_dual_session.py
# BSJP IDX DUAL-SESSION CONFIRMATION SCANNER
#
# Cara pakai utama:
#   1) Jam 12:00-13:30  : python scanner_bsjp_v4_dual_session.py --session session1
#   2) Jam 15:45-15:55  : python scanner_bsjp_v4_dual_session.py --session session2
#
# Prinsip utama v4:
#   - Sesi siang hanya WATCHLIST, belum sinyal beli.
#   - Sesi sore hanya BUY CONFIRMED jika ticker muncul di sesi siang DAN lolos lagi di sesi sore.
#   - Filter dibuat lebih defensif untuk mengurangi noise, saham spike sesaat, dan false signal.
#   - Menggunakan data harian + intraday 5 menit dari yfinance jika tersedia.
# ================================================================

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ================================================================
# 0. BASIC CONFIG
# ================================================================
VERSION = "4.0.0-DualSession-NoiseGuard"
TIMEZONE = "Asia/Jakarta"

DAILY_PERIOD = "9mo"
DAILY_INTERVAL = "1d"
INTRADAY_PERIOD = "5d"
INTRADAY_INTERVAL = "5m"

STATE_DIR = "state"
RESULT_DIR = "results"
LOG_DIR = "logs"
for _d in [STATE_DIR, RESULT_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"scanner_v4_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("BSJP_V4_DUAL_SESSION")

# ================================================================
# 1. STRATEGY CONSTANTS
# ================================================================
# Trading rule user
TP_PERCENT = 0.06              # Take Profit 6%
MAX_HOLD_DAYS = 7              # Jika 7 hari tidak TP, keluar
HARD_CL_PERCENT = 0.05         # Stop darurat opsional untuk proteksi risiko intraday/harian

# Trend periods
MA20_PERIOD = 20
MA50_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
OBV_MA_PERIOD = 10
MONEY_FLOW_FAST = 5
MONEY_FLOW_SLOW = 20

# Base liquidity & price sanity
MIN_PRICE = 50
MAX_PRICE = 20_000
MIN_AVG_VALUE_20D = 1_500_000_000     # rata-rata value 20 hari minimal 1.5B
MIN_VALUE_SESSION1 = 750_000_000      # karena sesi siang belum full day
MIN_VALUE_SESSION2 = 2_000_000_000    # sore mendekati full day

# Momentum guard: jangan beli saham yang sudah terlalu panas
MIN_CHANGE_PCT = 2.0
MAX_CHANGE_PCT = 10.5                # filter chasing/pump terlalu tinggi
RSI_MIN_SESSION1 = 52
RSI_MAX_SESSION1 = 73
RSI_MIN_SESSION2 = 54
RSI_MAX_SESSION2 = 72
MAX_DISTANCE_FROM_MA20_PCT = 14.0
MAX_GAP_UP_PCT = 5.5
MAX_ATR_PCT = 9.0

# Volume/value confirmation, session-specific
VOL_RATIO_SESSION1 = 0.65             # volume sampai sesi 1 >= 65% rerata volume harian 20D
VALUE_RATIO_SESSION1 = 0.65           # value sampai sesi 1 >= 65% rerata value harian 20D
VOL_RATIO_SESSION2 = 1.15             # volume full day harus benar-benar di atas normal
VALUE_RATIO_SESSION2 = 1.10
MAX_VOL_RATIO = 8.0                   # terlalu ekstrem sering rawan euforia/fade

# Candle quality: kurangi sinyal candle yang ditutup lemah/rejection
MIN_CLOSE_POSITION = 0.70             # close minimal di 70% atas range candle
MIN_BODY_RATIO = 0.40                 # body minimal 40% range
MAX_UPPER_WICK_RATIO = 0.35           # upper wick maksimal 35% range

# Score threshold
MIN_SCORE_SESSION1 = 10               # watchlist siang
MIN_SCORE_SESSION2 = 12               # konfirmasi sore

# Cooldown: hindari ticker yang baru saja menjadi final signal beberapa hari terakhir
COOLDOWN_DAYS = 2

MAX_MESSAGE_LENGTH = 4096

# ================================================================
# 2. TELEGRAM CONFIG
# ================================================================
TELEGRAM_OK = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if REQUESTS_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        TELEGRAM_OK = True
        logger.info("Telegram config loaded.")
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram dinonaktifkan.")


@dataclass
class Thresholds:
    session: str
    min_score: int
    min_value: int
    vol_ratio: float
    value_ratio: float
    rsi_min: float
    rsi_max: float


def get_thresholds(session: str) -> Thresholds:
    if session == "session1":
        return Thresholds(
            session=session,
            min_score=MIN_SCORE_SESSION1,
            min_value=MIN_VALUE_SESSION1,
            vol_ratio=VOL_RATIO_SESSION1,
            value_ratio=VALUE_RATIO_SESSION1,
            rsi_min=RSI_MIN_SESSION1,
            rsi_max=RSI_MAX_SESSION1,
        )
    return Thresholds(
        session=session,
        min_score=MIN_SCORE_SESSION2,
        min_value=MIN_VALUE_SESSION2,
        vol_ratio=VOL_RATIO_SESSION2,
        value_ratio=VALUE_RATIO_SESSION2,
        rsi_min=RSI_MIN_SESSION2,
        rsi_max=RSI_MAX_SESSION2,
    )


# ================================================================
# 3. TELEGRAM HELPERS
# ================================================================
def split_telegram_message(message: str) -> List[str]:
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks, current_chunk = [], ""
    for line in message.split("\n"):
        test = current_chunk + line + "\n"
        if len(test) > MAX_MESSAGE_LENGTH:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk = test
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        return False
    ok = True
    for chunk in split_telegram_message(message):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}
            resp = requests.post(url, data=data, timeout=15)
            if resp.status_code != 200:
                logger.error("Gagal kirim Telegram: %s", resp.text)
                ok = False
            time.sleep(0.5)
        except Exception as e:
            logger.error("Error Telegram: %s", e)
            ok = False
    return ok


# ================================================================
# 4. DATA LOADER
# ================================================================
def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper().replace(".JK", "")
    t = "".join(ch for ch in t if ch.isalnum())
    return t


def load_tickers_from_csv(path: str) -> List[str]:
    candidates = [path, os.path.join("data", path), os.path.join("data", "data.csv"), "data.csv"]
    existing = next((p for p in candidates if os.path.exists(p)), None)
    if not existing:
        logger.error("File ticker tidak ditemukan. Siapkan data.csv dengan kolom Ticker/Kode.")
        return []

    df = pd.read_csv(existing)
    col = next((c for c in ["Ticker", "ticker", "Kode", "kode", "Code", "code", "Emiten"] if c in df.columns), df.columns[0])
    tickers = sorted({normalize_ticker(x) for x in df[col].dropna() if len(normalize_ticker(x)) >= 3})
    logger.info("Loaded %s tickers from %s", len(tickers), existing)
    return tickers


def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    out = df[required].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def fetch_daily_data(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=DAILY_PERIOD, interval=DAILY_INTERVAL, progress=False, auto_adjust=False, threads=False)
        return clean_yf_df(df)
    except Exception as e:
        logger.debug("daily fetch failed %s: %s", symbol, e)
        return pd.DataFrame()


def fetch_intraday_today(symbol: str) -> Optional[pd.Series]:
    """Ambil candle berjalan hari ini dari data 5m: open pertama, high max, low min, close terakhir, volume sum."""
    try:
        df = yf.download(symbol, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL, progress=False, auto_adjust=False, threads=False)
        df = clean_yf_df(df)
        if df.empty:
            return None

        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            # Yahoo kadang mengembalikan naive timestamp. Anggap UTC agar aman untuk konversi tanggal.
            idx = idx.tz_localize("UTC").tz_convert(TIMEZONE)
        else:
            idx = idx.tz_convert(TIMEZONE)
        df = df.copy()
        df.index = idx

        today = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date()
        today_df = df[df.index.date == today]
        if today_df.empty:
            return None

        return pd.Series(
            {
                "Open": float(today_df["Open"].iloc[0]),
                "High": float(today_df["High"].max()),
                "Low": float(today_df["Low"].min()),
                "Close": float(today_df["Close"].iloc[-1]),
                "Volume": float(today_df["Volume"].sum()),
            },
            name=pd.Timestamp(today),
        )
    except Exception as e:
        logger.debug("intraday fetch failed %s: %s", symbol, e)
        return None


def fetch_stock_data(ticker: str, use_intraday: bool = True) -> pd.DataFrame:
    symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
    daily = fetch_daily_data(symbol)
    if daily.empty:
        return pd.DataFrame()

    if use_intraday:
        today_bar = fetch_intraday_today(symbol)
        if today_bar is not None and today_bar["Close"] > 0 and today_bar["Volume"] > 0:
            today_key = pd.Timestamp(today_bar.name).normalize()
            daily = daily.copy()
            # Normalize daily index agar bisa replace candle hari ini bila ada
            daily_index_norm = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
            daily.index = daily_index_norm
            if today_key in daily.index:
                daily.loc[today_key, ["Open", "High", "Low", "Close", "Volume"]] = today_bar[["Open", "High", "Low", "Close", "Volume"]].values
            else:
                daily.loc[today_key, ["Open", "High", "Low", "Close", "Volume"]] = today_bar[["Open", "High", "Low", "Close", "Volume"]].values
            daily = daily.sort_index()
    return daily.dropna()


# ================================================================
# 5. INDICATORS
# ================================================================
def calculate_rsi_series(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["Close"].shift(1)

    df["Value"] = df["Close"] * df["Volume"]
    # Rata-rata pembanding sengaja exclude hari ini agar tidak bias oleh candle berjalan.
    df["Value_MA20_Prev"] = df["Value"].shift(1).rolling(MA20_PERIOD).mean()
    df["Vol_MA20_Prev"] = df["Volume"].shift(1).rolling(MA20_PERIOD).mean()

    df["MA20"] = df["Close"].rolling(MA20_PERIOD).mean()
    df["MA50"] = df["Close"].rolling(MA50_PERIOD).mean()
    df["MA20_5D_AGO"] = df["MA20"].shift(5)
    df["MA50_5D_AGO"] = df["MA50"].shift(5)

    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - prev_close).abs()
    low_prev_close = (df["Low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR14"] = true_range.rolling(ATR_PERIOD).mean()
    df["ATR_PCT"] = (df["ATR14"] / df["Close"]) * 100

    df["RSI"] = calculate_rsi_series(df["Close"], RSI_PERIOD)
    df["OBV"] = calculate_obv(df["Close"], df["Volume"])
    df["OBV_MA10"] = df["OBV"].rolling(OBV_MA_PERIOD).mean()
    df["OBV_SLOPE3"] = df["OBV"] - df["OBV"].shift(3)

    # Money Flow proxy lebih baik daripada sekadar Close > Open.
    # Close Location Value: semakin dekat close ke high, semakin positif.
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng).fillna(0)
    df["MFV"] = mfm * df["Value"]
    df["MFV_MA5"] = df["MFV"].rolling(MONEY_FLOW_FAST).mean()
    df["MFV_MA20"] = df["MFV"].rolling(MONEY_FLOW_SLOW).mean()

    df["High20_Prev"] = df["High"].shift(1).rolling(20).max()
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Close"] = prev_close
    return df


# ================================================================
# 6. SIGNAL ENGINE
# ================================================================
def safe_ratio(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0:
        return 0.0
    return float(a / b)


def bool_icon(x: bool) -> str:
    return "✅" if x else "❌"


def analyze_stock(ticker: str, session: str, use_intraday: bool = True) -> Optional[Dict]:
    thresholds = get_thresholds(session)
    raw = fetch_stock_data(ticker, use_intraday=use_intraday)
    if raw.empty or len(raw) < MA50_PERIOD + 10:
        return None

    df = add_indicators(raw)
    current = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(current["Close"])
    open_ = float(current["Open"])
    high = float(current["High"])
    low = float(current["Low"])
    volume = float(current["Volume"])
    value = float(current["Value"])
    prev_close = float(current["Prev_Close"])

    if close <= 0 or volume <= 0 or prev_close <= 0:
        return None

    day_range = max(high - low, 1e-9)
    price_change_pct = ((close - prev_close) / prev_close) * 100
    gap_pct = ((open_ - prev_close) / prev_close) * 100
    close_position = (close - low) / day_range
    body_ratio = abs(close - open_) / day_range
    upper_wick_ratio = (high - max(close, open_)) / day_range
    distance_ma20_pct = ((close / current["MA20"]) - 1) * 100 if current["MA20"] else 999
    vol_ratio = safe_ratio(volume, current["Vol_MA20_Prev"])
    value_ratio = safe_ratio(value, current["Value_MA20_Prev"])

    checks: Dict[str, bool] = {}
    checks["price_ok"] = MIN_PRICE <= close <= MAX_PRICE
    checks["avg_value_ok"] = current["Value_MA20_Prev"] >= MIN_AVG_VALUE_20D
    checks["today_value_ok"] = value >= thresholds.min_value
    checks["change_ok"] = MIN_CHANGE_PCT <= price_change_pct <= MAX_CHANGE_PCT
    checks["gap_ok"] = gap_pct <= MAX_GAP_UP_PCT
    checks["trend_ok"] = all([
        close > current["MA20"],
        current["MA20"] > current["MA50"],
        current["MA20"] > current["MA20_5D_AGO"],
    ])
    checks["volume_ok"] = thresholds.vol_ratio <= vol_ratio <= MAX_VOL_RATIO
    checks["value_spike_ok"] = value_ratio >= thresholds.value_ratio
    checks["rsi_ok"] = thresholds.rsi_min <= current["RSI"] <= thresholds.rsi_max
    checks["not_extended_ok"] = distance_ma20_pct <= MAX_DISTANCE_FROM_MA20_PCT
    checks["atr_ok"] = current["ATR_PCT"] <= MAX_ATR_PCT
    checks["green_candle_ok"] = close > open_
    checks["close_strength_ok"] = close_position >= MIN_CLOSE_POSITION
    checks["body_ok"] = body_ratio >= MIN_BODY_RATIO
    checks["wick_ok"] = upper_wick_ratio <= MAX_UPPER_WICK_RATIO
    checks["breakout_ok"] = (close > current["Prev_High"]) or (close >= current["High20_Prev"] * 0.995)
    checks["obv_ok"] = (current["OBV"] > current["OBV_MA10"]) and (current["OBV_SLOPE3"] > 0)
    checks["money_flow_ok"] = (current["MFV"] > 0) and (current["MFV_MA5"] > current["MFV_MA20"])

    score = int(sum(checks.values()))

    # Critical filters: kalau ini gagal, biasanya sinyal noisy/chasing.
    critical_keys = [
        "price_ok", "avg_value_ok", "today_value_ok", "trend_ok", "volume_ok",
        "value_spike_ok", "rsi_ok", "not_extended_ok", "close_strength_ok",
        "gap_ok", "money_flow_ok",
    ]
    critical_pass = all(checks[k] for k in critical_keys)

    if not (critical_pass and score >= thresholds.min_score):
        return None

    failed = [k for k, v in checks.items() if not v]
    entry = int(round(close))
    tp = int(round(entry * (1 + TP_PERCENT)))
    hard_cl = int(round(entry * (1 - HARD_CL_PERCENT)))

    return {
        "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y-%m-%d"),
        "scan_time": dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S"),
        "session": session,
        "ticker": ticker,
        "entry": entry,
        "tp_6pct": tp,
        "hard_cl_5pct": hard_cl,
        "max_hold_days": MAX_HOLD_DAYS,
        "close": round(close, 2),
        "change_pct": round(price_change_pct, 2),
        "gap_pct": round(gap_pct, 2),
        "rsi": round(float(current["RSI"]), 1),
        "value_b": round(value / 1e9, 2),
        "avg_value20_b": round(float(current["Value_MA20_Prev"]) / 1e9, 2),
        "value_ratio": round(value_ratio, 2),
        "volume_ratio": round(vol_ratio, 2),
        "close_position": round(close_position, 2),
        "body_ratio": round(body_ratio, 2),
        "upper_wick_ratio": round(upper_wick_ratio, 2),
        "distance_ma20_pct": round(distance_ma20_pct, 2),
        "atr_pct": round(float(current["ATR_PCT"]), 2),
        "score": score,
        "failed_noncritical": ",".join(failed),
        "checks_json": json.dumps(checks, ensure_ascii=False),
    }


# ================================================================
# 7. SESSION STATE & HISTORY
# ================================================================
def today_str() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y-%m-%d")


def state_path(session: str, date_str: Optional[str] = None) -> str:
    d = date_str or today_str()
    return os.path.join(STATE_DIR, f"{d}_{session}_candidates.csv")


def result_path(name: str, date_str: Optional[str] = None) -> str:
    d = date_str or today_str()
    return os.path.join(RESULT_DIR, f"{d}_{name}.csv")


def load_recent_history() -> set:
    path = os.path.join(STATE_DIR, "final_signal_history.csv")
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        if df.empty or "date" not in df.columns or "ticker" not in df.columns:
            return set()
        cutoff = pd.Timestamp(today_str()) - pd.Timedelta(days=COOLDOWN_DAYS)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        recent = df[df["date"] >= cutoff]
        return set(recent["ticker"].dropna().astype(str).str.upper())
    except Exception:
        return set()


def append_final_history(final_df: pd.DataFrame) -> None:
    if final_df.empty:
        return
    path = os.path.join(STATE_DIR, "final_signal_history.csv")
    cols = ["date", "scan_time", "ticker", "entry", "tp_6pct", "hard_cl_5pct", "score", "value_b", "rsi"]
    out = final_df[[c for c in cols if c in final_df.columns]].copy()
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, out], ignore_index=True)
        out = out.drop_duplicates(subset=["date", "ticker"], keep="last")
    out.to_csv(path, index=False)


# ================================================================
# 8. OUTPUT FORMAT
# ================================================================
def format_results_console(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    if df.empty:
        print("Tidak ada saham yang lolos.")
        return
    show_cols = [
        "ticker", "score", "entry", "tp_6pct", "hard_cl_5pct", "change_pct", "rsi",
        "value_b", "value_ratio", "volume_ratio", "close_position", "distance_ma20_pct"
    ]
    print(df[show_cols].to_string(index=False))


def format_telegram_message(df: pd.DataFrame, session: str, total_scanned: int, final_mode: bool = False) -> str:
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")
    if session == "session1":
        title = "WATCHLIST SIANG — BELUM BUY"
        note = "Saham ini baru kandidat. Buy hanya jika muncul ulang di screener sore."
    elif final_mode:
        title = "CONFIRMED SORE — BOLEH JADI SAHAM TRADING"
        note = "Ticker lolos sesi siang dan lolos ulang 10-15 menit sebelum closing."
    else:
        title = "SCAN SORE — RAW RESULT"
        note = "Ini hasil sore mentah, belum diintersect dengan sesi siang."

    lines = [
        f"<b>BSJP Scanner V{VERSION}</b>",
        f"<b>{title}</b>",
        f"<i>{now}</i>",
        "",
        f"Scanned: {total_scanned} | Match: <b>{len(df)}</b>",
        f"Rule: TP {int(TP_PERCENT*100)}% | Max Hold {MAX_HOLD_DAYS} hari | Hard CL opsional {int(HARD_CL_PERCENT*100)}%",
        f"<i>{note}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if df.empty:
        lines.append("Tidak ada saham yang lolos filter v4.")
        return "\n".join(lines)

    for _, r in df.iterrows():
        lines.append(
            f"<b>{r['ticker']}</b> | Score {int(r['score'])} | Chg {r['change_pct']:+.2f}% | "
            f"RSI {r['rsi']:.1f} | Val {r['value_b']:.2f}B"
        )
        lines.append(
            f"Entry {int(r['entry']):,} → TP6% {int(r['tp_6pct']):,} | Hard CL {int(r['hard_cl_5pct']):,}"
        )
        lines.append(
            f"Volx {r['volume_ratio']:.2f} | Valx {r['value_ratio']:.2f} | ClosePos {r['close_position']:.2f} | MA20+{r['distance_ma20_pct']:.1f}%"
        )
        lines.append("")
    return "\n".join(lines)


# ================================================================
# 9. SCANNER RUNNER
# ================================================================
def scan_tickers(tickers: List[str], session: str, use_intraday: bool, skip_recent: bool) -> pd.DataFrame:
    rows = []
    recent = load_recent_history() if skip_recent else set()
    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        print(f"\r[{i:>4}/{total}] {ticker:<6}", end="", flush=True)
        if ticker in recent:
            continue
        try:
            res = analyze_stock(ticker, session=session, use_intraday=use_intraday)
            if res:
                print(f"  ✅ {ticker} score={res['score']} chg={res['change_pct']}%", flush=True)
                rows.append(res)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.debug("Analyze failed %s: %s", ticker, e)
            continue
    print()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["score", "value_b", "change_pct"], ascending=[False, False, False]).reset_index(drop=True)


def run_session(session: str, tickers_file: str, use_intraday: bool, skip_recent: bool) -> None:
    tickers = load_tickers_from_csv(tickers_file)
    if not tickers:
        print("Ticker kosong. Pastikan file data.csv tersedia.")
        return

    print(f"\nBSJP Scanner V{VERSION}")
    print(f"Session: {session} | Use intraday: {use_intraday} | Tickers: {len(tickers)}")

    raw_df = scan_tickers(tickers, session=session, use_intraday=use_intraday, skip_recent=skip_recent)
    raw_out = result_path(f"{session}_raw")
    raw_df.to_csv(raw_out, index=False)

    if session == "session1":
        # Sesi 1 hanya disimpan sebagai kandidat.
        raw_df.to_csv(state_path("session1"), index=False)
        format_results_console(raw_df, "WATCHLIST SIANG — BELUM BUY. Tunggu konfirmasi sore.")
        print(f"\nSaved session1 candidates: {state_path('session1')}")
        print(f"Saved raw result        : {raw_out}")
        if TELEGRAM_OK:
            send_telegram_message(format_telegram_message(raw_df, session="session1", total_scanned=len(tickers)))
        return

    # Session2: harus intersect dengan kandidat session1.
    session1_file = state_path("session1")
    if not os.path.exists(session1_file):
        print("\nPERINGATAN: file kandidat sesi 1 tidak ditemukan.")
        print("Jalankan dulu: python scanner_bsjp_v4_dual_session.py --session session1")
        final_df = pd.DataFrame()
    else:
        s1 = pd.read_csv(session1_file)
        if s1.empty or raw_df.empty:
            final_df = pd.DataFrame()
        else:
            s1_tickers = set(s1["ticker"].astype(str).str.upper())
            final_df = raw_df[raw_df["ticker"].astype(str).str.upper().isin(s1_tickers)].copy()
            final_df = final_df.sort_values(["score", "value_b", "change_pct"], ascending=[False, False, False]).reset_index(drop=True)

    final_out = result_path("FINAL_CONFIRMED_TRADING")
    final_df.to_csv(final_out, index=False)
    append_final_history(final_df)

    format_results_console(raw_df, "SCAN SORE — RAW RESULT")
    format_results_console(final_df, "FINAL CONFIRMED — MUNCUL DI SIANG DAN SORE")
    print(f"\nSaved session2 raw      : {raw_out}")
    print(f"Saved FINAL confirmed  : {final_out}")

    if TELEGRAM_OK:
        send_telegram_message(format_telegram_message(final_df, session="session2", total_scanned=len(tickers), final_mode=True))


def guess_session_from_time() -> str:
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).time()
    # Default sederhana: sebelum 14:30 dianggap sesi 1, setelah itu sesi 2.
    return "session1" if now < dt.time(14, 30) else "session2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BSJP dual-session noise-guard scanner")
    parser.add_argument("--session", choices=["auto", "session1", "session2"], default="auto", help="session1=watchlist siang, session2=konfirmasi sore")
    parser.add_argument("--tickers", default="data.csv", help="file CSV ticker, default data.csv")
    parser.add_argument("--no-intraday", action="store_true", help="pakai data harian saja; tidak direkomendasikan untuk dual-session")
    parser.add_argument("--no-cooldown", action="store_true", help="jangan skip ticker yang baru masuk final signal beberapa hari terakhir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = guess_session_from_time() if args.session == "auto" else args.session
    run_session(
        session=session,
        tickers_file=args.tickers,
        use_intraday=not args.no_intraday,
        skip_recent=not args.no_cooldown,
    )


if __name__ == "__main__":
    main()
