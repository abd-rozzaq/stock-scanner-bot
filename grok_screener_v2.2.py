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
import requests
from pathlib import Path
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING & PATH
# ======================================================
LOG_DIR = "logs"
CANDIDATES_DIR = "candidates"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CANDIDATES_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"scanner_v2.2.2_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GROK_SCANNER_V2.2.2")

# ======================================================
# 1. CONSTANTS
# ======================================================
VERSION = "2.2.2"

CLOSE_HIGH_RATIO       = 0.99
MIN_PRICE_CHANGE_PCT   = 4.0
MIN_VALUE_IDR          = 5_000_000_000
VOL_SPIKE_MULTIPLIER   = 3.0
MIN_CLOSE_PRICE        = 150
MIN_GREEN_BODY_PCT     = 2.0

RSI_PERIOD   = 14
MA20_PERIOD  = 20
MA50_PERIOD  = 50
RSI_MIN      = 55
RSI_MAX      = 75

MIN_EXTRA_SCORE = 4

TP_PERCENT = 0.06
CL_PERCENT = 0.05

MAX_MESSAGE_LENGTH = 4096
YFINANCE_PERIOD    = "3mo"
YFINANCE_INTERVAL  = "1d"

# ======================================================
# 2. TELEGRAM CONFIG — env var → config.py → fallback
# ======================================================
TELEGRAM_OK        = False
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    try:
        from config import TELEGRAM_BOT_TOKEN as _tok, TELEGRAM_CHAT_ID as _cid
        TELEGRAM_BOT_TOKEN = str(_tok).strip()
        TELEGRAM_CHAT_ID   = str(_cid).strip()
        logger.info("📄 Token dimuat dari config.py")
    except ImportError:
        logger.warning("⚠️  config.py tidak ditemukan")
    except Exception as e:
        logger.warning(f"⚠️  Error baca config.py: {e}")

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    TELEGRAM_OK = True
    logger.info(f"✅ Telegram ready | chat_id={TELEGRAM_CHAT_ID}")
else:
    logger.warning("⚠️  Telegram TIDAK AKTIF — token/chat_id kosong")

# ======================================================
# 3. JSON SANITIZER
# ======================================================
def make_json_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ======================================================
# 4. TELEGRAM HELPERS
# ======================================================
def split_telegram_message(message: str) -> List[str]:
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks, current = [], ""
    for line in message.split("\n"):
        test = current + line + "\n"
        if len(test) > MAX_MESSAGE_LENGTH:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current = test
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [message[:MAX_MESSAGE_LENGTH]]


def send_telegram_message(message: str, retries: int = 3) -> bool:
    """Kirim pesan ke Telegram dengan retry otomatis."""
    if not TELEGRAM_OK:
        logger.warning("🔕 Telegram tidak aktif — pesan tidak dikirim.")
        return False

    chunks      = split_telegram_message(message)
    url         = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    all_success = True

    for idx, chunk in enumerate(chunks, start=1):
        sent = False
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"},
                    timeout=15
                )
                if resp.status_code == 200:
                    logger.info(f"✅ Telegram terkirim chunk {idx}/{len(chunks)}")
                    sent = True
                    break
                else:
                    body = resp.json()
                    logger.error(
                        f"❌ Telegram HTTP {resp.status_code} "
                        f"(attempt {attempt}/{retries}): {body.get('description', '')}"
                    )
            except requests.exceptions.ConnectionError:
                logger.error(f"❌ Telegram: gagal koneksi (attempt {attempt}/{retries})")
            except requests.exceptions.Timeout:
                logger.error(f"❌ Telegram: timeout (attempt {attempt}/{retries})")
            except Exception as e:
                logger.error(f"❌ Telegram error: {e}")

            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s

        if not sent:
            logger.error(f"❌ Chunk {idx}/{len(chunks)} GAGAL setelah {retries} percobaan")
            all_success = False

        if len(chunks) > 1 and idx < len(chunks):
            time.sleep(1)

    return all_success


def build_telegram_message(session: str, scan_time_str: str,
                            confirmed: List[Dict], total_scanned: int) -> str:
    """
    Bangun pesan Telegram.
    SELALU dikirim — baik ada hasil maupun tidak.
    """
    if confirmed:
        header = (
            f"<b>🔍 GROK SCREENER V{VERSION} — {session.upper()} CONFIRMED</b>\n"
            f"📅 {scan_time_str}\n"
            f"✅ <b>{len(confirmed)} saham lolos</b> dari {total_scanned} discan\n\n"
        )
        body = ""
        for r in confirmed[:15]:
            body += (
                f"<b>{r['ticker']}</b>  "
                f"+{r['change_pct']:.1f}%  "
                f"RSI:{r['rsi']:.0f}  "
                f"Val:{r['value_b']:.2f}B\n"
                f"   🎯 Entry:{r['entry']:,} → TP:{r['tp']:,} | CL:{r['cl']:,}\n"
            )
        if len(confirmed) > 15:
            body += f"\n<i>...dan {len(confirmed) - 15} saham lainnya</i>\n"
        return header + body
    else:
        # ← FIX UTAMA: tetap kirim notifikasi meski tidak ada hasil
        return (
            f"<b>🔍 GROK SCREENER V{VERSION} — {session.upper()}</b>\n"
            f"📅 {scan_time_str}\n"
            f"📊 {total_scanned} saham discan\n\n"
            f"❌ <b>Tidak ada sinyal hari ini.</b>\n"
            f"<i>Semua saham tidak memenuhi kriteria filter.</i>"
        )

# ======================================================
# 5. DATA HELPERS
# ======================================================
def load_tickers_from_csv() -> List[str]:
    file_path = os.path.join("data", "data.csv")
    if not os.path.exists(file_path) and os.path.exists("data.csv"):
        file_path = "data.csv"
    try:
        df  = pd.read_csv(file_path)
        col = next(
            (c for c in ['Ticker', 'ticker', 'Kode', 'kode', 'Code', 'code', 'Symbol', 'symbol']
             if c in df.columns),
            df.columns[0]
        )
        tickers = df[col].dropna().astype(str).str.strip().str.upper().tolist()
        cleaned = sorted(set(t for t in tickers if len(t) >= 4))
        logger.info(f"📋 Loaded {len(cleaned)} tickers dari {file_path}")
        return cleaned
    except FileNotFoundError:
        logger.error(f"❌ File ticker tidak ditemukan: {file_path}")
        return []
    except Exception as e:
        logger.error(f"❌ Error load tickers: {e}")
        return []


def get_candidates_file(date_str: str, session: str) -> Path:
    return Path(CANDIDATES_DIR) / f"candidates_{date_str}_{session}.json"

# ======================================================
# 6. FETCH + RSI
# ======================================================
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(
            symbol,
            period=YFINANCE_PERIOD,
            interval=YFINANCE_INTERVAL,
            progress=False,
            auto_adjust=False
        )
        if df.empty or len(df) < MA50_PERIOD + 1:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.debug(f"Fetch error {ticker}: {e}")
        return pd.DataFrame()


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    try:
        if len(series) < period + 1:
            return -1.0
        delta    = series.diff()
        gain     = delta.where(delta > 0, 0.0)
        loss     = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        cur_gain = avg_gain.iloc[-1]
        cur_loss = avg_loss.iloc[-1]
        if cur_loss == 0:
            return 100.0 if cur_gain > 0 else 50.0
        return round(100.0 - (100.0 / (1.0 + cur_gain / cur_loss)), 1)
    except Exception:
        return -1.0

# ======================================================
# 7. ANALYZE STOCK
# ======================================================
def analyze_stock(ticker: str) -> Optional[Dict]:
    df = fetch_stock_data(ticker)
    if df.empty:
        return None

    current = df.iloc[-1]
    if current['Close'] < MIN_CLOSE_PRICE or current['Volume'] <= 0:
        return None

    prev_close       = df['Close'].iloc[-2]
    price_change_pct = ((current['Close'] - prev_close) / prev_close) * 100

    cond_close_high = bool(current['Close'] >= current['High'] * CLOSE_HIGH_RATIO)
    cond_change     = bool(price_change_pct >= MIN_PRICE_CHANGE_PCT)
    cond_green      = bool(
        ((current['Close'] - current['Open']) / current['Open']) * 100 >= MIN_GREEN_BODY_PCT
    )

    ma20            = df['Close'].rolling(MA20_PERIOD).mean().iloc[-1]
    ma50            = df['Close'].rolling(MA50_PERIOD).mean().iloc[-1]
    cond_ma20       = bool(not pd.isna(ma20) and current['Close'] > ma20)
    cond_ma50       = bool(not pd.isna(ma50) and current['Close'] > ma50)
    cond_ma_trend   = bool(cond_ma20 and cond_ma50 and ma20 > ma50)

    current_vol     = float(current['Volume'])
    value_today     = current['Close'] * current_vol
    vol_ma20        = df['Volume'].rolling(MA20_PERIOD).mean().iloc[-1]
    cond_vol_spike  = bool(
        not pd.isna(vol_ma20) and vol_ma20 > 0 and
        current_vol >= VOL_SPIKE_MULTIPLIER * vol_ma20
    )
    cond_value      = bool(value_today >= MIN_VALUE_IDR)

    value_series    = df['Close'] * df['Volume']
    value_ma20      = value_series.rolling(MA20_PERIOD).mean().iloc[-1]
    cond_value_str  = bool(not pd.isna(value_ma20) and value_today > value_ma20)

    rsi_value       = calculate_rsi(df['Close'])
    cond_rsi        = bool(RSI_MIN <= rsi_value <= RSI_MAX)

    if not all([cond_close_high, cond_change, cond_green, cond_ma_trend,
                cond_vol_spike, cond_value, cond_value_str, cond_rsi]):
        return None

    extra_score = sum([cond_rsi, cond_value_str, cond_ma_trend, cond_vol_spike])
    if extra_score < MIN_EXTRA_SCORE:
        return None

    entry = float(current['Close'])
    return {
        "ticker":      ticker,
        "close":       int(current['Close']),
        "change_pct":  round(float(price_change_pct), 2),
        "rsi":         float(rsi_value),
        "value_b":     round(float(value_today) / 1e9, 2),
        "ma20":        round(float(ma20), 0),
        "ma50":        round(float(ma50), 0),
        "vol_spike":   cond_vol_spike,
        "extra_score": int(extra_score),
        "entry":       int(entry),
        "tp":          int(entry * (1 + TP_PERCENT)),
        "cl":          int(entry * (1 - CL_PERCENT)),
        "status":      "MATCH"
    }

# ======================================================
# 8. SAVE CANDIDATES
# ======================================================
def save_candidates(date_str: str, session: str, results: List[Dict]):
    try:
        file = get_candidates_file(date_str, session)
        data = {
            "tickers": [r["ticker"] for r in results],
            "full":    [make_json_serializable(r) for r in results]
        }
        file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Saved {len(results)} candidates → {file}")
    except Exception as e:
        logger.error(f"❌ Error saving candidates: {e}")

# ======================================================
# 9. RUN SCANNER
# ======================================================
def run_scanner(session: str = "eod"):
    scan_start    = dt.datetime.now()
    date_str      = scan_start.strftime("%Y-%m-%d")
    scan_time_str = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"🚀 GROK SCANNER V{VERSION} — SESSION: {session.upper()} | {scan_time_str}")

    tickers = load_tickers_from_csv()
    if not tickers:
        logger.error("❌ Tidak ada ticker — scanner dihentikan.")
        return

    results: List[Dict] = []
    for i, ticker in enumerate(tickers):
        print(f"\r[{(i + 1) / len(tickers) * 100:5.1f}%] Scanning {ticker:<6}", end="", flush=True)
        res = analyze_stock(ticker)
        if res:
            results.append(res)
            logger.info(
                f"HIT {ticker} | +{res['change_pct']:.1f}% "
                f"| RSI:{res['rsi']:.1f} | Val:{res['value_b']}B"
            )

    print("\r" + " " * 80 + "\r", end="")

    # Dual-session: sore hanya ambil yang juga muncul di siang
    confirmed = results
    if session == "sore":
        siang_file = get_candidates_file(date_str, "siang")
        if siang_file.exists():
            try:
                siang_data    = json.loads(siang_file.read_text(encoding="utf-8"))
                siang_tickers = set(siang_data["tickers"])
                confirmed     = [r for r in results if r["ticker"] in siang_tickers]
                logger.info(f"✅ CONFIRMED siang+sore: {len(confirmed)} saham")
            except Exception as e:
                logger.error(f"❌ Error load siang candidates: {e}")
        else:
            logger.warning("⚠️  File siang tidak ditemukan — tampilkan semua hasil sore.")

    if session in ["siang", "sore"]:
        save_candidates(date_str, session, results)

    confirmed.sort(key=lambda x: (-x['extra_score'], -x['change_pct']))

    # ── Ringkasan console ──────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"GROK SCANNER V{VERSION} — {session.upper()} | {len(confirmed)} CONFIRMED")
    print("=" * 80)
    if confirmed:
        for r in confirmed:
            print(
                f"✅ {r['ticker']:6} | +{r['change_pct']:5.1f}% "
                f"| RSI:{r['rsi']:5.1f} | Val:{r['value_b']:6.2f}B "
                f"| Entry:{r['entry']:,} → TP:{r['tp']:,} | CL:{r['cl']:,}"
            )
    else:
        print("❌ Tidak ada sinyal CONFIRMED hari ini.")

    # ── Kirim Telegram — SELALU kirim, ada hasil atau tidak ──
    if TELEGRAM_OK:
        msg = build_telegram_message(session, scan_time_str, confirmed, len(tickers))
        send_telegram_message(msg)
    else:
        logger.warning("🔕 Telegram tidak aktif — hasil tidak dikirim.")

    elapsed = (dt.datetime.now() - scan_start).total_seconds()
    logger.info(f"⏱️  Selesai dalam {elapsed:.1f} detik")
    print(f"⏱️  Selesai dalam {elapsed:.1f} detik")

# ======================================================
# 10. MAIN
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"GROK SCREENER V{VERSION}")
    parser.add_argument(
        "--session",
        choices=["siang", "sore", "eod"],
        default="eod",
        help="Sesi scan: siang / sore / eod"
    )
    args = parser.parse_args()
    run_scanner(session=args.session)