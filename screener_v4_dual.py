"""
Claude Screener V4.1 — Dual Session Edition
============================================================
Cara pakai:
  python screener_v4_dual.py          → auto-detect session by waktu
  python screener_v4_dual.py sesi1    → paksa mode Sesi 1 (watchlist)
  python screener_v4_dual.py sesi2    → paksa mode Sesi 2 (konfirmasi)

Flow Dual Session:
  11:30-12:30 WIB → Sesi 1: tangkap kandidat, simpan ke JSON
  14:30-14:55 WIB → Sesi 2: konfirmasi, bandingkan dgn Sesi 1

  ⭐ CONFIRMED = muncul di KEDUA sesi → prioritas entry (sinyal terkuat)
  🆕 NEW       = hanya Sesi 2 → valid, tapi tidak ada konfirmasi Sesi 1
  ⚠️  DROPPED  = ada di Sesi 1, hilang di Sesi 2 → hindari, kemungkinan
                 distribusi bandar saat break / harga melemah di sesi 2
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import json
import sys
import os
import logging
import time

# ── Timezone WIB (UTC+7) ──────────────────────────────
try:
    from zoneinfo import ZoneInfo
    WIB = ZoneInfo("Asia/Jakarta")
except ImportError:
    try:
        import pytz
        WIB = pytz.timezone("Asia/Jakarta")
    except ImportError:
        import datetime as _dt
        WIB = _dt.timezone(_dt.timedelta(hours=7))

def now_wib() -> dt.datetime:
    """Kembalikan waktu sekarang dalam timezone WIB."""
    return dt.datetime.now(tz=WIB)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING SETUP
# ======================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
  f"scanner_{now_wib().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DUAL_SCREENER")

# ======================================================
# 1. CONSTANTS — BASE (tidak berubah antar sesi)
# ======================================================
VERSION = "4.1.0"

# ── Market Schedule IDX (WIB) ──────────────────────────
MARKET_SESI1_OPEN_H  = 9
MARKET_SESI1_OPEN_M  = 0
MARKET_SESI1_CLOSE_H = 11
MARKET_SESI1_CLOSE_M = 30    # Sesi 1: 09:00 - 11:30 → 150 menit
MARKET_SESI2_OPEN_H  = 13
MARKET_SESI2_OPEN_M  = 30
MARKET_SESI2_CLOSE_H = 15
MARKET_SESI2_CLOSE_M = 0     # Sesi 2: 13:30 - 15:00 → 90 menit
TOTAL_MARKET_MINUTES = 240.0  # 150 + 90

# ── Window auto-deteksi sesi (WIB) ────────────────────
SESI1_WINDOW_START   = dt.time(11, 30)   # setelah Sesi 1 tutup
SESI1_WINDOW_END     = dt.time(13, 30)   # sebelum Sesi 2 buka
SESI2_WINDOW_START   = dt.time(13, 30)   # Sesi 2 mulai
SESI2_WINDOW_END     = dt.time(15, 30)   # buffer 30 menit setelah tutup

# ── Technical Periods ─────────────────────────────────
RSI_PERIOD          = 14
MA5_PERIOD          = 5
MA20_PERIOD         = 20
MA50_PERIOD         = 50
CMF_PERIOD          = 14
OBV_SLOPE_PERIOD    = 5
AD_SLOPE_PERIOD     = 5

# ── Risk Management ───────────────────────────────────
TP_PERCENT          = 0.06     # 6% TP
CL_PERCENT          = 0.05     # 5% CL

# ── Data Fetch ────────────────────────────────────────
YFINANCE_HIST_PERIOD    = "4mo"    # historical daily (untuk MA50 + buffer)
YFINANCE_HIST_INTERVAL  = "1d"
YFINANCE_INTRA_PERIOD   = "1d"     # intraday hari ini
YFINANCE_INTRA_INTERVAL = "5m"     # resolusi 5 menit

# ── Bandar Screener ───────────────────────────────────
BANDAR_VALUE_MA20_MIN   = 1_000_000_000    # Value MA20 > 1 Miliar

# ── Likuiditas ────────────────────────────────────────
MIN_VALUE_LIKUIDITAS    = 100_000_000      # 100 Juta

# ── Session Data Storage ──────────────────────────────
SESSION_DATA_DIR    = os.path.join("data", "sessions")

# ── Telegram ─────────────────────────────────────────
MAX_MESSAGE_LENGTH  = 4096

# ======================================================
# 2. SESSION PROFILES
# ── Threshold berbeda per sesi: Sesi 1 lebih longgar
#    (candle & volume belum final), Sesi 2 pakai V4 penuh
# ======================================================
SESSION_PROFILES = {
    "SESI1": {
        # Label
        "label":                "SESI 1 — Watchlist Kandidat (11:30–12:30)",
        "description":          "Filter longgar, tujuan: tangkap kandidat potensial seluas mungkin",

        # M1: Close dekat High — lebih longgar krn candle bisa berubah di sesi 2
        "close_high_ratio":     0.975,

        # M4: Min kenaikan — lebih rendah krn hari belum selesai
        "min_price_change_pct": 2.5,

        # M2: Min frekuensi
        "min_frequency":        1_500,

        # M5: RSI — lebih lebar untuk kandidat
        "rsi_min":              35,
        "rsi_max":              75,

        # M7: Candle body — lebih longgar
        "min_candle_body":      0.25,

        # N1: Min value — lebih rendah
        "min_value_idr":        1_500_000_000,   # 1.5 Miliar

        # Noise filter — cukup 2/3
        "min_noise_score":      2,

        # B1: CMF minimum — lebih longgar
        "cmf_min":              0.02,

        # Anti-pump — lebih longgar (kandidat boleh sedikit sudah naik)
        "max_prerun_5d":        20.0,
        "max_prerun_10d":       38.0,

        # Volume projection — Sesi 1 volume belum final, proyeksikan ke full day
        "use_projected_vol":    True,

        # Vol spike multiplier
        "vol_spike_mult":       1.5,             # lebih longgar dari 2x
    },

    "SESI2": {
        # Label
        "label":                "SESI 2 — Sinyal Konfirmasi (14:30–14:55)",
        "description":          "Filter V4 penuh, tujuan: konfirmasi sinyal terkuat sebelum entry",

        # M1: Close dekat High — ketat (V4 original)
        "close_high_ratio":     0.985,

        # M4: Min kenaikan
        "min_price_change_pct": 3.0,

        # M2: Min frekuensi
        "min_frequency":        2_000,

        # M5: RSI — ketat
        "rsi_min":              40,
        "rsi_max":              72,

        # M7: Candle body — ketat
        "min_candle_body":      0.35,

        # N1: Min value — ketat
        "min_value_idr":        3_000_000_000,   # 3 Miliar

        # Noise filter — semua 3/3 wajib
        "min_noise_score":      3,

        # B1: CMF minimum — ketat
        "cmf_min":              0.05,

        # Anti-pump — ketat (V4 original)
        "max_prerun_5d":        15.0,
        "max_prerun_10d":       30.0,

        # Volume tidak diproyeksi (hampir final menjelang closing)
        "use_projected_vol":    False,

        # Vol spike multiplier
        "vol_spike_mult":       2.0,
    }
}

# ── Filter bersama (tidak berubah di kedua sesi) ──────
SHARED_FILTERS = {
    "min_price_idr":    50,             # min harga saham (anti penny)
    "vol_ma20_min":     2.0,            # Volume MA20 multiplier (trend screener)
    "max_vol_proj_cap": 3.0,            # cap proyeksi volume (max 3x lipat)
}

# ======================================================
# 3. TELEGRAM CONFIG
# ======================================================
TELEGRAM_OK         = False
TELEGRAM_BOT_TOKEN  = ""
TELEGRAM_CHAT_ID    = ""

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if REQUESTS_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        TELEGRAM_OK = True
        logger.info(f"Telegram Config Loaded. Chat ID: {TELEGRAM_CHAT_ID}")
    else:
        logger.warning("Telegram config ditemukan tapi token/chat_id kosong.")
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram dinonaktifkan.")
except Exception as e:
    logger.warning(f"Error loading config.py: {e}")

# ======================================================
# 4. TELEGRAM HELPERS
# ======================================================
def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        return False
    for i, chunk in enumerate(split_telegram_message(message)):
        try:
            url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            resp = requests.post(url, data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML"
            }, timeout=15)
            if resp.status_code != 200:
                logger.error(f"Telegram error {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
        if i > 0:
            time.sleep(1)
    return True


def split_telegram_message(message: str) -> List[str]:
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks, current = [], ""
    for line in message.split("\n"):
        test = current + line + "\n"
        if len(test) > MAX_MESSAGE_LENGTH:
            if current:
                chunks.append(current.strip())
            current = line + "\n"
        else:
            current = test
    if current.strip():
        chunks.append(current.strip())
    return chunks or [message[:MAX_MESSAGE_LENGTH]]

# ======================================================
# 5. DATA HELPERS
# ======================================================
def load_tickers_from_csv() -> List[str]:
    for path in [os.path.join("data", "data.csv"), "data.csv"]:
        if os.path.exists(path):
            try:
                df  = pd.read_csv(path)
                col = next(
                    (c for c in ['Ticker','ticker','Kode','kode','Code','code','Symbol','symbol']
                     if c in df.columns),
                    df.columns[0]
                )
                logger.info(f"Membaca '{path}' (kolom: {col})")
                tickers = df[col].dropna().astype(str).tolist()
                cleaned = sorted(set(t.strip().upper() for t in tickers if len(t.strip()) >= 4))
                logger.info(f"Total ticker unik: {len(cleaned)}")
                return cleaned
            except Exception as e:
                logger.error(f"Error membaca '{path}': {e}")
                return []
    logger.error("File data.csv tidak ditemukan.")
    return []


def load_blacklist() -> List[str]:
    path = os.path.join("data", "blacklist.csv")
    if not os.path.exists(path):
        return []
    try:
        df  = pd.read_csv(path)
        col = next(
            (c for c in ['Ticker','ticker','Kode','kode','Code','code','Symbol','symbol']
             if c in df.columns),
            df.columns[0]
        )
        cleaned = list(set(t.strip().upper() for t in df[col].dropna().astype(str).tolist()))
        logger.info(f"Blacklist aktif: {len(cleaned)} saham")
        return cleaned
    except Exception as e:
        logger.warning(f"Error membaca blacklist: {e}")
        return []


def fetch_stock_data_historical(ticker: str) -> pd.DataFrame:
    """
    Ambil data harian historis (4 bulan) dari Yahoo Finance.
    Digunakan untuk semua kalkulasi MA, CMF, OBV, A/D, RSI, anti-pump.
    """
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(
            symbol,
            period=YFINANCE_HIST_PERIOD,
            interval=YFINANCE_HIST_INTERVAL,
            progress=False,
            auto_adjust=False
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in cols):
            return pd.DataFrame()
        return df[cols].dropna().copy()
    except Exception as e:
        logger.debug(f"Error fetching historical {ticker}: {e}")
        return pd.DataFrame()


def get_elapsed_market_minutes(t: dt.time) -> float:
    """
    Hitung berapa menit sesi market sudah berjalan hingga waktu t (WIB).

    Jadwal IDX:
      Sesi 1: 09:00–11:30 (150 menit)
      Break:  11:30–13:30
      Sesi 2: 13:30–15:00 (90 menit)
      Total:  240 menit
    """
    h, m = t.hour, t.minute
    cur    = h * 60 + m
    s1_o   = MARKET_SESI1_OPEN_H  * 60 + MARKET_SESI1_OPEN_M   # 540
    s1_c   = MARKET_SESI1_CLOSE_H * 60 + MARKET_SESI1_CLOSE_M  # 690
    s2_o   = MARKET_SESI2_OPEN_H  * 60 + MARKET_SESI2_OPEN_M   # 810
    s2_c   = MARKET_SESI2_CLOSE_H * 60 + MARKET_SESI2_CLOSE_M  # 900

    if cur < s1_o:
        return 0.0
    elif cur <= s1_c:
        return float(cur - s1_o)       # dalam sesi 1
    elif cur < s2_o:
        return 150.0                    # break — sesi 1 sudah selesai
    elif cur <= s2_c:
        return 150.0 + float(cur - s2_o)  # dalam sesi 2
    else:
        return TOTAL_MARKET_MINUTES     # setelah market tutup


def fetch_intraday_today(ticker: str, use_projection: bool = True) -> Optional[Dict]:
    """
    Ambil data intraday hari ini (interval 5 menit) dari Yahoo Finance,
    lalu agregasikan ke satu bar OHLCV representatif.

    Mengapa intraday (bukan daily)?
    Ketika screener dijalankan SAAT market berjalan (11:45 atau 14:30),
    yfinance daily data hanya punya data sampai hari KEMARIN.
    Untuk sinyal hari ini, kita perlu fetch intraday lalu aggregate.

    Volume Projection (khusus Sesi 1):
    Saat 11:45 (break), hanya 150/240 menit market yg sudah berjalan.
    Volume sesungguhnya baru ~62.5% dari total hari.
    Proyeksi: volume_projected = volume_raw × (240 / 150) = ×1.6
    Ini agar perbandingan vs volume MA20 (daily) lebih fair.

    Returns dict atau None jika data tidak tersedia.
    """
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(
            symbol,
            period=YFINANCE_INTRA_PERIOD,
            interval=YFINANCE_INTRA_INTERVAL,
            progress=False,
            auto_adjust=False
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in cols):
            return None
        df.dropna(subset=["Close", "Volume"], inplace=True)
        if df.empty or len(df) < 3:
            return None

        # Aggregasi ke 1 bar
        today_open   = float(df["Open"].iloc[0])
        today_high   = float(df["High"].max())
        today_low    = float(df["Low"].min())
        today_close  = float(df["Close"].iloc[-1])
        today_vol    = float(df["Volume"].sum())

        # Volume projection
        if use_projection:
            now_time    = now_wib().time()
            elapsed_min = get_elapsed_market_minutes(now_time)
            if elapsed_min > 0:
                raw_mult = TOTAL_MARKET_MINUTES / elapsed_min
                mult     = min(raw_mult, SHARED_FILTERS["max_vol_proj_cap"])
            else:
                mult = 1.0
            proj_vol = today_vol * mult
        else:
            mult     = 1.0
            proj_vol = today_vol

        return {
            "open":             today_open,
            "high":             today_high,
            "low":              today_low,
            "close":            today_close,
            "volume_raw":       today_vol,
            "volume":           proj_vol,       # volume yg dipakai untuk filter
            "vol_multiplier":   round(mult, 2),
            "bars_count":       len(df),
            "source":           "INTRADAY",
        }
    except Exception as e:
        logger.debug(f"Error fetching intraday {ticker}: {e}")
        return None

# ======================================================
# 6. TECHNICAL INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """RSI Wilder's Smoothing (EWM)."""
    try:
        if len(series) < period + 1:
            return -1.0
        delta = series.diff()
        gain  = delta.where(delta > 0, 0.0)
        loss  = (-delta.where(delta < 0, 0.0))
        ag    = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        al    = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        cag, cal = ag.iloc[-1], al.iloc[-1]
        if cal == 0:
            return 100.0 if cag > 0 else 50.0
        return round(100.0 - (100.0 / (1.0 + cag / cal)), 1)
    except Exception:
        return -1.0


def compute_cmf(df: pd.DataFrame, period: int = CMF_PERIOD) -> pd.Series:
    """
    Chaikin Money Flow — mengukur tekanan beli vs jual berdasarkan
    posisi close dalam range hari itu (bukan sekadar volume).
    CMF > 0 = buying pressure / akumulasi.
    CMF < 0 = selling pressure / distribusi.
    """
    rng  = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm  = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    mfm  = mfm.fillna(0)
    mfv  = mfm * df["Volume"]
    vs   = df["Volume"].rolling(period).sum().replace(0, np.nan)
    cmf  = mfv.rolling(period).sum() / vs
    return cmf.fillna(0)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — akumulasi volume berdasarkan arah harga.
    OBV naik = lebih banyak volume di hari hijau = buying pressure.
    """
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def compute_ad_line(df: pd.DataFrame) -> pd.Series:
    """
    Accumulation/Distribution Line — lebih sensitif dari OBV.
    Mendeteksi distribusi tersembunyi: harga naik tapi A/D turun
    = tanda bandar distribusi diam-diam.
    """
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    clv = clv.fillna(0)
    return (clv * df["Volume"]).cumsum()


def get_slope_sign(series: pd.Series, lookback: int) -> int:
    """
    Cek tren series dalam N periode terakhir via linear regression.
    Return: +1 (naik), -1 (turun), 0 (tidak cukup data/flat).
    """
    if len(series) < lookback + 1:
        return 0
    recent = series.iloc[-lookback:].values.astype(float)
    if np.any(np.isnan(recent)):
        return 0
    try:
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        return 1 if slope > 0 else (-1 if slope < 0 else 0)
    except Exception:
        return 0

# ======================================================
# 7. SESSION MANAGEMENT
# ======================================================
def detect_session() -> str:
    """
    Auto-deteksi sesi berdasarkan waktu WIB sekarang.
    Return: "SESI1" | "SESI2" | "UNKNOWN"
    """
    now = now_wib().time()
    if SESI1_WINDOW_START <= now < SESI1_WINDOW_END:
        return "SESI1"
    elif SESI2_WINDOW_START <= now < SESI2_WINDOW_END:
        return "SESI2"
    return "UNKNOWN"


def parse_session_arg() -> str:
    """
    Baca argument command line untuk override session.
    Usage:
      python screener_v4_dual.py         → auto
      python screener_v4_dual.py sesi1   → paksa Sesi 1
      python screener_v4_dual.py sesi2   → paksa Sesi 2
    """
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().strip()
        if arg in ("sesi1", "1", "s1"):
            return "SESI1"
        if arg in ("sesi2", "2", "s2"):
            return "SESI2"
    return detect_session()


def get_today_str() -> str:
    return now_wib().strftime("%Y%m%d")


def save_session1_results(results: List[Dict]) -> bool:
    """
    Simpan hasil Sesi 1 ke JSON untuk dibandingkan saat Sesi 2.
    Path: data/sessions/sesi1_YYYYMMDD.json
    """
    os.makedirs(SESSION_DATA_DIR, exist_ok=True)
    path = os.path.join(SESSION_DATA_DIR, f"sesi1_{get_today_str()}.json")
    try:
        payload = {
            "date":      get_today_str(),
            "timestamp": now_wib().isoformat(),
            "version":   VERSION,
            "count":     len(results),
            "tickers":   [r["ticker"] for r in results],
            "results":   results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Sesi 1 tersimpan → {path} ({len(results)} saham)")
        return True
    except Exception as e:
        logger.error(f"Error menyimpan Sesi 1: {e}")
        return False


def load_session1_results() -> Optional[Dict]:
    """
    Muat hasil Sesi 1 hari ini.
    Return None jika tidak ada atau file rusak.
    """
    path = os.path.join(SESSION_DATA_DIR, f"sesi1_{get_today_str()}.json")
    if not os.path.exists(path):
        logger.warning(f"File Sesi 1 tidak ditemukan: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Sesi 1 dimuat → {len(data.get('results', []))} saham")
        return data
    except Exception as e:
        logger.error(f"Error memuat Sesi 1: {e}")
        return None


def compare_sessions(
    sesi1_tickers: set,
    sesi2_results: List[Dict]
) -> Dict:
    """
    Bandingkan hasil Sesi 1 dan Sesi 2.

    Returns dict berisi:
      confirmed → saham yg muncul di KEDUA sesi (sinyal terkuat)
      new       → hanya di Sesi 2 (valid tapi belum terkonfirmasi S1)
      dropped   → ada di Sesi 1, hilang di Sesi 2 (tanda kelemahan)
    """
    confirmed   = []
    new_signals = []

    for r in sesi2_results:
        if r["ticker"] in sesi1_tickers:
            r["confirmation"] = "CONFIRMED"
            confirmed.append(r)
        else:
            r["confirmation"] = "NEW"
            new_signals.append(r)

    sesi2_tickers = {r["ticker"] for r in sesi2_results}
    dropped       = sorted(sesi1_tickers - sesi2_tickers)

    return {
        "confirmed":  confirmed,
        "new":        new_signals,
        "dropped":    dropped,
    }

# ======================================================
# 8. CORE ANALYSIS — SESSION-AWARE
# ======================================================
def analyze_stock_dual(ticker: str, session: str, cfg: Dict) -> Optional[Dict]:
    """
    Analisis satu saham dengan threshold sesuai session profile.

    Alur data:
    ┌─────────────────────────────────────────────────────┐
    │  Historical daily (4mo)  →  semua MA, RSI, CMF,    │
    │                              OBV, A/D, anti-pump    │
    │                                                     │
    │  Intraday hari ini (5m)  →  harga & volume terkini  │
    │  (jika tersedia)            untuk sinyal hari ini   │
    │                             + proyeksi volume S1    │
    └─────────────────────────────────────────────────────┘

    Jika intraday tidak tersedia (market tutup, weekend,
    yfinance gagal), fallback ke bar terakhir historical.
    """
    # ── 1. Fetch historical daily ──────────────────────
    hist = fetch_stock_data_historical(ticker)
    if hist.empty or len(hist) < 65:
        return None

    # ── 2. Fetch intraday hari ini ─────────────────────
    use_proj    = cfg["use_projected_vol"]
    intra       = fetch_intraday_today(ticker, use_projection=use_proj)
    using_intra = False

    if intra is not None:
        # Gunakan data intraday sebagai bar "hari ini"
        close       = intra["close"]
        high        = intra["high"]
        low         = intra["low"]
        open_price  = intra["open"]
        volume      = intra["volume"]          # sudah diproyeksikan jika S1
        vol_raw     = intra["volume_raw"]
        vol_mult    = intra["vol_multiplier"]
        prev_close  = float(hist["Close"].iloc[-1])   # close kemarin
        df_for_ma   = hist                            # MA dihitung dari historical
        using_intra = True
        data_note   = (
            f"INTRADAY ({intra['bars_count']} bars"
            + (f", proj ×{vol_mult:.2f}" if use_proj and vol_mult > 1.01 else "")
            + ")"
        )
    else:
        # Fallback: gunakan 2 bar terakhir historical
        cur         = hist.iloc[-1]
        prv         = hist.iloc[-2]
        close       = float(cur["Close"])
        high        = float(cur["High"])
        low         = float(cur["Low"])
        open_price  = float(cur["Open"])
        volume      = float(cur["Volume"])
        vol_raw     = volume
        vol_mult    = 1.0
        prev_close  = float(prv["Close"])
        df_for_ma   = hist
        data_note   = "HISTORICAL_FALLBACK"

    # Validasi data dasar
    if close <= 0 or high <= 0 or volume <= 0 or prev_close <= 0:
        return None

    # ── 3. Derived series dari historical ─────────────
    value_series = df_for_ma["Close"] * df_for_ma["Volume"]
    value_today  = close * volume

    vol_ma5      = df_for_ma["Volume"].rolling(MA5_PERIOD).mean().iloc[-1]
    vol_ma20     = df_for_ma["Volume"].rolling(MA20_PERIOD).mean().iloc[-1]
    ma20         = df_for_ma["Close"].rolling(MA20_PERIOD).mean().iloc[-1]
    ma50         = df_for_ma["Close"].rolling(MA50_PERIOD).mean().iloc[-1]
    value_ma20   = value_series.rolling(MA20_PERIOD).mean().iloc[-1]

    if any(pd.isna(x) for x in [vol_ma5, vol_ma20, ma20, ma50, value_ma20]):
        return None

    price_change_pct = ((close - prev_close) / prev_close) * 100

    # ── 4. Anti-pump lookback ─────────────────────────
    if len(df_for_ma) < 12:
        return None

    price_5d_ago  = float(df_for_ma["Close"].iloc[-7])
    price_10d_ago = float(df_for_ma["Close"].iloc[-12])
    prior_run_5d  = ((prev_close - price_5d_ago)  / price_5d_ago)  * 100
    prior_run_10d = ((prev_close - price_10d_ago) / price_10d_ago) * 100

    # ── 5. Indikator Bandar ────────────────────────────
    cmf_series = compute_cmf(df_for_ma, CMF_PERIOD)
    obv_series = compute_obv(df_for_ma)
    ad_series  = compute_ad_line(df_for_ma)

    cmf_value  = float(cmf_series.iloc[-1])
    obv_slope  = get_slope_sign(obv_series, OBV_SLOPE_PERIOD)
    ad_slope   = get_slope_sign(ad_series,  AD_SLOPE_PERIOD)

    # RSI dari historical close
    rsi_value  = calculate_rsi(df_for_ma["Close"], RSI_PERIOD)

    # ── 6. Candle quality ─────────────────────────────
    candle_range = high - low
    candle_body  = close - open_price
    body_ratio   = (candle_body / candle_range) if candle_range > 0 else 0.0

    # ══════════════════════════════════════════════════
    # MAIN CONDITIONS (semua wajib) — session-aware
    # ══════════════════════════════════════════════════
    m1 = close >= (high * cfg["close_high_ratio"])
    m2 = volume > cfg["min_frequency"]
    m3 = volume > float(vol_ma5)
    m4 = price_change_pct >= cfg["min_price_change_pct"]
    m5 = (rsi_value >= cfg["rsi_min"]) and (rsi_value <= cfg["rsi_max"])
    m6 = close >= SHARED_FILTERS["min_price_idr"]
    m7 = (candle_body > 0) and (body_ratio >= cfg["min_candle_body"])
    m8 = prior_run_5d  <= cfg["max_prerun_5d"]
    m9 = prior_run_10d <= cfg["max_prerun_10d"]

    if not all([m1, m2, m3, m4, m5, m6, m7, m8, m9]):
        return None

    # ══════════════════════════════════════════════════
    # NOISE FILTERS — session-aware scoring
    # ══════════════════════════════════════════════════
    vol_mult_for_noise = cfg["vol_spike_mult"]
    n1 = value_today >= cfg["min_value_idr"]
    n2 = close > float(ma20)
    n3 = volume >= (vol_mult_for_noise * float(vol_ma20))

    noise_score = sum([n1, n2, n3])
    if noise_score < cfg["min_noise_score"]:
        return None

    # ══════════════════════════════════════════════════
    # BANDAR SCREENER — CMF + OBV + A/D + Value MA20
    # ══════════════════════════════════════════════════
    b1 = cmf_value > cfg["cmf_min"]    # buying pressure aktif
    b2 = obv_slope > 0                  # OBV tren naik
    b3 = ad_slope  > 0                  # A/D tren naik (bukan distribusi tersembunyi)
    b4 = float(value_ma20) > BANDAR_VALUE_MA20_MIN

    if not all([b1, b2, b3, b4]):
        return None

    # ══════════════════════════════════════════════════
    # TREND SCREENER
    # ══════════════════════════════════════════════════
    t1 = close > float(ma20)
    t2 = close > float(ma50)
    t3 = volume >= (SHARED_FILTERS["vol_ma20_min"] * float(vol_ma20))
    t4 = value_today > float(value_ma20)

    if not all([t1, t2, t3, t4]):
        return None

    # ══════════════════════════════════════════════════
    # LIKUIDITAS SCREENER
    # ══════════════════════════════════════════════════
    l1 = volume > (2 * float(vol_ma20))
    l2 = value_today >= MIN_VALUE_LIKUIDITAS

    if not all([l1, l2]):
        return None

    # ── Risk Management ───────────────────────────────
    tp = int(close * (1 + TP_PERCENT))
    cl = int(close * (1 - CL_PERCENT))

    return {
        "ticker":           ticker,
        "session":          session,
        "confirmation":     "-",        # diisi saat compare (Sesi 2)
        "data_source":      data_note if using_intra else "HIST_FALLBACK",

        "close":            int(close),
        "high":             int(high),
        "low":              int(low),
        "open":             int(open_price),
        "change_pct":       round(float(price_change_pct), 2),
        "volume":           int(volume),
        "vol_raw":          int(vol_raw),
        "vol_multiplier":   vol_mult,
        "vol_ma20":         int(vol_ma20),

        "rsi":              rsi_value if rsi_value >= 0 else 0.0,

        "value_b":          round(float(value_today) / 1e9, 3),
        "value_ma20_b":     round(float(value_ma20) / 1e9, 3),
        "ma20":             round(float(ma20), 0),
        "ma50":             round(float(ma50), 0),
        "above_ma20":       n2,
        "above_ma50":       t2,
        "vol_spike":        n3,
        "value_ok":         n1,
        "noise_score":      noise_score,

        "prior_run_5d":     round(prior_run_5d, 2),
        "prior_run_10d":    round(prior_run_10d, 2),
        "body_ratio":       round(body_ratio, 3),

        "cmf":              round(cmf_value, 4),
        "obv_slope":        obv_slope,
        "ad_slope":         ad_slope,
        "b1_cmf_ok":        b1,
        "b2_obv_up":        b2,
        "b3_ad_up":         b3,

        "entry":            int(close),
        "tp":               tp,
        "cl":               cl,
    }

# ======================================================
# 9. OUTPUT FORMATTERS
# ======================================================
def _confirmation_prefix(r: Dict) -> str:
    c = r.get("confirmation", "-")
    if c == "CONFIRMED":
        return "⭐"
    elif c == "NEW":
        return "🆕"
    return "  "


def format_terminal_table(results: List[Dict], show_confirmation: bool = False) -> str:
    if not results:
        return "  (tidak ada)"

    conf_col = "Conf" if show_confirmation else ""
    header = (
        f"  {'No':>3} | {'':2}{'Ticker':<6} | {'Close':>7} | {'Chg%':>6} | "
        f"{'RSI':>5} | {'CMF':>6} | {'Val(B)':>6} | "
        f"{'5dRun':>6} | {'Body':>5} | "
        f"{'MA20':>4} | {'MA50':>4} | "
        f"{'Entry':>7} | {'TP':>7} | {'CL':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    for i, r in enumerate(results, 1):
        pfx     = _confirmation_prefix(r)
        ma20_i  = "✓" if r["above_ma20"] else "✗"
        ma50_i  = "✓" if r["above_ma50"] else "✗"
        prerun  = f"{r['prior_run_5d']:+.1f}%"
        line = (
            f"  {i:>3} | {pfx}{r['ticker']:<6} | {r['close']:>7,} | "
            f"{r['change_pct']:>+5.1f}% | {r['rsi']:>5.1f} | "
            f"{r['cmf']:>+6.3f} | {r['value_b']:>5.3f}B | "
            f"{prerun:>6} | {r['body_ratio']:>5.3f} | "
            f"   {ma20_i} | "
            f"   {ma50_i} | "
            f"{r['entry']:>7,} | {r['tp']:>7,} | {r['cl']:>7,}"
        )
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)


def format_telegram_sesi1(results: List[Dict], scan_time: str,
                           total: int, skipped: int,
                           now_time: dt.time) -> str:
    elapsed = get_elapsed_market_minutes(now_time)
    proj_note = ""
    if elapsed > 0:
        mult = min(TOTAL_MARKET_MINUTES / elapsed, SHARED_FILTERS["max_vol_proj_cap"])
        proj_note = f" | Vol proj ×{mult:.2f}"

    lines = [
        f"<b>📋 Claude Screener V{VERSION} — SESI 1</b>",
        f"<i>Watchlist Kandidat | {scan_time}{proj_note}</i>",
        "",
        f"Scanned: {total} | Skipped: {skipped}",
        f"<b>Kandidat: {len(results)} saham</b>",
        "<i>⚠️ Belum sinyal final — jalankan Sesi 2 pukul 14:30 untuk konfirmasi</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not results:
        lines.append("<i>Tidak ada kandidat ditemukan.</i>")
        return "\n".join(lines)

    for r in results:
        ma20_i = "✅" if r["above_ma20"] else "❌"
        ma50_i = "✅" if r["above_ma50"] else "❌"
        lines.append(
            f"<b>{r['ticker']}</b> | {r['change_pct']:+.2f}% | "
            f"RSI:{r['rsi']:.0f} | CMF:{r['cmf']:+.3f} | Val:{r['value_b']:.3f}B"
        )
        lines.append(
            f"   Entry: {r['entry']:,} → TP:{r['tp']:,} | CL:{r['cl']:,}"
        )
        lines.append(
            f"   MA20:{ma20_i} MA50:{ma50_i} | "
            f"5dRun:{r['prior_run_5d']:+.1f}% | Body:{r['body_ratio']:.2f}"
        )
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"<i>TP: +{TP_PERCENT*100:.0f}% | CL: -{CL_PERCENT*100:.0f}%</i>")
    return "\n".join(lines)


def format_telegram_sesi2(
    confirmed: List[Dict],
    new: List[Dict],
    dropped: List[str],
    scan_time: str,
    total: int,
    skipped: int,
    has_sesi1: bool
) -> str:
    lines = [
        f"<b>🔔 Claude Screener V{VERSION} — SESI 2</b>",
        f"<i>Sinyal Konfirmasi | {scan_time}</i>",
        "",
        f"Scanned: {total} | Skipped: {skipped}",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # ── CONFIRMED ───────────────────────────────
    if confirmed:
        lines.append(
            f"<b>⭐ CONFIRMED — {len(confirmed)} saham</b>"
        )
        lines.append("<i>Muncul Sesi 1 &amp; Sesi 2 → PRIORITAS ENTRY</i>")
        lines.append("")
        for r in confirmed:
            ma20_i = "✅" if r["above_ma20"] else "❌"
            ma50_i = "✅" if r["above_ma50"] else "❌"
            lines.append(
                f"⭐ <b>{r['ticker']}</b> | {r['change_pct']:+.2f}% | "
                f"RSI:{r['rsi']:.0f} | CMF:{r['cmf']:+.3f} | Val:{r['value_b']:.3f}B"
            )
            lines.append(
                f"   Entry: {r['entry']:,} → TP:{r['tp']:,} | CL:{r['cl']:,}"
            )
            lines.append(
                f"   MA20:{ma20_i} MA50:{ma50_i} | "
                f"5dRun:{r['prior_run_5d']:+.1f}% | Body:{r['body_ratio']:.2f}"
            )
            cmf_i = "✅" if r["b1_cmf_ok"] else "❌"
            obv_i = "✅" if r["b2_obv_up"] else "❌"
            ad_i  = "✅" if r["b3_ad_up"]  else "❌"
            lines.append(f"   Bandar: CMF{cmf_i} OBV{obv_i} AD{ad_i}")
            lines.append("")
    else:
        lines.append("⭐ <b>CONFIRMED: tidak ada</b>")
        lines.append("")

    # ── NEW ─────────────────────────────────────
    if new:
        lines.append(f"<b>🆕 NEW — {len(new)} saham</b>")
        lines.append("<i>Hanya muncul Sesi 2 — valid, tapi lebih hati-hati</i>")
        lines.append("")
        for r in new:
            lines.append(
                f"🆕 <b>{r['ticker']}</b> | {r['change_pct']:+.2f}% | "
                f"RSI:{r['rsi']:.0f} | CMF:{r['cmf']:+.3f} | Val:{r['value_b']:.3f}B"
            )
            lines.append(
                f"   Entry: {r['entry']:,} → TP:{r['tp']:,} | CL:{r['cl']:,}"
            )
            lines.append("")
    else:
        lines.append("🆕 <b>NEW: tidak ada</b>")
        lines.append("")

    # ── DROPPED ─────────────────────────────────
    if has_sesi1:
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        if dropped:
            lines.append(
                f"<b>⚠️ DROPPED dari Sesi 1 ({len(dropped)} saham)</b>"
            )
            lines.append(
                "<i>Ada di Sesi 1, hilang di Sesi 2 → "
                "melemah/terdistribusi, hindari entry</i>"
            )
            lines.append(", ".join(dropped))
        else:
            lines.append("⚠️ <b>DROPPED: tidak ada</b>")
            lines.append("<i>Semua kandidat Sesi 1 masih terkonfirmasi</i>")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"<i>TP: +{TP_PERCENT*100:.0f}% | CL: -{CL_PERCENT*100:.0f}%</i>")
    return "\n".join(lines)


def print_sesi1_results(results: List[Dict], total: int, skipped: int,
                         blacklisted: int, elapsed_sec: float,
                         now_time: dt.time):
    elapsed_mkt = get_elapsed_market_minutes(now_time)
    proj_mult   = min(TOTAL_MARKET_MINUTES / elapsed_mkt,
                      SHARED_FILTERS["max_vol_proj_cap"]) if elapsed_mkt > 0 else 1.0

    print(f"\n{'='*78}")
    print(f"  📋 SESI 1 — WATCHLIST KANDIDAT")
    print(f"  Scanned: {total} | Blacklisted: {blacklisted} | Skipped: {skipped}")
    print(f"  Market elapsed: {elapsed_mkt:.0f}/{TOTAL_MARKET_MINUTES:.0f} menit"
          f" | Vol projection: ×{proj_mult:.2f}")
    print(f"  Kandidat ditemukan: {len(results)} saham")
    print(f"{'='*78}")

    if results:
        print(format_terminal_table(results, show_confirmation=False))
    else:
        print("\n  (Tidak ada kandidat ditemukan dengan filter Sesi 1)")
        print("  Normal — filter masih cukup ketat meski lebih longgar dari Sesi 2.")

    print(f"\n{'='*78}")
    print(f"  ⏱  Selesai dalam {elapsed_sec:.1f} detik")
    print(f"  💾 Hasil disimpan → data/sessions/sesi1_{get_today_str()}.json")
    print(f"  ⏳ Jalankan SESI 2 pukul 14:30–14:55 WIB untuk konfirmasi")
    print(f"{'='*78}\n")


def print_sesi2_results(comparison: Dict, total: int, skipped: int,
                         blacklisted: int, elapsed_sec: float,
                         has_sesi1: bool):
    confirmed = comparison["confirmed"]
    new       = comparison["new"]
    dropped   = comparison["dropped"]
    all_s2    = confirmed + new

    print(f"\n{'='*78}")
    print(f"  🔔 SESI 2 — SINYAL KONFIRMASI")
    print(f"  Scanned: {total} | Blacklisted: {blacklisted} | Skipped: {skipped}")
    print(f"  Total sinyal Sesi 2: {len(all_s2)} saham")
    if has_sesi1:
        print(f"  ⭐ CONFIRMED (Sesi 1 + Sesi 2): {len(confirmed)} saham")
        print(f"  🆕 NEW (hanya Sesi 2)          : {len(new)} saham")
        print(f"  ⚠️  DROPPED (S1 hilang di S2)  : {len(dropped)} saham")
    print(f"{'='*78}")

    if confirmed:
        print(f"\n  ⭐ CONFIRMED — PRIORITAS ENTRY (muncul di kedua sesi)")
        print(format_terminal_table(confirmed, show_confirmation=True))

    if new:
        print(f"\n  🆕 NEW — Sinyal Baru Sesi 2 (perlu lebih hati-hati)")
        print(format_terminal_table(new, show_confirmation=True))

    if not confirmed and not new:
        print("\n  (Tidak ada sinyal yang lolos filter Sesi 2)")

    if has_sesi1 and dropped:
        print(f"\n  ⚠️  DROPPED — ada di Sesi 1, HILANG di Sesi 2 (hindari entry):")
        print(f"  {'  '.join(dropped)}")
        print(f"  → Kemungkinan: harga melemah sesi 2, distribusi bandar saat break,")
        print(f"    atau volume tidak terkonfirmasi. Jangan kejar saham ini.")

    print(f"\n{'='*78}")
    print(f"  ⏱  Selesai dalam {elapsed_sec:.1f} detik")
    print(f"{'='*78}\n")

# ======================================================
# 10. MAIN — DUAL SESSION SCANNER
# ======================================================
def run_scanner_dual():
    scan_start   = now_wib()
    scan_time    = scan_start.strftime("%Y-%m-%d %H:%M:%S")
    now_time     = scan_start.time()

    # ── Deteksi / baca session mode ───────────────────
    session = parse_session_arg()

    # Handle UNKNOWN (di luar jam)
    if session == "UNKNOWN":
        now_str = scan_start.strftime("%H:%M")
        print(f"\n⚠️  Waktu sekarang ({now_str} WIB) di luar window sesi yang diketahui.")
        print(f"   Sesi 1 window: {SESI1_WINDOW_START.strftime('%H:%M')} – "
              f"{SESI1_WINDOW_END.strftime('%H:%M')} WIB")
        print(f"   Sesi 2 window: {SESI2_WINDOW_START.strftime('%H:%M')} – "
              f"{SESI2_WINDOW_END.strftime('%H:%M')} WIB")
        print(f"\n   Override manual:")
        print(f"   python screener_v4_dual.py sesi1   (paksa Sesi 1)")
        print(f"   python screener_v4_dual.py sesi2   (paksa Sesi 2)")
        return

    cfg = SESSION_PROFILES[session]

    # ── Header ────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"  Claude Screener V{VERSION} — Dual Session Edition")
    print(f"  Mode: {cfg['label']}")
    print(f"  {scan_time}")
    print("=" * 78)
    print(f"\n  {cfg['description']}")
    print(f"\n  Threshold aktif:")
    print(f"    Close/High   : ≥ {cfg['close_high_ratio']}")
    print(f"    RSI          : {cfg['rsi_min']}–{cfg['rsi_max']}")
    print(f"    Min Value    : {cfg['min_value_idr']/1e9:.1f} Miliar IDR")
    print(f"    CMF min      : {cfg['cmf_min']:+.2f}")
    print(f"    Candle body  : ≥ {cfg['min_candle_body']}")
    print(f"    Noise filter : {cfg['min_noise_score']}/3 wajib")
    print(f"    Anti-pump    : 5d ≤ {cfg['max_prerun_5d']}%, 10d ≤ {cfg['max_prerun_10d']}%")
    print(f"    Vol proj     : {'Ya' if cfg['use_projected_vol'] else 'Tidak'}")

    if session == "SESI1":
        elapsed_mkt = get_elapsed_market_minutes(now_time)
        if elapsed_mkt > 0:
            mult = min(TOTAL_MARKET_MINUTES / elapsed_mkt,
                       SHARED_FILTERS["max_vol_proj_cap"])
            print(f"    Vol mult     : ×{mult:.2f} "
                  f"({elapsed_mkt:.0f}/{TOTAL_MARKET_MINUTES:.0f} menit elapsed)")
    print()

    logger.info(f"Screener V{VERSION} mode={session} dimulai.")

    # ── Load Sesi 1 data (hanya untuk Sesi 2) ────────
    sesi1_data    = None
    sesi1_tickers = set()
    has_sesi1     = False

    if session == "SESI2":
        sesi1_data = load_session1_results()
        if sesi1_data:
            sesi1_tickers = set(sesi1_data.get("tickers", []))
            has_sesi1     = True
            ts_s1 = sesi1_data.get("timestamp", "?")
            print(f"  📂 Data Sesi 1 dimuat: {len(sesi1_tickers)} kandidat"
                  f" (disimpan {ts_s1[:16]})")
            print(f"  Kandidat Sesi 1: {', '.join(sorted(sesi1_tickers))}")
        else:
            print("  ℹ️  Data Sesi 1 tidak ditemukan — Sesi 2 berjalan tanpa perbandingan")
            print("       (semua sinyal akan ditandai 🆕 NEW)")
        print()

    # ── Load Tickers ──────────────────────────────────
    tickers = load_tickers_from_csv()
    if not tickers:
        logger.error("Tidak ada saham untuk di-scan.")
        return

    total_scanned = len(tickers)

    # ── Load Blacklist ────────────────────────────────
    blacklist         = load_blacklist()
    blacklisted_count = 0
    if blacklist:
        orig    = len(tickers)
        tickers = [t for t in tickers if t not in blacklist]
        blacklisted_count = orig - len(tickers)
        if blacklisted_count:
            logger.info(f"{blacklisted_count} saham di-skip (blacklist)")

    # ── Scanning ─────────────────────────────────────
    results = []
    skipped = 0
    print(f"  Memulai scanning {len(tickers)} saham...\n")

    for i, ticker in enumerate(tickers):
        pct = (i + 1) / len(tickers) * 100
        print(f"\r  [{pct:5.1f}%] {i+1}/{len(tickers)}: {ticker:<6}", end="", flush=True)

        try:
            res = analyze_stock_dual(ticker, session, cfg)
            if res:
                # Pre-label untuk Sesi 2 (kalau ada data S1)
                if session == "SESI2" and has_sesi1:
                    res["confirmation"] = (
                        "CONFIRMED" if ticker in sesi1_tickers else "NEW"
                    )
                conf_str = f" [{res.get('confirmation','-')}]" if session == "SESI2" else ""
                print(
                    f"\n  ✅ HIT: {ticker}{conf_str} "
                    f"({res['change_pct']:+.2f}%, RSI:{res['rsi']:.0f}, "
                    f"CMF:{res['cmf']:+.3f}, Body:{res['body_ratio']:.2f})"
                )
                logger.info(
                    f"HIT[{session}]: {ticker}{conf_str} | "
                    f"Chg:{res['change_pct']:+.2f}% | RSI:{res['rsi']:.1f} | "
                    f"CMF:{res['cmf']:+.3f} | Val:{res['value_b']:.3f}B | "
                    f"5dRun:{res['prior_run_5d']:+.1f}%"
                )
                results.append(res)
        except KeyboardInterrupt:
            print("\n\n⚠️  Scanner dihentikan.")
            break
        except Exception as e:
            logger.debug(f"Error {ticker}: {e}")
            skipped += 1

    print("\r" + " " * 78 + "\r", end="")

    # ── Sort: CONFIRMED dulu, lalu CMF DESC ──────────
    def sort_key(r):
        conf_order = 0 if r.get("confirmation") == "CONFIRMED" else 1
        return (conf_order, -r["cmf"], -r["change_pct"])

    results.sort(key=sort_key)

    elapsed_sec = (now_wib() - scan_start).total_seconds()

    # ══════════════════════════════════════════════════
    # OUTPUT per sesi
    # ══════════════════════════════════════════════════
    if session == "SESI1":
        # ── Sesi 1: simpan + tampilkan watchlist ──────
        print_sesi1_results(results, total_scanned, skipped,
                             blacklisted_count, elapsed_sec, now_time)
        save_session1_results(results)

        if TELEGRAM_OK:
            tg_msg = format_telegram_sesi1(
                results, scan_time, total_scanned, skipped, now_time
            )
            ok = send_telegram_message(tg_msg)
            print("✅ Telegram terkirim." if ok else "❌ Telegram gagal.")

    else:
        # ── Sesi 2: bandingkan + tampilkan konfirmasi ─
        comparison = compare_sessions(sesi1_tickers, results)
        print_sesi2_results(
            comparison, total_scanned, skipped,
            blacklisted_count, elapsed_sec, has_sesi1
        )

        if TELEGRAM_OK:
            tg_msg = format_telegram_sesi2(
                comparison["confirmed"],
                comparison["new"],
                comparison["dropped"],
                scan_time,
                total_scanned,
                skipped,
                has_sesi1
            )
            ok = send_telegram_message(tg_msg)
            print("✅ Telegram terkirim." if ok else "❌ Telegram gagal.")

    logger.info(f"Scanner selesai ({elapsed_sec:.1f}s)")
    print(f"⏱  Total waktu: {elapsed_sec:.1f} detik\n")

# ======================================================
# 11. ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_scanner_dual()