# ================================================================
# scanner_bsjp_v2_combo.py - ULTIMATE BSJP & TREND SCANNER
# Desc    : Scanner saham IDX menggabungkan 4 Filter Utama:
#           1. BSJP Base (Harga & Momentum)
#           2. Trend & Volume Spike (MA20, MA50)
#           3. Value & Liquidity (Syarat likuiditas ketat)
#           4. Bandarmologi Proxy (Akumulasi)
#           Dilengkapi Notifikasi Telegram
# ================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import logging
import time

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

# ======================================================
# 0. LOGGING SETUP
# ======================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"scanner_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BSJP_ULTIMATE")

# ======================================================
# 1. CONSTANTS & FILTER THRESHOLDS
# ======================================================
VERSION = "3.0.0-Combo"

# Filter 1: BSJP Base
CLOSE_HIGH_RATIO = 0.98          
MIN_FREQUENCY = 1000             
MIN_PRICE_CHANGE_PCT = 3         
RSI_MAX = 80                     

# Filter 2: Trend & Volume
MA20_PERIOD = 20
MA50_PERIOD = 50
VOL_SPIKE_MULTIPLIER = 2         

# Filter 3: Liquidity 
MIN_VALUE_IDR = 100_000_000           
MIN_VALUE_MA20_IDR = 1_000_000_000    

# Filter 4: Bandarmologi (Proxy)
BANDAR_MA10_PERIOD = 10
BANDAR_MA20_PERIOD = 20

# Risk Management
TP_PERCENT = 0.08                
CL_PERCENT = 0.05                

# Data fetch & Telegram
YFINANCE_PERIOD = "6mo"  
YFINANCE_INTERVAL = "1d"
MAX_MESSAGE_LENGTH = 4096        

# ======================================================
# 2. TELEGRAM CONFIG & HELPERS
# ======================================================
TELEGRAM_OK = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if REQUESTS_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        TELEGRAM_OK = True
        logger.info(f"Telegram Config Loaded. Chat ID: {TELEGRAM_CHAT_ID}")
except ImportError:
    logger.warning("config.py tidak ditemukan. Telegram notifikasi dinonaktifkan.")

def split_telegram_message(message: str) -> List[str]:
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks = []
    current_chunk = ""
    lines = message.split("\n")
    for line in lines:
        if len(line) > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for j in range(0, len(line), MAX_MESSAGE_LENGTH):
                chunks.append(line[j:j + MAX_MESSAGE_LENGTH])
            continue
        test_chunk = current_chunk + line + "\n"
        if len(test_chunk) > MAX_MESSAGE_LENGTH:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk = test_chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks if chunks else [message[:MAX_MESSAGE_LENGTH]]

def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_OK:
        return False
    messages_to_send = split_telegram_message(message)
    all_success = True
    for i, msg_chunk in enumerate(messages_to_send):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg_chunk, "parse_mode": "HTML"}
            response = requests.post(url, data=data, timeout=15)
            if response.status_code != 200:
                logger.error(f"Gagal kirim Telegram: {response.text}")
                all_success = False
        except Exception as e:
            logger.error(f"Error koneksi Telegram: {e}")
            all_success = False
        if len(messages_to_send) > 1 and i < len(messages_to_send) - 1:
            time.sleep(1)
    return all_success

def format_telegram_message(results: List[Dict], scan_time: str, total_scanned: int) -> str:
    msg_lines = [
        f"<b>Gemini Scanner V{VERSION}</b>",
        f"<i>{scan_time}</i>",
        "",
        f"Scanned: {total_scanned} | <b>Match: {len(results)} saham</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not results:
        msg_lines.append("\n<i>Screener sangat ketat. Tidak ada saham yang lolos 4 filter hari ini.</i>")
        return "\n".join(msg_lines)

    for r in results:
        detail_line = f"<b>{r['ticker']}</b> | {r['change_pct']:+.2f}% | RSI: {r['rsi']:.0f} | Val: {r['value_b']:.2f}B"
        risk_line = f"   Entry: {r['entry']:,} → TP: {r['tp']:,} | CL: {r['cl']:,}"
        filter_line = f"   ⭐ Lolos 4 Kombinasi Filter (Strict)"
        
        msg_lines.append(detail_line)
        msg_lines.append(risk_line)
        msg_lines.append(filter_line)
        msg_lines.append("")

    return "\n".join(msg_lines)

# ======================================================
# 3. CORE LOGIC (4 GABUNGAN FILTER)
# ======================================================
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    try:
        symbol = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        df = yf.download(symbol, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL, progress=False, auto_adjust=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols): return pd.DataFrame()
        return df[required_cols].copy().dropna()
    except Exception:
        return pd.DataFrame()

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else 100
        return round(100.0 - (100.0 / (1.0 + rs)), 1)
    except Exception:
        return -1.0

def analyze_stock(ticker: str) -> Optional[Dict]:
    df = fetch_stock_data(ticker)
    
    if df.empty or len(df) < (MA50_PERIOD + 1):
        return None

    df['Value'] = df['Close'] * df['Volume']
    df['Value_MA20'] = df['Value'].rolling(MA20_PERIOD).mean()
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()
    df['Vol_MA20'] = df['Volume'].rolling(MA20_PERIOD).mean()
    df['MA20'] = df['Close'].rolling(MA20_PERIOD).mean()
    df['MA50'] = df['Close'].rolling(MA50_PERIOD).mean()
    
    df['Bandar_Value'] = np.where(df['Close'] > df['Open'], df['Value'] * 0.5, df['Value'] * -0.5)
    df['Bandar_MA10'] = df['Bandar_Value'].rolling(BANDAR_MA10_PERIOD).mean()
    df['Bandar_MA20'] = df['Bandar_Value'].rolling(BANDAR_MA20_PERIOD).mean()

    current = df.iloc[-1]
    prev = df.iloc[-2]

    if current['Close'] <= 0 or current['Volume'] <= 0: return None

    price_change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
    rsi_value = calculate_rsi(df['Close'])

    # F1: BSJP Base
    pass_f1 = all([
        current['Close'] >= (current['High'] * CLOSE_HIGH_RATIO),
        current['Volume'] > MIN_FREQUENCY,
        current['Volume'] > current['Vol_MA5'],
        price_change_pct >= MIN_PRICE_CHANGE_PCT,
        rsi_value < RSI_MAX
    ])

    # F2: Trend & Volume
    pass_f2 = all([
        current['Close'] > current['MA20'],
        current['Close'] > current['MA50'],
        current['Volume'] >= (VOL_SPIKE_MULTIPLIER * current['Vol_MA20'])
    ])

    # F3: Liquidity
    pass_f3 = all([
        current['Value'] >= MIN_VALUE_IDR,
        current['Value'] > current['Value_MA20'],
        current['Value_MA20'] > MIN_VALUE_MA20_IDR
    ])

    # F4: Bandarmologi Proxy
    pass_f4 = all([
        current['Bandar_Value'] > current['Bandar_MA20'],
        prev['Bandar_Value'] <= current['Bandar_Value'],
        current['Bandar_MA10'] > current['Bandar_MA20']
    ])

    if not (pass_f1 and pass_f2 and pass_f3 and pass_f4):
        return None

    entry_price = int(current['Close'])
    return {
        "ticker": ticker,
        "close": entry_price,
        "change_pct": round(float(price_change_pct), 2),
        "volume": int(current['Volume']),
        "value_b": round(float(current['Value']) / 1e9, 2),
        "rsi": rsi_value,
        "entry": entry_price,
        "tp": int(entry_price * (1 + TP_PERCENT)),
        "cl": int(entry_price * (1 - CL_PERCENT))
    }

# ======================================================
# 4. EXECUTION HELPERS & MAIN
# ======================================================
def load_tickers_from_csv() -> List[str]:
    file_path = os.path.join("data", "data.csv") if os.path.exists(os.path.join("data", "data.csv")) else "data.csv"
    try:
        df = pd.read_csv(file_path)
        col = next((c for c in ['Ticker','ticker','Kode','kode'] if c in df.columns), df.columns[0])
        return sorted(list(set([str(t).strip().upper() for t in df[col].dropna() if len(str(t).strip()) >= 4])))
    except Exception: return []

def run_scanner():
    scan_start = dt.datetime.now()
    scan_time_str = scan_start.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}\n  ULTIMATE 4-FILTER SCANNER V{VERSION}\n{'='*60}")
    tickers = load_tickers_from_csv()
    if not tickers:
        print("Data ticker tidak ditemukan.")
        return

    results = []
    print(f"Scanning {len(tickers)} saham dengan filter ketat...\n")
    
    for i, ticker in enumerate(tickers):
        print(f"\r  [{(i+1)/len(tickers)*100:5.1f}%] Scanning: {ticker:<6}", end="", flush=True)
        try:
            res = analyze_stock(ticker)
            if res:
                print(f"\n  ✅ MATCH: {ticker} (+{res['change_pct']}%, Value: {res['value_b']}B)")
                results.append(res)
        except KeyboardInterrupt:
            break
        except Exception:
            continue

    print("\n\n" + "="*60)
    print(f"  SCAN SELESAI — Ditemukan: {len(results)} saham")
    print("="*60)
    
    if results:
        results.sort(key=lambda x: -x['change_pct'])
        for r in results:
            print(f"[{r['ticker']}] Close: {r['close']} | Chg: {r['change_pct']}% | Val: {r['value_b']}B")
    
    # --- TELEGRAM NOTIFICATION ---
    if TELEGRAM_OK:
        print("\nMengirim hasil ke Telegram...")
        telegram_msg = format_telegram_message(results, scan_time_str, len(tickers))
        success = send_telegram_message(telegram_msg)
        if success:
            print("✅ Hasil terkirim ke Telegram.")
        else:
            print("❌ Gagal mengirim ke Telegram.")
    else:
        print("\nℹ️ Telegram tidak dikonfigurasi. Hasil hanya ditampilkan di terminal.")

if __name__ == "__main__":
    run_scanner()