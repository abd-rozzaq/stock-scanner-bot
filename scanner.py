# scanner.py - UT BOT SCANNER (FIX yfinance MultiIndex + Series Error)
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
from tickers import load_all_tickers
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except:
    TELEGRAM_OK = False
    print("⚠️  config.py belum lengkap - Telegram skip")


def fix_yfinance_data(data):
    """Fix MultiIndex columns dari yfinance terbaru"""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_OK:
        print("📱 Telegram skip - config belum lengkap")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text  # tanpa parse_mode, kirim teks biasa
        },
    )
    print(f"📱 DEBUG HTTP: {response.status_code} - {response.text[:120]}...")


def ut_bot_signals(df: pd.DataFrame, key_value: float = 2.0, atr_period: int = 10) -> tuple[bool, bool]:
    """UT BOT ENGINE - FIXED untuk yfinance terbaru"""
    if len(df) < atr_period + 5:
        return False, False
    
    # Pastikan kolom ada dan clean
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    
    # Heikin Ashi close
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(atr_period).mean()
    
    if pd.isna(atr.iloc[-1]):
        return False, False
    
    # xATRTrailingStop - simplified
    nloss = key_value * atr
    xatr = [0.0] * len(df)
    
    for i in range(atr_period, len(df)):
        src = ha_close.iloc[i]
        src_prev = ha_close.iloc[i-1]
        nloss_i = nloss.iloc[i]
        xatr_prev = xatr[i-1]
        
        if src > xatr_prev and src_prev > xatr_prev:
            xatr[i] = max(xatr_prev, src - nloss_i)
        elif src < xatr_prev and src_prev < xatr_prev:
            xatr[i] = min(xatr_prev, src + nloss_i)
        elif src > xatr_prev:
            xatr[i] = src - nloss_i
        else:
            xatr[i] = src + nloss_i
    
    # Position - ambil 3 baris terakhir saja (cepat)
    pos = 0
    for i in range(max(0, len(df)-3), len(df)):
        src_curr = ha_close.iloc[i]
        src_prev = ha_close.iloc[i-1] if i > 0 else src_curr
        xatr_curr = xatr[i]
        xatr_prev = xatr[i-1] if i > 0 else xatr_curr
        
        if src_prev < xatr_prev and src_curr > xatr_curr:
            pos = 1
        elif src_prev > xatr_prev and src_curr < xatr_curr:
            pos = -1
    
    # RSI sederhana - ambil nilai terakhir
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_last = rsi.iloc[-1]
    
    # Signal - scalar boolean
    buy_signal = pos == 1 and rsi_last < 65
    sell_signal = pos == -1 and rsi_last > 35
    
    return bool(buy_signal), bool(sell_signal)


def check_ut_bot_signal(ticker: str) -> dict:
    try:
        data = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=False)
        data = fix_yfinance_data(data)
        
        if len(data) < 30:
            return None
        
        ut_buy, ut_sell = ut_bot_signals(data)
        
        if ut_buy:
            return {
                "ticker": ticker,
                "action": "🟢 UT BOT BUY",
                "price": float(data['Close'].iloc[-1])
            }
        return None
    except Exception as e:
        return None


def main():
    print("🚀 UT BOT SCANNER v2.0 - FIXED yfinance")
    print("Loading tickers...")
    
    tickers = load_all_tickers()
    print(f"📊 {len(tickers)} ticker loaded")
    
    signals = []
    test_count = len(tickers)  # SCAN FULL 954 SAHAM
    
    print(f"🔍 Scanning {test_count} ticker...\n")
    
    for i, ticker in enumerate(tickers[:test_count]):
        print(f"[{i+1:2d}/{test_count}] {ticker}", end=" ... ")
        signal = check_ut_bot_signal(ticker)
        
        if signal:
            print(f"🟢 BUY Rp{signal['price']:,.0f}")
            signals.append(signal)
        else:
            print("No signal")
    
    print(f"\n✅ Scan selesai! Sinyal: {len(signals)}")
    
    if signals and TELEGRAM_OK:
        now = dt.datetime.now().strftime("%d/%m %H:%M")
        message = f"🟢 UT BOT SIGNALS ({now})\n\n"
        for s in signals:
            message += f"📈 {s['action']}\n{s['ticker']} Rp{s['price']:,.0f}\n\n"
        send_telegram_message(message)
        print("📱 Telegram OK!")
    elif signals:
        print("📱 Telegram skip - lengkapi config.py")
    else:
        print("ℹ️  Belum ada sinyal (normal)")


if __name__ == "__main__":
    main()
