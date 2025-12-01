# scanner.py - UT BOT SCANNER v2.7 (667 SAHAM BEI + RSI FILTER)
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime as dt
import time

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_OK = True
except:
    TELEGRAM_OK = False
    print("⚠️  config.py belum lengkap - Telegram skip")

# ⚙️ SETTING UT BOT - BISA DIUBAH DI SINI
UT_KEY_VALUE = 2.0    # Default: 2.0 (1.0=agresif, 3.0=konservatif)
UT_ATR_PERIOD = 10    # Default: 10 (5=cepat, 14=lambat)
RSI_PERIOD = 14       # RSI period
RSI_BUY_THRESHOLD = 50  # RSI > 50 untuk BUY signal

def load_bei_tickers():
    """🚀 LOAD 667 SAHAM BEI dari data.csv"""
    try:
        df = pd.read_csv('data/data.csv', header=None, names=['Symbol', 'Nama_Perusahaan'])
        df_clean = df.dropna().reset_index(drop=True)
        tickers = df_clean['Symbol'].tolist()
        print(f"✅ {len(tickers)} saham BEI loaded dari data.csv")
        print(f"📋 Range: {tickers[0]} → {tickers[-1]}")
        return tickers
    except Exception as e:
        print(f"❌ Error load data.csv: {e}")
        return []

def calculate_rsi(data, period=14):
    """🔥 RSI CALCULATION - Filter sinyal UT BOT"""
    try:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except:
        return 50

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
            "text": text
        },
    )
    print(f"📱 DEBUG HTTP: {response.status_code}")

def ut_bot_signals(df: pd.DataFrame, key_value: float = 2.0, atr_period: int = 10) -> tuple[bool, bool]:
    """UT BOT ENGINE - FIXED untuk yfinance terbaru"""
    if len(df) < atr_period + 5:
        return False, False
    
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(atr_period).mean()
    
    if pd.isna(atr.iloc[-1]):
        return False, False
    
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
    
    return bool(pos == 1), bool(pos == -1)

def check_ut_bot_signal(ticker: str, key_value: float = 2.0, atr_period: int = 10) -> dict:
    """🔍 UT BOT + RSI FILTER - SINYAL INSTAN + AKURAT"""
    for attempt in range(3):  # Retry 3x kalau yfinance error
        try:
            data = yf.download(ticker+'.JK', period="15d", interval="1d", progress=False, auto_adjust=False)
            if data.empty or len(data) < 15:
                time.sleep(0.5)
                continue
                
            data = fix_yfinance_data(data)
            
            # UT BOT: Cek perubahan warna (merah → hijau)
            ut_buy_today, _ = ut_bot_signals(data.iloc[:-1], key_value, atr_period)
            ut_buy_now, _ = ut_bot_signals(data, key_value, atr_period)
            
            # RSI FILTER: Konfirmasi kekuatan (>50)
            rsi_value = calculate_rsi(data, RSI_PERIOD)
            
            # ✅ SINYAL FRESH: UT BOT hijau + RSI > 50
            if ut_buy_now and not ut_buy_today and rsi_value > RSI_BUY_THRESHOLD:
                return {
                    "ticker": ticker,
                    "action": f"🟢 UT BOT BUY (RSI {rsi_value:.0f})",
                    "price": float(data['Close'].iloc[-1]),
                    "rsi": rsi_value,
                    "date": data.index[-1].strftime("%d/%m")
                }
            return None
        except Exception as e:
            if attempt == 2:
                return None
            time.sleep(1)
    return None

def test_settings_comparison(tickers: list, sample_size: int = 50):
    """🧪 Test 3 setting UT BOT + RSI"""
    print("\n" + "="*70)
    print("🔥 UT BOT + RSI SCANNER v2.7 (667 SAHAM BEI)")
    print("="*70)
    
    settings = [
        ("AGRESIF", 1.0, 5),
        ("STANDAR", 2.0, 10), 
        ("KONSERVATIF", 3.0, 14)
    ]
    
    results = {}
    
    for name, kv, atr in settings:
        test_signals = []
        print(f"\n📊 Testing {name} (key={kv}, atr={atr}, RSI>{RSI_BUY_THRESHOLD})...")
        
        for i, ticker in enumerate(tickers[:sample_size]):
            if i % 10 == 0:
                print(f"   Progress: {i}/{sample_size}")
            signal = check_ut_bot_signal(ticker, kv, atr)
            if signal:
                test_signals.append(signal['ticker'])
        
        results[name] = {
            "count": len(test_signals),
            "tickers": test_signals,
            "key_value": kv,
            "atr_period": atr
        }
        
        print(f"   ✅ {len(test_signals)}/{sample_size} sinyal (RSI>{RSI_BUY_THRESHOLD})")
        if test_signals[:5]:
            print(f"   📌 Contoh: {', '.join(test_signals[:5])}")
    
    print("\n" + "="*70)
    print("📈 REKOMENDASI (RSI FILTER AKTIF):")
    print("="*70)
    for name in results:
        print(f"  {name:<12} : {results[name]['count']} sinyal")
    print("="*70)
    
    return results

def main():
    print("🚀 UT BOT + RSI SCANNER v2.7 - 667 SAHAM BEI")
    
    # 🔥 LOAD 667 SAHAM
    tickers = load_bei_tickers()
    if not tickers:
        print("❌ Gagal load data.csv")
        return
    
    # 🧪 SCAN 50 SAHAM CEPAT (bisa diubah ke len(tickers) untuk full)
    print(f"\n🔥 SCAN {len(tickers)} SAHAM BEI (sample 50)...")
    results = test_settings_comparison(tickers, sample_size=50)
    
    print(f"\n✅ SCAN + RSI FILTER SELESAI!")
    
    # 📱 TELEGRAM SELALU KIRIM
    jakarta_tz = dt.timezone(dt.timedelta(hours=7))
    now_wib = dt.datetime.now(jakarta_tz).strftime("%d/%m %H:%M WIB")
    
    message = f"🔥 UT BOT + RSI v2.7 - 667 SAHAM BEI\n({now_wib})\n\n"
    message += f"📊 SCAN: {len(tickers)} saham BEI (RSI>{RSI_BUY_THRESHOLD})\n\n"
    
    # RINGKASAN
    total_signals = sum(r['count'] for r in results.values())
    message += f"🔥 AGRESIF      : {results['AGRESIF']['count']}\n"
    message += f"⭐ STANDAR      : {results['STANDAR']['count']}\n"
    message += f"🛡️ KONSERVATIF  : {results['KONSERVATIF']['count']}\n\n"
    
    # DAFTAR SINYAL
    for name in ['AGRESIF', 'STANDAR', 'KONSERVATIF']:
        signals = results[name]['tickers']
        message += f"📌 {name}:\n"
        if signals:
            for t in signals[:15]:  # Max 15 per kategori
                message += f"• {t}\n"
            if len(signals) > 15:
                message += f"... +{len(signals)-15} lagi\n"
        else:
            message += "-\n"
        message += "\n"
    
    # STATUS
    if total_signals == 0:
        message += "ℹ️ PASAR SIDEWAYS - Belum ada sinyal kuat\n⏳ Tunggu RSI>50 + UT BOT hijau!"
    else:
        message += f"📈 {total_signals} SINYAL KUAT (RSI>{RSI_BUY_THRESHOLD})!"
    
    message += f"\n\n⚙️ RSI>{RSI_BUY_THRESHOLD} | Update data.csv → rerun"
    
    if TELEGRAM_OK:
        send_telegram_message(message)
        print("📱 Telegram OK! (RSI Filter aktif)")
    else:
        print("📱 Telegram skip - cek config.py")
        print(message)

if __name__ == "__main__":
    main()
