# backtest_pro.py - LINKED TO SCANNER
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
# ========================================================
# 🔥 IMPORT LOGIC DARI SCANNER_PRO.PY
# ========================================================
import scanner_pro as sp 

warnings.filterwarnings('ignore')

# Config Backtest
INITIAL_CAPITAL = 100_000_000
TEST_PERIOD = "2y"

def run_backtest(ticker: str):
    print(f"\n🔗 LINKED TEST: {ticker} ... ", end="")
    
    symbol = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
    try:
        df = yf.download(symbol, period=TEST_PERIOD, interval="1d", progress=False, auto_adjust=False)
        if len(df) < sp.EMA_TREND_PERIOD: 
            print("SKIP (Data)")
            return
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    except: 
        return

    # --- PANGGIL RUMUS DARI FILE SCANNER ---
    # Jika rumus di scanner_pro.py salah, backtest ini otomatis salah.
    df['EMA_Trend'] = sp.calculate_ema(df['Close'], sp.EMA_TREND_PERIOD)
    df['RSI'] = sp.calculate_rsi(df['Close'], sp.RSI_PERIOD)
    
    # Ambil array signal dari scanner logic
    signals, _ = sp.calculate_ut_bot(df, sp.UT_KEY, sp.UT_ATR)
    df['UT_Signal'] = signals
    
    df['Avg_Vol'] = df['Volume'].rolling(20).mean()

    # --- SIMULASI (Menggunakan Setting Scanner) ---
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    highest_price = 0
    trades = []
    
    # Convert to Numpy
    dates = df.index
    closes = df['Close'].values
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    emas = df['EMA_Trend'].values
    rsis = df['RSI'].values
    sig_arr = df['UT_Signal'].values
    avg_vols = df['Avg_Vol'].values
    
    start_idx = sp.EMA_TREND_PERIOD + 20
    
    for i in range(start_idx, len(df)-1):
        # LOGIC EXIT (Menggunakan Setting Trailing Scanner)
        if position > 0:
            if highs[i] > highest_price: highest_price = highs[i]
            
            # Ambil Trailing % langsung dari Scanner Config
            trail_level = highest_price * (1 - sp.TRAILING_STOP_PCT)
            
            if lows[i] <= trail_level:
                exit_price = trail_level
                profit = (position * exit_price * 0.9975) - (position * entry_price * 1.0015)
                capital += profit
                trades.append({'Ret%': (profit/(position*entry_price))*100})
                position = 0
                highest_price = 0
                continue
            
            if sig_arr[i] == -1: # Technical Exit
                exit_price = closes[i]
                profit = (position * exit_price * 0.9975) - (position * entry_price * 1.0015)
                capital += profit
                trades.append({'Ret%': (profit/(position*entry_price))*100})
                position = 0
                highest_price = 0
                continue

        # LOGIC ENTRY (Menggunakan Setting Filter Scanner)
        if position == 0:
            if closes[i] < emas[i]: continue
            if not (sp.RSI_MIN <= rsis[i] <= sp.RSI_MAX): continue
            if volumes[i] < (avg_vols[i] * sp.VOLUME_MULTIPLIER): continue
            if not (sig_arr[i] == 1 and sig_arr[i-1] == -1): continue
            if closes[i] <= opens[i]: continue

            entry_price = closes[i]
            position = int((capital * 0.99) / entry_price)
            highest_price = entry_price

    # REPORT
    if not trades:
        print("No Trades.")
        return

    df_res = pd.DataFrame(trades)
    win_rate = len(df_res[df_res['Ret%'] > 0]) / len(df_res) * 100
    tot_ret = ((capital - INITIAL_CAPITAL)/INITIAL_CAPITAL)*100
    
    print(f"WinRate: {win_rate:.1f}% | Return: {tot_ret:.1f}% | Cap: {capital:,.0f}")

if __name__ == "__main__":
    targets = ["ADRO", "BRIS", "BBRI"]
    print("🔍 VERIFIKASI KONEKSI FILE...")
    print(f"   Menggunakan RSI Period dari Scanner: {sp.RSI_PERIOD}")
    print(f"   Menggunakan Trailing % dari Scanner: {sp.TRAILING_STOP_PCT*100}%")
    
    for t in targets:
        run_backtest(t)
