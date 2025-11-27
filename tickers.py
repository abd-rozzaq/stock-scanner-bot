# tickers.py - FIXED
import pandas as pd
from config import TICKERS_EXCEL_PATH  # Benar! Excel bukan CSV

def load_all_tickers() -> list[str]:
    """
    Membaca daftar saham dari file Excel, kolom 'Kode',
    lalu mengembalikan list simbol dengan suffix .JK untuk yfinance.
    """
    try:
        df = pd.read_excel(TICKERS_EXCEL_PATH, engine="openpyxl")
        print(f"✅ Excel loaded: {len(df)} rows")
        
        # pastikan kolom bernama 'Kode' persis
        if 'Kode' not in df.columns:
            print("❌ Kolom 'Kode' tidak ditemukan! Kolom tersedia:", df.columns.tolist())
            return []
            
        codes = (
            df["Kode"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        
        tickers = [code + ".JK" for code in codes]
        print(f"✅ {len(tickers)} ticker siap scan")
        return tickers
        
    except Exception as e:
        print(f"❌ Error load Excel: {e}")
        return []
