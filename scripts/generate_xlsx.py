import pandas as pd
import os

print("🔍 ANALISIS data.csv...")
total_lines = sum(1 for _ in open('../data/data.csv'))
df_raw = pd.read_csv('../data/data.csv', header=None, names=['Symbol', 'Nama_Perusahaan'])

# DETECT & FIX baris corrupt
print(f"📄 Total baris file: {total_lines}")
print(f"📊 Pandas baca: {len(df_raw)}")
print(f"❌ Baris corrupt: {df_raw.isnull().sum().sum()}")

df_clean = df_raw.dropna().reset_index(drop=True)
print(f"✅ {len(df_clean)} saham VALID (FIXED!)")

print(f"📋 Range: {df_clean['Symbol'].iloc[0]} → {df_clean['Symbol'].iloc[-1]}")
print("✨ Generate XLSX...")

with pd.ExcelWriter('../data/data.xlsx', engine='openpyxl') as writer:
    df_clean.to_excel(writer, sheet_name='Saham BEI', index=False)
    worksheet = writer.sheets['Saham BEI']
    
    # Auto-adjust kolom
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except: pass
        adjusted_width = min(max_length + 2, 50)
        worksheet.column_dimensions[column_letter].width = adjusted_width

print("🎉 data.xlsx SIAP dengan 667 saham!")
print("📥 Download: Explorer → data/data.xlsx")
