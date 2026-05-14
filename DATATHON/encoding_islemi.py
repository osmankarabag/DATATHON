import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "train_filled.csv"))

print("=" * 70)
print("ENCODING ISLEMI")
print("=" * 70)
print(f"Giris boyutu: {df.shape}")

# =============================================================
# 1. BINARY KATEGORIKLER -> Label Encoding (0/1)
# =============================================================
print("\n" + "-" * 70)
print("ADIM 1: Binary kategorikler -> Label Encoding")
print("-" * 70)

binary_mappings = {
    'cinsiyet': {'Erkek': 0, 'Kadin': 1},
    'mevsim': {'Sonbahar-Kis': 0, 'Ilkbahar-Yaz': 1},
    'gun_tipi': {'Hafta ici': 0, 'Hafta sonu': 1},
}

for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)
    print(f"  {col}: {mapping}")

# =============================================================
# 2. ORDINAL KATEGORIKLER -> Ordinal Encoding
# =============================================================
print("\n" + "-" * 70)
print("ADIM 2: Ordinal kategorikler -> Ordinal Encoding")
print("-" * 70)

# kronotip: Gece insani < Notr < Sabah insani (uyku duzeni sirasina gore)
kronotip_mapping = {'Gece insani': 0, 'Notr': 1, 'Sabah insani': 2}
df['kronotip'] = df['kronotip'].map(kronotip_mapping)
print(f"  kronotip: {kronotip_mapping}")

# ruh_sagligi_durumu: En kotu -> En iyi sirasi
ruh_mapping = {
    'Anksiyete ve depresyon': 0,
    'Depresyon': 1,
    'Anksiyete': 2,
    'Saglikli': 3
}
df['ruh_sagligi_durumu'] = df['ruh_sagligi_durumu'].map(ruh_mapping)
print(f"  ruh_sagligi_durumu: {ruh_mapping}")

# =============================================================
# 3. NOMINAL KATEGORIKLER -> One-Hot Encoding
# =============================================================
print("\n" + "-" * 70)
print("ADIM 3: Nominal kategorikler -> One-Hot Encoding")
print("-" * 70)

# Meslek (11 sinif)
print(f"  meslek: {df['meslek'].nunique()} sinif -> One-Hot")
df = pd.get_dummies(df, columns=['meslek'], prefix='meslek', dtype=int)

# Ulke (15 sinif)
print(f"  ulke: {df['ulke'].nunique()} sinif -> One-Hot")
df = pd.get_dummies(df, columns=['ulke'], prefix='ulke', dtype=int)

# =============================================================
# 4. DOGRULAMA
# =============================================================
print("\n" + "=" * 70)
print("DOGRULAMA")
print("=" * 70)

# Kalan string sutun var mi?
str_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
if str_cols:
    print(f"  UYARI: Hala string sutunlar var: {str_cols}")
else:
    print("  Tum sutunlar sayisal formata donusturuldu.")

print(f"\nSonuc boyutu: {df.shape}")
print(f"  Orijinal sutun sayisi: 24")
print(f"  Encoding sonrasi sutun sayisi: {df.shape[1]}")

# Sutun listesi
print(f"\nSutun listesi:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col} ({df[col].dtype})")

# Eksik deger kontrolu
missing = df.isnull().sum().sum()
print(f"\nEksik deger: {missing}")

# Kaydet
output_path = os.path.join(BASE_DIR, "train_encoded.csv")
df.to_csv(output_path, index=False)
print(f"\nEncoded veri kaydedildi: {output_path}")
