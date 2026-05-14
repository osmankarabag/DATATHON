import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# 1. VERİYİ YÜKLE
# =============================================================
df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
print("=" * 70)
print("EKSIK DEGER DOLDURMA ISLEMI")
print("=" * 70)
print(f"Orijinal veri boyutu: {df.shape}")
print(f"\nDoldurma oncesi eksik deger sayilari:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# =============================================================
# 2. KATEGORİK: ruh_sagligi_durumu & kronotip -> MOD ile doldur
# =============================================================
print("\n" + "-" * 70)
print("ADIM 1: ruh_sagligi_durumu & kronotip -> Mod (en sik tekrar eden)")
print("-" * 70)

for col in ['ruh_sagligi_durumu', 'kronotip']:
    df[col] = df[col].astype(object)  # StringDtype -> object (fillna uyumu)
    mode_val = df[col].mode()[0]
    eksik = df[col].isnull().sum()
    df[col] = df[col].fillna(mode_val)
    print(f"  {col}: {eksik} eksik deger '{mode_val}' ile dolduruldu.")
    print(f"    Kalan eksik: {df[col].isnull().sum()}")

# =============================================================
# 3. KATEGORİK: meslek -> KNN ile doldur
# =============================================================
print("\n" + "-" * 70)
print("ADIM 2: meslek -> KNN Imputation")
print("-" * 70)

meslek_eksik = df['meslek'].isnull().sum()
df['meslek'] = df['meslek'].astype(object)  # StringDtype -> object
print(f"  meslek: {meslek_eksik} eksik deger KNN ile doldurulacak.")

# KNN icin kategorik degiskenleri encode etmemiz gerekiyor
# Meslek icin: Diger kategorik sutunlari da feature olarak kullanacagiz
cat_cols_for_knn = ['cinsiyet', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi', 'meslek']
# ulke cok fazla unique (15) - knn mesafe hesabinda noise yaratabilir, disarida birakalim

# Label Encoding uygula (gecici, sadece KNN icin)
label_encoders = {}
df_encoded = df.copy()

# String dtype'i object'e cevir (LabelEncoder uyumlulugu icin)
for col in cat_cols_for_knn:
    df_encoded[col] = df_encoded[col].astype(object)
    # NaN'lari koru
    df_encoded.loc[df[col].isnull(), col] = np.nan

for col in cat_cols_for_knn:
    le = LabelEncoder()
    mask = df_encoded[col].isnull()
    # Sadece dolu degerleri fit et
    non_null_vals = df_encoded.loc[~mask, col].values
    encoded_vals = le.fit_transform(non_null_vals)
    df_encoded.loc[~mask, col] = encoded_vals.astype(float)
    df_encoded.loc[mask, col] = np.nan
    df_encoded[col] = df_encoded[col].astype(float)
    label_encoders[col] = le

# KNN icin kullanilacak feature'lar (meslek haric diger bilgiler)
num_cols = ['yas', 'vucut_kitle_indeksi', 'rem_yuzdesi', 'derin_uyku_yuzdesi',
            'uykuya_dalma_suresi_dk', 'gecelik_uyanma_sayisi', 'uyku_oncesi_kafein_mg',
            'uyku_oncesi_ekran_suresi_dk', 'gunluk_adim_sayisi', 'sekerleme_suresi_dk',
            'stres_skoru', 'gunluk_calisma_saati', 'dinlenik_nabiz_bpm',
            'oda_sicakligi_celsius', 'hafta_sonu_uyku_farki_saat', 'bilissel_performans_skoru']

features_for_meslek = [c for c in cat_cols_for_knn if c != 'ulke'] + num_cols

# Scaling (KNN mesafe tabanli, olcek onemli)
scaler_meslek = StandardScaler()
df_meslek_features = df_encoded[features_for_meslek].copy()

# Once NaN olmayan satirlarda fit edelim
not_null_mask = df_meslek_features.dropna().index
scaler_meslek.fit(df_meslek_features.loc[not_null_mask])

# Tum veriyi scale et (NaN'lar NaN kalir)
df_meslek_scaled = pd.DataFrame(
    scaler_meslek.transform(df_meslek_features),
    columns=features_for_meslek,
    index=df_meslek_features.index
)

# KNN Imputer - meslek dahil tum eksikler doldurulacak
knn_imputer_meslek = KNNImputer(n_neighbors=5, weights='distance')
print("  KNN Imputer calistiriliyor (k=5, weights='distance')...")
imputed_meslek = knn_imputer_meslek.fit_transform(df_meslek_scaled)
imputed_meslek_df = pd.DataFrame(imputed_meslek, columns=features_for_meslek, index=df.index)

# Inverse scale
imputed_meslek_unscaled = pd.DataFrame(
    scaler_meslek.inverse_transform(imputed_meslek_df),
    columns=features_for_meslek,
    index=df.index
)

# Meslek sutununu decode et
meslek_col_idx = features_for_meslek.index('meslek')
meslek_imputed_encoded = imputed_meslek_unscaled['meslek'].round().astype(int)
# Sinir disi degerleri clamp et
n_classes = len(label_encoders['meslek'].classes_)
meslek_imputed_encoded = meslek_imputed_encoded.clip(0, n_classes - 1)
meslek_imputed_decoded = label_encoders['meslek'].inverse_transform(meslek_imputed_encoded)

# Sadece NaN olan yerleri doldur
meslek_null_mask = df['meslek'].isnull()
df.loc[meslek_null_mask, 'meslek'] = meslek_imputed_decoded[meslek_null_mask]
print(f"  meslek: {meslek_eksik} eksik deger KNN ile dolduruldu.")
print(f"  Doldurulan deger dagilimi:")
print(df.loc[meslek_null_mask, 'meslek'].value_counts().to_string())

# =============================================================
# 4. NÜMERİK: vucut_kitle_indeksi, stres_skoru,
#    uyku_oncesi_kafein_mg -> KNN ile doldur
# =============================================================
print("\n" + "-" * 70)
print("ADIM 3: Numerik degiskenler -> KNN Imputation")
print("-" * 70)

numerik_eksik_cols = ['vucut_kitle_indeksi', 'stres_skoru', 'uyku_oncesi_kafein_mg']
for col in numerik_eksik_cols:
    print(f"  {col}: {df[col].isnull().sum()} eksik deger")

# Tum numerik sutunlari kullanarak KNN uygula
# Kategorikleri de encode edip feature olarak kullanalim
df_num_knn = df.copy()

# Kategorikleri encode et (artik eksik yok kategoriklerde)
cat_all = ['cinsiyet', 'meslek', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
le_dict = {}
for col in cat_all:
    le = LabelEncoder()
    df_num_knn[col] = le.fit_transform(df_num_knn[col])
    le_dict[col] = le

# KNN icin feature secimi
knn_features = num_cols + cat_all  # ulke haric

# Scaling
scaler_num = StandardScaler()
df_knn_input = df_num_knn[knn_features].copy().astype(float)
not_null_idx = df_knn_input.dropna().index
scaler_num.fit(df_knn_input.loc[not_null_idx])

df_knn_scaled = pd.DataFrame(
    scaler_num.transform(df_knn_input),
    columns=knn_features,
    index=df_knn_input.index
)

# KNN Imputer
knn_imputer_num = KNNImputer(n_neighbors=5, weights='distance')
print("  KNN Imputer calistiriliyor (k=5, weights='distance')...")
imputed_num = knn_imputer_num.fit_transform(df_knn_scaled)

# Inverse scale
imputed_num_unscaled = pd.DataFrame(
    scaler_num.inverse_transform(imputed_num),
    columns=knn_features,
    index=df.index
)

# Sadece eksik olan numerik sutunlari doldur
for col in numerik_eksik_cols:
    null_mask = df[col].isnull()
    df.loc[null_mask, col] = imputed_num_unscaled.loc[null_mask, col]
    print(f"  {col}: KNN ile dolduruldu. (Ort: {df.loc[null_mask, col].mean():.2f})")

# =============================================================
# 5. DOĞRULAMA
# =============================================================
print("\n" + "=" * 70)
print("DOGRULAMA: Doldurma sonrasi eksik deger kontrolu")
print("=" * 70)
remaining = df.isnull().sum()
print(remaining[remaining > 0].to_string() if remaining.sum() > 0 else "Hicbir sutunda eksik deger kalmadi!")
print(f"\nToplam kalan eksik: {remaining.sum()}")

# =============================================================
# 6. SONUCU KAYDET
# =============================================================
output_path = os.path.join(BASE_DIR, "train_filled.csv")
df.to_csv(output_path, index=False)
print(f"\nTemizlenmis veri kaydedildi: {output_path}")
print(f"Kayit boyutu: {df.shape}")

# Ozet istatistikler
print("\n" + "=" * 70)
print("DOLDURULAN DEGISKENLERIN OZET ISTATISTIKLERI")
print("=" * 70)
summary_cols = ['vucut_kitle_indeksi', 'stres_skoru', 'uyku_oncesi_kafein_mg']
print(df[summary_cols].describe().round(3).to_string())

print("\nKategorik degiskenlerin dagilimi (doldurma sonrasi):")
for col in ['meslek', 'kronotip', 'ruh_sagligi_durumu']:
    print(f"\n  {col}:")
    print(df[col].value_counts().to_string())
