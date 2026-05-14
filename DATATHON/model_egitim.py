import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# 1. VERİYİ YÜKLE
# =============================================================
print("=" * 70)
print("MODEL EĞİTİM VE KARŞILAŞTIRMA")
print("=" * 70)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_x.csv"))

print(f"Train boyutu: {train_df.shape}")
print(f"Test boyutu:  {test_df.shape}")

TARGET = 'bilissel_performans_skoru'

# =============================================================
# 2. TRAIN VE TEST VERİSİNİ BİRLEŞTİR (Tutarlı ön işleme için)
# =============================================================
print("\n" + "-" * 70)
print("ADIM 1: Veri Birleştirme ve Ön İşleme")
print("-" * 70)

train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan

combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
print(f"Birlestirilmis veri boyutu: {combined.shape}")

# =============================================================
# 3. EKSİK DEĞER DOLDURMA (Train ile aynı strateji)
# =============================================================
print("\n" + "-" * 70)
print("ADIM 2: Eksik Değer Doldurma")
print("-" * 70)

# 3a. Kategorik: ruh_sagligi_durumu & kronotip -> MOD
for col in ['ruh_sagligi_durumu', 'kronotip']:
    combined[col] = combined[col].astype(object)
    mode_val = combined[col].mode()[0]
    eksik = combined[col].isnull().sum()
    combined[col] = combined[col].fillna(mode_val)
    print(f"  {col}: {eksik} eksik -> '{mode_val}' (mod) ile dolduruldu.")

# 3b. Kategorik: meslek -> KNN
meslek_eksik = combined['meslek'].isnull().sum()
combined['meslek'] = combined['meslek'].astype(object)
print(f"  meslek: {meslek_eksik} eksik -> KNN ile doldurulacak.")

cat_cols_for_knn = ['cinsiyet', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi', 'meslek']
num_cols = ['yas', 'vucut_kitle_indeksi', 'rem_yuzdesi', 'derin_uyku_yuzdesi',
            'uykuya_dalma_suresi_dk', 'gecelik_uyanma_sayisi', 'uyku_oncesi_kafein_mg',
            'uyku_oncesi_ekran_suresi_dk', 'gunluk_adim_sayisi', 'sekerleme_suresi_dk',
            'stres_skoru', 'gunluk_calisma_saati', 'dinlenik_nabiz_bpm',
            'oda_sicakligi_celsius', 'hafta_sonu_uyku_farki_saat']

# Label encoding (geçici - KNN için)
label_encoders_knn = {}
df_enc = combined.copy()
for col in cat_cols_for_knn:
    df_enc[col] = df_enc[col].astype(object)
    df_enc.loc[combined[col].isnull(), col] = np.nan

for col in cat_cols_for_knn:
    le = LabelEncoder()
    mask = df_enc[col].isnull()
    non_null = df_enc.loc[~mask, col].values
    df_enc.loc[~mask, col] = le.fit_transform(non_null).astype(float)
    df_enc.loc[mask, col] = np.nan
    df_enc[col] = df_enc[col].astype(float)
    label_encoders_knn[col] = le

features_for_meslek = [c for c in cat_cols_for_knn] + num_cols
scaler_m = StandardScaler()
df_feat = df_enc[features_for_meslek].copy()
not_null_idx = df_feat.dropna().index
scaler_m.fit(df_feat.loc[not_null_idx])
df_scaled = pd.DataFrame(scaler_m.transform(df_feat), columns=features_for_meslek, index=df_feat.index)

knn_imp = KNNImputer(n_neighbors=5, weights='distance')
imputed = knn_imp.fit_transform(df_scaled)
imputed_df = pd.DataFrame(scaler_m.inverse_transform(imputed), columns=features_for_meslek, index=combined.index)

# Meslek decode
n_cls = len(label_encoders_knn['meslek'].classes_)
meslek_enc = imputed_df['meslek'].round().astype(int).clip(0, n_cls - 1)
meslek_dec = label_encoders_knn['meslek'].inverse_transform(meslek_enc)
meslek_mask = combined['meslek'].isnull()
combined.loc[meslek_mask, 'meslek'] = meslek_dec[meslek_mask]
print(f"  meslek: {meslek_eksik} eksik KNN ile dolduruldu.")

# 3c. Numerik eksikler -> KNN
numerik_eksik = ['vucut_kitle_indeksi', 'stres_skoru', 'uyku_oncesi_kafein_mg']
for col in numerik_eksik:
    null_mask = combined[col].isnull()
    combined.loc[null_mask, col] = imputed_df.loc[null_mask, col]
    print(f"  {col}: {null_mask.sum()} eksik KNN ile dolduruldu.")

remaining = combined.drop(columns=[TARGET, 'is_train']).isnull().sum().sum()
print(f"\n  Kalan toplam eksik (hedef haric): {remaining}")

# =============================================================
# 4. ENCODING (Train ile aynı strateji)
# =============================================================
print("\n" + "-" * 70)
print("ADIM 3: Encoding")
print("-" * 70)

# Binary
binary_mappings = {
    'cinsiyet': {'Erkek': 0, 'Kadin': 1},
    'mevsim': {'Sonbahar-Kis': 0, 'Ilkbahar-Yaz': 1},
    'gun_tipi': {'Hafta ici': 0, 'Hafta sonu': 1},
}
for col, mapping in binary_mappings.items():
    combined[col] = combined[col].map(mapping)
    print(f"  Binary: {col} -> {mapping}")

# Ordinal
kronotip_map = {'Gece insani': 0, 'Notr': 1, 'Sabah insani': 2}
combined['kronotip'] = combined['kronotip'].map(kronotip_map)
print(f"  Ordinal: kronotip -> {kronotip_map}")

ruh_map = {'Anksiyete ve depresyon': 0, 'Depresyon': 1, 'Anksiyete': 2, 'Saglikli': 3}
combined['ruh_sagligi_durumu'] = combined['ruh_sagligi_durumu'].map(ruh_map)
print(f"  Ordinal: ruh_sagligi_durumu -> {ruh_map}")

# One-Hot
print(f"  One-Hot: meslek ({combined['meslek'].nunique()} sinif)")
combined = pd.get_dummies(combined, columns=['meslek'], prefix='meslek', dtype=int)
print(f"  One-Hot: ulke ({combined['ulke'].nunique()} sinif)")
combined = pd.get_dummies(combined, columns=['ulke'], prefix='ulke', dtype=int)

print(f"\n  Encoding sonrasi boyut: {combined.shape}")

# =============================================================
# 5. TRAIN / TEST AYIR
# =============================================================
train_data = combined[combined['is_train'] == 1].copy()
test_data = combined[combined['is_train'] == 0].copy()

drop_cols = ['id', 'is_train', TARGET]
feature_cols = [c for c in combined.columns if c not in drop_cols]

X_train = train_data[feature_cols].values
y_train = train_data[TARGET].values
X_test = test_data[feature_cols].values
test_ids = test_data['id'].values

print(f"\n  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Feature sayisi: {len(feature_cols)}")

# =============================================================
# 6. FEATURE SCALING (Mesafe tabanlı modeller için)
# =============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================
# 7. MODEL TANIMLARI
# =============================================================
print("\n" + "=" * 70)
print("MODEL EĞİTİM VE DEĞERLENDİRME (5-Fold Cross Validation)")
print("=" * 70)

models = {
    'Linear Regression': (LinearRegression(), False),
    'Ridge Regression': (Ridge(alpha=1.0), False),
    'Lasso Regression': (Lasso(alpha=0.01), False),
    'ElasticNet': (ElasticNet(alpha=0.01, l1_ratio=0.5), False),
    'KNN Regressor': (KNeighborsRegressor(n_neighbors=10), True),  # True = scaled
    'Random Forest': (RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ), False),
    'Extra Trees': (ExtraTreesRegressor(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ), False),
    'Gradient Boosting': (GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ), False),
    'XGBoost': (XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    ), False),
    'LightGBM': (LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    ), False),
    'CatBoost': (CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=0
    ), False),
}

# =============================================================
# 8. CROSS-VALIDATION İLE RMSE HESAPLA
# =============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
print(f"\n{'Model':<25} {'RMSE (CV Ort)':>15} {'RMSE (Std)':>12} {'MAE (CV Ort)':>15} {'R² (CV Ort)':>12} {'Süre (s)':>10}")
print("-" * 95)

from sklearn.base import clone

for name, (model, use_scaled) in models.items():
    X = X_train_scaled if use_scaled else X_train
    
    start = time.time()
    
    # Manuel cross-validation (CatBoost uyumlulugu icin)
    rmse_list, mae_list, r2_list = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)
        
        rmse_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_list.append(mean_absolute_error(y_val, y_pred))
        r2_list.append(r2_score(y_val, y_pred))
    
    rmse_scores = np.array(rmse_list)
    mae_scores = np.array(mae_list)
    r2_scores = np.array(r2_list)
    
    elapsed = time.time() - start
    
    result = {
        'Model': name,
        'RMSE_mean': rmse_scores.mean(),
        'RMSE_std': rmse_scores.std(),
        'MAE_mean': mae_scores.mean(),
        'MAE_std': mae_scores.std(),
        'R2_mean': r2_scores.mean(),
        'R2_std': r2_scores.std(),
        'Time_s': elapsed
    }
    results.append(result)
    
    print(f"{name:<25} {result['RMSE_mean']:>15.6f} {result['RMSE_std']:>12.6f} "
          f"{result['MAE_mean']:>15.6f} {result['R2_mean']:>12.6f} {elapsed:>10.2f}")

results_df = pd.DataFrame(results).sort_values('RMSE_mean')

# =============================================================
# 9. SONUÇLARI GÖRSELLEŞTIR
# =============================================================
print("\n" + "=" * 70)
print("SONUÇLAR (RMSE'ye göre sıralı)")
print("=" * 70)
print(results_df[['Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'R2_mean']].to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE_mean']
print(f"\n[EN IYI] En iyi model: {best_model_name} (RMSE: {best_rmse:.6f})")

# --- Grafik 1: RMSE Karşılaştırma Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sorted_results = results_df.sort_values('RMSE_mean')
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_results)))

bars = axes[0].barh(sorted_results['Model'], sorted_results['RMSE_mean'], 
                     xerr=sorted_results['RMSE_std'], color=colors, edgecolor='white',
                     capsize=3, height=0.6)
axes[0].set_xlabel('RMSE (Cross-Validation)', fontsize=13)
axes[0].set_title('Model RMSE Karşılaştırması (5-Fold CV)', fontsize=15, fontweight='bold')
for bar, val in zip(bars, sorted_results['RMSE_mean']):
    axes[0].text(bar.get_width() + sorted_results['RMSE_std'].max() * 0.3,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
axes[0].invert_yaxis()

# --- Grafik 2: R² Karşılaştırma ---
sorted_r2 = results_df.sort_values('R2_mean', ascending=False)
colors_r2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_r2)))
bars2 = axes[1].barh(sorted_r2['Model'], sorted_r2['R2_mean'],
                      xerr=sorted_r2['R2_std'], color=colors_r2, edgecolor='white',
                      capsize=3, height=0.6)
axes[1].set_xlabel('R² Score (Cross-Validation)', fontsize=13)
axes[1].set_title('Model R² Karşılaştırması (5-Fold CV)', fontsize=15, fontweight='bold')
for bar, val in zip(bars2, sorted_r2['R2_mean']):
    axes[1].text(bar.get_width() + sorted_r2['R2_std'].max() * 0.3,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_model_karsilastirma.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[GRAFIK] Karsilastirma grafigi kaydedildi: analiz_ciktilari/06_model_karsilastirma.png")

# --- Grafik 3: Tüm metriklerin heatmap'i ---
fig, ax = plt.subplots(figsize=(12, 8))
heatmap_data = results_df.set_index('Model')[['RMSE_mean', 'MAE_mean', 'R2_mean', 'Time_s']]
heatmap_data.columns = ['RMSE', 'MAE', 'R²', 'Süre (s)']

# Normalize et (0-1 arası)
heatmap_norm = heatmap_data.copy()
for col in heatmap_norm.columns:
    mn, mx = heatmap_norm[col].min(), heatmap_norm[col].max()
    if mx > mn:
        heatmap_norm[col] = (heatmap_norm[col] - mn) / (mx - mn)

sns.heatmap(heatmap_norm, annot=heatmap_data.round(4), fmt='', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalize Değer'})
ax.set_title('Model Performans Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_model_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[GRAFIK] Heatmap kaydedildi: analiz_ciktilari/07_model_heatmap.png")

# =============================================================
# 10. EN İYİ MODELİ TÜM VERİ İLE EĞİT + TAHMİN
# =============================================================
print("\n" + "=" * 70)
print(f"EN İYİ MODEL İLE TAHMİN: {best_model_name}")
print("=" * 70)

best_model_obj, best_use_scaled = models[best_model_name]

# Tüm train verisi ile eğit
X_final = X_train_scaled if best_use_scaled else X_train
X_test_final = X_test_scaled if best_use_scaled else X_test

best_model_obj.fit(X_final, y_train)

# Feature importance (ağaç tabanlı modeller için)
if hasattr(best_model_obj, 'feature_importances_'):
    importances = best_model_obj.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nEn önemli 15 özellik:")
    print(feat_imp.head(15).to_string(index=False))
    
    # Feature importance grafiği
    fig, ax = plt.subplots(figsize=(14, 8))
    top20 = feat_imp.head(20)
    colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(top20)))
    ax.barh(top20['Feature'][::-1], top20['Importance'][::-1], color=colors_fi)
    ax.set_xlabel('Önem Skoru', fontsize=13)
    ax.set_title(f'{best_model_name} - En Önemli 20 Özellik', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[GRAFIK] Feature importance grafigi kaydedildi: analiz_ciktilari/08_feature_importance.png")

# Test tahmini
predictions = best_model_obj.predict(X_test_final)

# Submission dosyası
submission = pd.DataFrame({
    'id': test_ids.astype(int),
    TARGET: predictions
})
submission_path = os.path.join(BASE_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"\n[DOSYA] Submission dosyasi kaydedildi: {submission_path}")
print(f"   Satır sayısı: {len(submission)}")
print(f"   Tahmin istatistikleri:")
print(f"     Ortalama: {predictions.mean():.4f}")
print(f"     Std:      {predictions.std():.4f}")
print(f"     Min:      {predictions.min():.4f}")
print(f"     Max:      {predictions.max():.4f}")

# =============================================================
# 11. TAHMİN DAĞILIMI GRAFİĞİ
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Train gerçek değerlerin dağılımı
axes[0].hist(y_train, bins=50, alpha=0.7, color='#3498db', edgecolor='white', label='Train (Gerçek)')
axes[0].axvline(y_train.mean(), color='red', linestyle='--', label=f'Ort: {y_train.mean():.2f}')
axes[0].set_title('Train - Gerçek Değer Dağılımı', fontsize=13, fontweight='bold')
axes[0].set_xlabel(TARGET)
axes[0].legend()

# Test tahmin dağılımı
axes[1].hist(predictions, bins=50, alpha=0.7, color='#2ecc71', edgecolor='white', label='Test (Tahmin)')
axes[1].axvline(predictions.mean(), color='red', linestyle='--', label=f'Ort: {predictions.mean():.2f}')
axes[1].set_title('Test - Tahmin Dağılımı', fontsize=13, fontweight='bold')
axes[1].set_xlabel(TARGET)
axes[1].legend()

plt.suptitle(f'Değer Dağılımları ({best_model_name})', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_tahmin_dagilimi.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[GRAFIK] Tahmin dagilimi grafigi kaydedildi: analiz_ciktilari/09_tahmin_dagilimi.png")

# Sonuçları CSV'ye kaydet
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_sonuclari.csv"), index=False)
print(f"[DOSYA] Model sonuclari kaydedildi: analiz_ciktilari/model_sonuclari.csv")

print("\n" + "=" * 70)
print("[OK] MODEL EGITIM VE KARSILASTIRMA TAMAMLANDI!")
print("=" * 70)
