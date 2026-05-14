import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from lightgbm import early_stopping as lgbm_es
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET = 'bilissel_performans_skoru'
N_FOLDS = 5

print("=" * 70)
print("NOISE REDUCTION MODEL - K-MEANS & RESIDUAL FILTERING")
print("=" * 70)

# =============================================================
# 1. VERI YUKLE + ON ISLEME
# =============================================================
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_x.csv"))

ulke_map = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
train_df['ulke'] = train_df['ulke'].replace(ulke_map)
test_df['ulke'] = test_df['ulke'].replace(ulke_map)

train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

print("[1/6] Eksik deger doldurma...")
for col in ['ruh_sagligi_durumu', 'kronotip']:
    combined[col] = combined[col].astype(object)
    combined[col] = combined[col].fillna(combined[col].mode()[0])

combined['meslek'] = combined['meslek'].astype(object)
cat_cols_for_knn = ['cinsiyet', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi', 'meslek']
num_cols = ['yas', 'vucut_kitle_indeksi', 'rem_yuzdesi', 'derin_uyku_yuzdesi',
            'uykuya_dalma_suresi_dk', 'gecelik_uyanma_sayisi', 'uyku_oncesi_kafein_mg',
            'uyku_oncesi_ekran_suresi_dk', 'gunluk_adim_sayisi', 'sekerleme_suresi_dk',
            'stres_skoru', 'gunluk_calisma_saati', 'dinlenik_nabiz_bpm',
            'oda_sicakligi_celsius', 'hafta_sonu_uyku_farki_saat']

le_knn = {}
df_enc = combined.copy()
for col in cat_cols_for_knn:
    df_enc[col] = df_enc[col].astype(object)
    df_enc.loc[combined[col].isnull(), col] = np.nan
for col in cat_cols_for_knn:
    le = LabelEncoder()
    mask = df_enc[col].isnull()
    df_enc.loc[~mask, col] = le.fit_transform(df_enc.loc[~mask, col].values).astype(float)
    df_enc.loc[mask, col] = np.nan
    df_enc[col] = df_enc[col].astype(float)
    le_knn[col] = le

feats_knn = cat_cols_for_knn + num_cols
sc_knn = StandardScaler()
df_f = df_enc[feats_knn].copy()
sc_knn.fit(df_f.dropna())
df_s = pd.DataFrame(sc_knn.transform(df_f), columns=feats_knn, index=df_f.index)
knn = KNNImputer(n_neighbors=5, weights='distance')
imp = knn.fit_transform(df_s)
imp_df = pd.DataFrame(sc_knn.inverse_transform(imp), columns=feats_knn, index=combined.index)

n_cls = len(le_knn['meslek'].classes_)
m_enc = imp_df['meslek'].round().astype(int).clip(0, n_cls - 1)
m_dec = le_knn['meslek'].inverse_transform(m_enc)
m_mask = combined['meslek'].isnull()
combined.loc[m_mask, 'meslek'] = m_dec[m_mask]

for col in ['vucut_kitle_indeksi', 'stres_skoru', 'uyku_oncesi_kafein_mg']:
    null_mask = combined[col].isnull()
    combined.loc[null_mask, col] = imp_df.loc[null_mask, col]

# =============================================================
# 2. FEATURE ENGINEERING & K-MEANS CLUSTERING
# =============================================================
print("[2/6] Feature Engineering ve Clustering...")

combined['rem_stres_interaction'] = combined['rem_yuzdesi'] * (10 - combined['stres_skoru'])
combined['uyku_kalite_skoru'] = (combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi'] - combined['gecelik_uyanma_sayisi'] * 2)
combined['stres_skoru_sq'] = combined['stres_skoru'] ** 2
combined['rem_derin_toplam'] = combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi']
combined['stres_calisma_interaction'] = combined['stres_skoru'] * combined['gunluk_calisma_saati']
combined['stres_nabiz_interaction'] = combined['stres_skoru'] * combined['dinlenik_nabiz_bpm']
combined['stres_bmi_interaction'] = combined['stres_skoru'] * combined['vucut_kitle_indeksi']
combined['uyku_hijyen_skoru'] = combined['uyku_oncesi_kafein_mg'] + combined['uyku_oncesi_ekran_suresi_dk']

# K-Means Clustering Features
cluster_features = ['rem_yuzdesi', 'stres_skoru', 'gunluk_calisma_saati', 'uyku_kalite_skoru', 'dinlenik_nabiz_bpm']
sc_cluster = StandardScaler()
scaled_clusters = sc_cluster.fit_transform(combined[cluster_features])

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
combined['kisi_profili_cluster'] = kmeans.fit_predict(scaled_clusters)

# Her merkeze olan uzakliklari da feature olarak ekleyelim
distances = kmeans.transform(scaled_clusters)
for i in range(8):
    combined[f'cluster_{i}_dist'] = distances[:, i]

# =============================================================
# 3. ENCODING
# =============================================================
print("[3/6] Encoding islemleri...")
binary_mappings = {
    'cinsiyet': {'Erkek': 0, 'Kadin': 1},
    'mevsim': {'Sonbahar-Kis': 0, 'Ilkbahar-Yaz': 1},
    'gun_tipi': {'Hafta ici': 0, 'Hafta sonu': 1},
}
for col, mapping in binary_mappings.items():
    combined[col] = combined[col].map(mapping)

combined['kronotip'] = combined['kronotip'].map({'Gece insani': 0, 'Notr': 1, 'Sabah insani': 2})
combined['ruh_sagligi_durumu'] = combined['ruh_sagligi_durumu'].map(
    {'Anksiyete ve depresyon': 0, 'Depresyon': 1, 'Anksiyete': 2, 'Saglikli': 3})

# Target Encoding
train_mask = combined['is_train'] == 1
for col in ['meslek', 'ulke']:
    train_tmp = combined[train_mask].copy()
    global_mean = train_tmp[TARGET].mean()
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    combined[f'{col}_target_enc'] = np.nan
    
    # Train icin out-of-fold encoding
    for tr_idx, val_idx in kf_te.split(train_tmp):
        tr_df, val_df = train_tmp.iloc[tr_idx], train_tmp.iloc[val_idx]
        smoothing = 10
        counts = tr_df.groupby(col)[TARGET].count()
        means = tr_df.groupby(col)[TARGET].mean()
        smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
        combined.loc[val_df.index, f'{col}_target_enc'] = val_df[col].map(smooth_means)
        
    # Test icin tum train datasindan encoding
    counts_all = train_tmp.groupby(col)[TARGET].count()
    means_all = train_tmp.groupby(col)[TARGET].mean()
    smooth_means_all = (counts_all * means_all + smoothing * global_mean) / (counts_all + smoothing)
    combined.loc[~train_mask, f'{col}_target_enc'] = combined.loc[~train_mask, col].map(smooth_means_all)
    
    # NaN doldur (gormedigi class olursa)
    combined[f'{col}_target_enc'].fillna(global_mean, inplace=True)

combined = combined.drop(columns=['meslek', 'ulke'])

# =============================================================
# 4. TRAIN/TEST AYIR & RESIDUAL ANALYSIS (Gürültü Temizleme)
# =============================================================
print("[4/6] Veriyi ayirma ve Residual (Gürültü) Analizi...")
train_data = combined[combined['is_train'] == 1].copy()
test_data = combined[combined['is_train'] == 0].copy()
drop_cols = ['id', 'is_train', TARGET]
feature_cols = [c for c in combined.columns if c not in drop_cols]

X_train_full = train_data[feature_cols].values.astype(np.float32)
y_train_full = train_data[TARGET].values.astype(np.float32)
X_test = test_data[feature_cols].values.astype(np.float32)
test_ids = test_data['id'].values

# Hizli bir LightGBM ile Residual hesaplama
print("  - Gürültülü satirlar tespit ediliyor (Fast LightGBM)...")
lgb_fast = LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
oof_fast = np.zeros(len(X_train_full))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, val_idx in kf.split(X_train_full):
    lgb_fast.fit(X_train_full[tr_idx], y_train_full[tr_idx])
    oof_fast[val_idx] = lgb_fast.predict(X_train_full[val_idx])

# Hatalari hesapla
residuals = np.abs(y_train_full - oof_fast)
error_threshold = np.percentile(residuals, 97) # En kötü %3'luk kismi at

clean_mask = residuals < error_threshold
X_train_clean = X_train_full[clean_mask]
y_train_clean = y_train_full[clean_mask]

print(f"  - Orijinal Egitim Seti: {len(X_train_full)} satir")
print(f"  - En fazla hata üreten %3 satir silindi.")
print(f"  - Temizlenmis Egitim Seti: {len(X_train_clean)} satir")

# =============================================================
# 5. MODEL EGITIMI (Temizlenmis Veri Uzerinde)
# =============================================================
print("\n[5/6] Modeller egitiliyor (Stacking)...")

# Optuna'dan alinan optimize edilmis parametreler
cat_params = {'iterations': 1937, 'depth': 6, 'learning_rate': 0.0143, 'l2_leaf_reg': 0.257, 
              'bagging_temperature': 0.83, 'random_strength': 5.79, 'border_count': 186, 
              'min_data_in_leaf': 29, 'verbose': 0}

lgb_params = {'n_estimators': 1935, 'max_depth': 7, 'learning_rate': 0.0064, 'num_leaves': 27, 
              'subsample': 0.63, 'colsample_bytree': 0.41, 'reg_alpha': 7.99, 'reg_lambda': 0.0018, 
              'min_child_samples': 56, 'n_jobs': -1, 'verbose': -1}

xgb_params = {'n_estimators': 1230, 'max_depth': 4, 'learning_rate': 0.0171, 'subsample': 0.70, 
              'colsample_bytree': 0.44, 'reg_alpha': 6.11, 'reg_lambda': 0.0022, 
              'min_child_weight': 11, 'gamma': 1.94, 'n_jobs': -1}

models = {
    'CatBoost': CatBoostRegressor(**cat_params, random_seed=42),
    'LightGBM': LGBMRegressor(**lgb_params, random_state=42),
    'XGBoost': XGBRegressor(**xgb_params, random_state=42)
}

oof_preds = {name: np.zeros(len(X_train_clean)) for name in models.keys()}
test_preds = {name: np.zeros((len(X_test), N_FOLDS)) for name in models.keys()}

kf_main = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"  -> {name} egitiliyor...")
    for fold, (tr_idx, val_idx) in enumerate(kf_main.split(X_train_clean)):
        X_tr, y_tr = X_train_clean[tr_idx], y_train_clean[tr_idx]
        X_va, y_va = X_train_clean[val_idx], y_train_clean[val_idx]
        
        if name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
        elif name == 'LightGBM':
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgbm_es(50, verbose=False)])
        else: # XGBoost
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
        oof_preds[name][val_idx] = model.predict(X_va)
        test_preds[name][:, fold] = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_train_clean, oof_preds[name]))
    print(f"     {name} Temizlenmis OOF RMSE: {rmse:.6f}")

# =============================================================
# 6. STACKING VE SONUCLANDIRMA
# =============================================================
print("\n[6/6] Meta-learner (Ridge) ile Stacking...")
X_oof = np.column_stack([oof_preds[n] for n in models.keys()])
X_test_meta = np.column_stack([test_preds[n].mean(axis=1) for n in models.keys()])

meta_model = Ridge(alpha=1.0)
meta_model.fit(X_oof, y_train_clean)

final_oof = meta_model.predict(X_oof)
final_test = meta_model.predict(X_test_meta)

# Clip to 0-10 bounds
final_test = np.clip(final_test, 0, 10)

final_rmse = np.sqrt(mean_squared_error(y_train_clean, final_oof))
print(f"  => Ridge Stacking OOF RMSE: {final_rmse:.6f}")

sub_path = os.path.join(BASE_DIR, "submission_noise_reduced.csv")
submission = pd.DataFrame({'id': test_ids.astype(int), TARGET: final_test})
submission.to_csv(sub_path, index=False)

print("\n" + "=" * 70)
print(f"[OK] Islemler Tamam! Dosya: {sub_path}")
print(f"Submission istatistikleri: Ort={final_test.mean():.4f}, Std={final_test.std():.4f}")
print("=" * 70)
