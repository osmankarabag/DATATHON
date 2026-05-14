import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from lightgbm import early_stopping as lgbm_es
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import os, time, json, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET = 'bilissel_performans_skoru'
N_FOLDS = 5

print("=" * 70)
print("NEURAL ENSEMBLE - TABNET + MULTI-SEED CATBOOST")
print("=" * 70)

# =============================================================
# 1. VERI YUKLE + ON ISLEME (onceki pipeline ayni)
# =============================================================
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_x.csv"))
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Ulke duplikatlari
ulke_map = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
train_df['ulke'] = train_df['ulke'].replace(ulke_map)
test_df['ulke'] = test_df['ulke'].replace(ulke_map)

# Birlestir
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Eksik deger doldurma
print("[1/5] Eksik deger doldurma...")
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
# 2. AGRESIF FEATURE ENGINEERING
# =============================================================
print("[2/5] Agresif Feature Engineering...")

# Temel etkilesimler
combined['rem_stres_interaction'] = combined['rem_yuzdesi'] * (10 - combined['stres_skoru'])
combined['uyku_kalite_skoru'] = (combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi']
                                  - combined['gecelik_uyanma_sayisi'] * 2)
combined['stres_skoru_sq'] = combined['stres_skoru'] ** 2
combined['rem_derin_toplam'] = combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi']
combined['stres_calisma_interaction'] = combined['stres_skoru'] * combined['gunluk_calisma_saati']
combined['stres_nabiz_interaction'] = combined['stres_skoru'] * combined['dinlenik_nabiz_bpm']
combined['stres_sicaklik_interaction'] = combined['stres_skoru'] * combined['oda_sicakligi_celsius']
combined['stres_bmi_interaction'] = combined['stres_skoru'] * combined['vucut_kitle_indeksi']
combined['rem_sq'] = combined['rem_yuzdesi'] ** 2
combined['uyku_hijyen_skoru'] = combined['uyku_oncesi_kafein_mg'] + combined['uyku_oncesi_ekran_suresi_dk']
combined['calisma_sq'] = combined['gunluk_calisma_saati'] ** 2

# YENI: Daha agresif featurelar
combined['rem_stres_ratio'] = combined['rem_yuzdesi'] / (combined['stres_skoru'] + 0.1)
combined['derin_stres_interaction'] = combined['derin_uyku_yuzdesi'] * (10 - combined['stres_skoru'])
combined['toplam_uyku_kalite'] = (combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi']) * (10 - combined['stres_skoru'])
combined['uyanma_stres'] = combined['gecelik_uyanma_sayisi'] * combined['stres_skoru']
combined['uyku_dalma_stres'] = combined['uykuya_dalma_suresi_dk'] * combined['stres_skoru']
combined['rem_calisma'] = combined['rem_yuzdesi'] * (10 - combined['gunluk_calisma_saati'])
combined['adim_rem'] = combined['gunluk_adim_sayisi'] / 1000 * combined['rem_yuzdesi']
combined['stres_inverse'] = 10 - combined['stres_skoru']
combined['rem_cubed'] = combined['rem_yuzdesi'] ** 3 / 1000
combined['stres_cubed'] = combined['stres_skoru'] ** 3 / 100
combined['nabiz_sicaklik'] = combined['dinlenik_nabiz_bpm'] * combined['oda_sicakligi_celsius']

# Frequency encoding
for col in ['meslek', 'ulke']:
    freq = combined[col].value_counts(normalize=True)
    combined[f'{col}_freq'] = combined[col].map(freq)

# Grup istatistikleri (grup ortalamaları)
train_tmp = combined[combined['is_train'] == 1]
for col in ['meslek', 'ulke', 'ruh_sagligi_durumu']:
    grp_stats = train_tmp.groupby(col)[TARGET].agg(['mean', 'std', 'median'])
    combined[f'{col}_target_mean'] = combined[col].map(grp_stats['mean'])
    combined[f'{col}_target_std'] = combined[col].map(grp_stats['std'])
    combined[f'{col}_target_median'] = combined[col].map(grp_stats['median'])

# gun_tipi bazli hedef ortalama
gt_mean = train_tmp.groupby('gun_tipi')[TARGET].mean()
combined['gun_tipi_target_mean'] = combined['gun_tipi'].map(gt_mean)

feat_count = len([c for c in combined.columns if c not in ['id', 'is_train', TARGET, 'meslek', 'ulke']])
print(f"  {feat_count} feature olusturuldu.")

# =============================================================
# 3. ENCODING
# =============================================================
print("[3/5] Encoding...")
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

# meslek ve ulke artik target encoding ile temsil ediliyor, drop et
combined = combined.drop(columns=['meslek', 'ulke'])

# =============================================================
# 4. TRAIN/TEST AYIR
# =============================================================
train_data = combined[combined['is_train'] == 1].copy()
test_data = combined[combined['is_train'] == 0].copy()
drop_cols = ['id', 'is_train', TARGET]
feature_cols = [c for c in combined.columns if c not in drop_cols]

X_train = train_data[feature_cols].values.astype(np.float32)
y_train = train_data[TARGET].values.astype(np.float32)
X_test = test_data[feature_cols].values.astype(np.float32)
test_ids = test_data['id'].values

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  Features: {len(feature_cols)}")

# =============================================================
# 5. MULTI-SEED CATBOOST ENSEMBLE
# =============================================================
print("\n" + "=" * 70)
print("[4/5] MULTI-SEED CATBOOST + LIGHTGBM ENSEMBLE")
print("=" * 70)

# Onceki en iyi CatBoost parametreleri
cat_params = {
    'iterations': 1937, 'depth': 6, 'learning_rate': 0.014362470802740026,
    'l2_leaf_reg': 0.25775178948108174, 'bagging_temperature': 0.8325212455806807,
    'random_strength': 5.7946027678891925, 'border_count': 186, 'min_data_in_leaf': 29,
    'verbose': 0,
}

lgb_params = {
    'n_estimators': 1935, 'max_depth': 7, 'learning_rate': 0.006483268176267772,
    'num_leaves': 27, 'subsample': 0.6326833112555242,
    'colsample_bytree': 0.4137279147083351, 'reg_alpha': 7.990934218390473,
    'reg_lambda': 0.0018077171648385587, 'min_child_samples': 56,
    'n_jobs': -1, 'verbose': -1,
}

SEEDS = [42, 123, 456, 789, 2024]
kf_main = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

all_oof = []
all_test = []
model_names = []

# Multi-seed CatBoost
for seed in SEEDS:
    print(f"\n  CatBoost seed={seed}:")
    oof = np.zeros(len(X_train))
    test_fold = np.zeros((len(X_test), N_FOLDS))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        m = CatBoostRegressor(**cat_params, random_seed=seed)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=(X_train[val_idx], y_train[val_idx]),
              early_stopping_rounds=50, verbose=0)
        oof[val_idx] = m.predict(X_train[val_idx])
        test_fold[:, fold] = m.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_train, oof))
    print(f"    OOF RMSE: {rmse:.6f}")
    all_oof.append(oof)
    all_test.append(test_fold.mean(axis=1))
    model_names.append(f'CatBoost_s{seed}')

# Multi-seed LightGBM
for seed in SEEDS:
    print(f"\n  LightGBM seed={seed}:")
    oof = np.zeros(len(X_train))
    test_fold = np.zeros((len(X_test), N_FOLDS))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        m = LGBMRegressor(**lgb_params, random_state=seed)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])],
              callbacks=[lgbm_es(50)])
        oof[val_idx] = m.predict(X_train[val_idx])
        test_fold[:, fold] = m.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_train, oof))
    print(f"    OOF RMSE: {rmse:.6f}")
    all_oof.append(oof)
    all_test.append(test_fold.mean(axis=1))
    model_names.append(f'LightGBM_s{seed}')

# =============================================================
# 6. TABNET
# =============================================================
print("\n" + "=" * 70)
print("[5/5] TABNET DEEP LEARNING")
print("=" * 70)

# Scaling for TabNet
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

for seed in [42, 123, 456]:
    print(f"\n  TabNet seed={seed}:")
    oof = np.zeros(len(X_train))
    test_fold = np.zeros((len(X_test), N_FOLDS))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_sc)):
        tabnet = TabNetRegressor(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params=dict(T_max=200, eta_min=1e-4),
            mask_type='entmax', seed=seed, verbose=0,
        )
        tabnet.fit(
            X_train_sc[tr_idx], y_train[tr_idx].reshape(-1, 1),
            eval_set=[(X_train_sc[val_idx], y_train[val_idx].reshape(-1, 1))],
            eval_metric=['rmse'],
            max_epochs=200, patience=30, batch_size=2048,
            virtual_batch_size=256, drop_last=False,
        )
        oof[val_idx] = tabnet.predict(X_train_sc[val_idx]).flatten()
        test_fold[:, fold] = tabnet.predict(X_test_sc).flatten()

    rmse = np.sqrt(mean_squared_error(y_train, oof))
    print(f"    OOF RMSE: {rmse:.6f}")
    all_oof.append(oof)
    all_test.append(test_fold.mean(axis=1))
    model_names.append(f'TabNet_s{seed}')

# =============================================================
# 7. MEGA STACKING
# =============================================================
print("\n" + "=" * 70)
print("MEGA STACKING ENSEMBLE")
print("=" * 70)

oof_matrix = np.column_stack(all_oof)
test_matrix = np.column_stack(all_test)

print(f"  Base model sayisi: {len(model_names)}")
for i, name in enumerate(model_names):
    rmse = np.sqrt(mean_squared_error(y_train, all_oof[i]))
    print(f"    {name}: {rmse:.6f}")

# Basit ortalama
avg_oof = oof_matrix.mean(axis=1)
avg_test = test_matrix.mean(axis=1)
avg_rmse = np.sqrt(mean_squared_error(y_train, avg_oof))
print(f"\n  Basit Ortalama OOF RMSE: {avg_rmse:.6f}")

# Weighted - sadece CatBoost + LightGBM ortalama
cat_oof = np.column_stack([all_oof[i] for i, n in enumerate(model_names) if 'CatBoost' in n])
cat_test = np.column_stack([all_test[i] for i, n in enumerate(model_names) if 'CatBoost' in n])
lgb_oof = np.column_stack([all_oof[i] for i, n in enumerate(model_names) if 'LightGBM' in n])
lgb_test = np.column_stack([all_test[i] for i, n in enumerate(model_names) if 'LightGBM' in n])
tab_oof = np.column_stack([all_oof[i] for i, n in enumerate(model_names) if 'TabNet' in n])
tab_test = np.column_stack([all_test[i] for i, n in enumerate(model_names) if 'TabNet' in n])

cat_avg_oof = cat_oof.mean(axis=1)
lgb_avg_oof = lgb_oof.mean(axis=1)
tab_avg_oof = tab_oof.mean(axis=1)
cat_avg_test = cat_test.mean(axis=1)
lgb_avg_test = lgb_test.mean(axis=1)
tab_avg_test = tab_test.mean(axis=1)

print(f"  CatBoost avg RMSE: {np.sqrt(mean_squared_error(y_train, cat_avg_oof)):.6f}")
print(f"  LightGBM avg RMSE: {np.sqrt(mean_squared_error(y_train, lgb_avg_oof)):.6f}")
print(f"  TabNet avg RMSE: {np.sqrt(mean_squared_error(y_train, tab_avg_oof)):.6f}")

# Ridge meta-learner
meta = Ridge(alpha=1.0)
meta.fit(oof_matrix, y_train)
stack_oof = meta.predict(oof_matrix)
stack_test = meta.predict(test_matrix)
stack_rmse = np.sqrt(mean_squared_error(y_train, stack_oof))
print(f"\n  Ridge Stacking OOF RMSE: {stack_rmse:.6f}")

# Farkli blend denemeleri
best_blend_rmse = 999
best_blend_name = ""
best_blend_test = None

for w_cat in np.arange(0.4, 0.8, 0.05):
    for w_lgb in np.arange(0.1, 0.5, 0.05):
        w_tab = 1.0 - w_cat - w_lgb
        if w_tab < 0 or w_tab > 0.3:
            continue
        blend = w_cat * cat_avg_oof + w_lgb * lgb_avg_oof + w_tab * tab_avg_oof
        rmse = np.sqrt(mean_squared_error(y_train, blend))
        if rmse < best_blend_rmse:
            best_blend_rmse = rmse
            best_blend_name = f"Cat:{w_cat:.2f} LGB:{w_lgb:.2f} Tab:{w_tab:.2f}"
            best_blend_test = w_cat * cat_avg_test + w_lgb * lgb_avg_test + w_tab * tab_avg_test

print(f"\n  En iyi blend: {best_blend_name}")
print(f"  Blend OOF RMSE: {best_blend_rmse:.6f}")

# En iyi yontemi sec
options = {
    'Basit Ortalama': (avg_test, avg_rmse),
    'Ridge Stacking': (stack_test, stack_rmse),
    'Best Blend': (best_blend_test, best_blend_rmse),
}
best_name = min(options, key=lambda k: options[k][1])
best_preds = np.clip(options[best_name][0], 0, 10)
best_rmse = options[best_name][1]

print(f"\n  FINAL: {best_name} -> OOF RMSE: {best_rmse:.6f}")

# =============================================================
# SONUCLAR
# =============================================================
print("\n" + "=" * 70)
print("SONUCLAR")
print("=" * 70)
print(f"  Onceki en iyi:  1.2191 (Stacking)")
print(f"  Yeni en iyi:    {best_rmse:.6f} ({best_name})")
print(f"  Iyilesme:       {1.2191 - best_rmse:.6f}")

# Submission kaydet
submission = pd.DataFrame({'id': test_ids.astype(int), TARGET: best_preds})
sub_path = os.path.join(BASE_DIR, "submission_neural.csv")
submission.to_csv(sub_path, index=False)
print(f"\n[DOSYA] Submission: {sub_path}")
print(f"  Ort: {best_preds.mean():.4f}, Std: {best_preds.std():.4f}")

# Params kaydet
params_out = {
    'best_method': best_name, 'best_rmse': float(best_rmse),
    'avg_rmse': float(avg_rmse), 'stack_rmse': float(stack_rmse),
    'blend_rmse': float(best_blend_rmse), 'blend_weights': best_blend_name,
    'model_rmses': {n: float(np.sqrt(mean_squared_error(y_train, all_oof[i])))
                    for i, n in enumerate(model_names)},
}
with open(os.path.join(OUTPUT_DIR, "neural_params.json"), 'w') as f:
    json.dump(params_out, f, indent=4)

print("\n" + "=" * 70)
print("[OK] NEURAL ENSEMBLE TAMAMLANDI!")
print("=" * 70)
