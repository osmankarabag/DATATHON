import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import os, time, json, warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET = 'bilissel_performans_skoru'
N_FOLDS = 5
RANDOM_STATE = 42
N_TRIALS = 200

# =============================================================
# 1. VERİ YÜKLE
# =============================================================
print("=" * 70)
print("GELISMIS MODEL - TUM STRATEJILER")
print("=" * 70)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_x.csv"))
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# =============================================================
# STRATEJI 1: Ulke duplikatlarini birlestir
# =============================================================
print("\n[STRATEJI 1] Ulke duplikatlari birlestiriliyor...")
ulke_map = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
train_df['ulke'] = train_df['ulke'].replace(ulke_map)
test_df['ulke'] = test_df['ulke'].replace(ulke_map)
print(f"  15 ulke -> {train_df['ulke'].nunique()} ulke")

# =============================================================
# 2. BIRLESTIR + EKSIK DEGER DOLDUR
# =============================================================
print("\n[ON ISLEME] Eksik deger doldurma...")
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

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

label_encoders_knn = {}
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
    label_encoders_knn[col] = le

features_for_knn = cat_cols_for_knn + num_cols
scaler_knn = StandardScaler()
df_feat = df_enc[features_for_knn].copy()
scaler_knn.fit(df_feat.dropna())
df_scaled = pd.DataFrame(scaler_knn.transform(df_feat), columns=features_for_knn, index=df_feat.index)

knn_imp = KNNImputer(n_neighbors=5, weights='distance')
imputed = knn_imp.fit_transform(df_scaled)
imputed_df = pd.DataFrame(scaler_knn.inverse_transform(imputed), columns=features_for_knn, index=combined.index)

n_cls = len(label_encoders_knn['meslek'].classes_)
meslek_enc = imputed_df['meslek'].round().astype(int).clip(0, n_cls - 1)
meslek_dec = label_encoders_knn['meslek'].inverse_transform(meslek_enc)
meslek_mask = combined['meslek'].isnull()
combined.loc[meslek_mask, 'meslek'] = meslek_dec[meslek_mask]

for col in ['vucut_kitle_indeksi', 'stres_skoru', 'uyku_oncesi_kafein_mg']:
    null_mask = combined[col].isnull()
    combined.loc[null_mask, col] = imputed_df.loc[null_mask, col]

print("  Eksik degerler dolduruldu.")

# =============================================================
# STRATEJI 2: Feature Engineering
# =============================================================
print("\n[STRATEJI 2] Feature Engineering (11 yeni feature)...")

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

print("  11 yeni feature eklendi.")

# =============================================================
# 3. ENCODING (binary + ordinal)
# =============================================================
print("\n[ON ISLEME] Encoding...")
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

# =============================================================
# STRATEJI 3: Target Encoding (meslek + ulke)
# =============================================================
print("\n[STRATEJI 3] Target Encoding (meslek + ulke)...")

train_data_tmp = combined[combined['is_train'] == 1].copy()
test_data_tmp = combined[combined['is_train'] == 0].copy()

def kfold_target_encode(train_df, test_df, col, target, n_folds=5, smoothing=10):
    """K-Fold target encoding to prevent data leakage."""
    global_mean = train_df[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    train_encoded = pd.Series(np.nan, index=train_df.index)

    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        stats = fold_train.groupby(col)[target].agg(['mean', 'count'])
        smoother = stats['count'] / (stats['count'] + smoothing)
        smoothed_mean = stats['mean'] * smoother + global_mean * (1 - smoother)
        train_encoded.iloc[val_idx] = train_df.iloc[val_idx][col].map(smoothed_mean)

    # Fill any NaN with global mean
    train_encoded = train_encoded.fillna(global_mean)

    # For test: use full train stats
    stats_full = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoother_full = stats_full['count'] / (stats_full['count'] + smoothing)
    smoothed_full = stats_full['mean'] * smoother_full + global_mean * (1 - smoother_full)
    test_encoded = test_df[col].map(smoothed_full).fillna(global_mean)

    return train_encoded, test_encoded

for col in ['meslek', 'ulke']:
    train_enc, test_enc = kfold_target_encode(train_data_tmp, test_data_tmp, col, TARGET)
    combined.loc[combined['is_train'] == 1, f'{col}_target_enc'] = train_enc.values
    combined.loc[combined['is_train'] == 0, f'{col}_target_enc'] = test_enc.values
    print(f"  {col}: target encoded (range: {train_enc.min():.3f} - {train_enc.max():.3f})")

# Drop original meslek and ulke (replaced by target encoding)
combined = combined.drop(columns=['meslek', 'ulke'])
print("  meslek ve ulke one-hot yerine target encoding kullanildi.")

# =============================================================
# 4. TRAIN / TEST AYIR
# =============================================================
train_data = combined[combined['is_train'] == 1].copy()
test_data = combined[combined['is_train'] == 0].copy()

drop_cols = ['id', 'is_train', TARGET]
feature_cols = [c for c in combined.columns if c not in drop_cols]

X_train = train_data[feature_cols].values.astype(np.float32)
y_train = train_data[TARGET].values
X_test = test_data[feature_cols].values.astype(np.float32)
test_ids = test_data['id'].values

print(f"\n  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  Feature sayisi: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

# =============================================================
# STRATEJI 5: Optuna Optimizasyon (200 trial)
# =============================================================
print("\n" + "=" * 70)
print(f"[STRATEJI 5] OPTUNA OPTIMIZASYONU ({N_TRIALS} trial / model)")
print("=" * 70)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_seed': RANDOM_STATE, 'verbose': 0,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        m = CatBoostRegressor(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=(X_train[val_idx], y_train[val_idx]),
              early_stopping_rounds=50, verbose=0)
        scores.append(np.sqrt(mean_squared_error(y_train[val_idx], m.predict(X_train[val_idx]))))
    return np.mean(scores)

def objective_lightgbm(trial):
    from lightgbm import early_stopping as lgbm_es
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        m = LGBMRegressor(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])],
              callbacks=[lgbm_es(50)])
        scores.append(np.sqrt(mean_squared_error(y_train[val_idx], m.predict(X_train[val_idx]))))
    return np.mean(scores)

def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log=True),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        m = XGBRegressor(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])], verbose=0)
        scores.append(np.sqrt(mean_squared_error(y_train[val_idx], m.predict(X_train[val_idx]))))
    return np.mean(scores)

# Run Optuna for each model
optuna_results = {}
for name, obj_fn in [('CatBoost', objective_catboost),
                       ('LightGBM', objective_lightgbm),
                       ('XGBoost', objective_xgboost)]:
    print(f"\n{'='*25} {name} {'='*25}")
    start = time.time()
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(obj_fn, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - start
    print(f"  En iyi RMSE: {study.best_value:.6f} ({elapsed:.0f}s)")
    optuna_results[name] = {'rmse': study.best_value, 'params': study.best_params, 'study': study}

# =============================================================
# STRATEJI 4: Stacking Ensemble
# =============================================================
print("\n" + "=" * 70)
print("[STRATEJI 4] STACKING ENSEMBLE")
print("=" * 70)

# Build base models with best params
cat_params = optuna_results['CatBoost']['params']
lgb_params = optuna_results['LightGBM']['params']
xgb_params = optuna_results['XGBoost']['params']

base_models = {
    'CatBoost': lambda: CatBoostRegressor(**cat_params, random_seed=RANDOM_STATE, verbose=0),
    'LightGBM': lambda: LGBMRegressor(**lgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    'XGBoost': lambda: XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    'ExtraTrees': lambda: ExtraTreesRegressor(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
}

# Generate OOF predictions for stacking
print("\nOut-of-fold predictions olusturuluyor...")
oof_preds = np.zeros((len(X_train), len(base_models)))
test_preds_base = np.zeros((len(X_test), len(base_models)))
oof_rmse = {}

for i, (name, model_fn) in enumerate(base_models.items()):
    print(f"\n  {name}:")
    oof = np.zeros(len(X_train))
    test_fold_preds = np.zeros((len(X_test), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        m = model_fn()
        if name in ['CatBoost']:
            m.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=(X_train[val_idx], y_train[val_idx]),
                  early_stopping_rounds=50, verbose=0)
        elif name in ['LightGBM']:
            from lightgbm import early_stopping as lgbm_es
            m.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[val_idx], y_train[val_idx])],
                  callbacks=[lgbm_es(50)])
        elif name in ['XGBoost']:
            m.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[val_idx], y_train[val_idx])], verbose=0)
        else:
            m.fit(X_train[tr_idx], y_train[tr_idx])

        oof[val_idx] = m.predict(X_train[val_idx])
        test_fold_preds[:, fold] = m.predict(X_test)

    oof_preds[:, i] = oof
    test_preds_base[:, i] = test_fold_preds.mean(axis=1)
    rmse = np.sqrt(mean_squared_error(y_train, oof))
    oof_rmse[name] = rmse
    print(f"    OOF RMSE: {rmse:.6f}")

# Meta-learner: Ridge
print("\n  Meta-learner (Ridge) egitiliyor...")
meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y_train)
stacking_oof = meta.predict(oof_preds)
stacking_rmse = np.sqrt(mean_squared_error(y_train, stacking_oof))
print(f"  Stacking OOF RMSE: {stacking_rmse:.6f}")
print(f"  Ridge weights: {dict(zip(base_models.keys(), meta.coef_.round(4)))}")

# Stacking test predictions
stacking_test = meta.predict(test_preds_base)

# =============================================================
# STRATEJI 6: Post-processing
# =============================================================
print("\n" + "=" * 70)
print("[STRATEJI 6] POST-PROCESSING")
print("=" * 70)

# Clip predictions to [0, 10]
stacking_test = np.clip(stacking_test, 0, 10)

# Also get best single model predictions
best_single_name = min(oof_rmse, key=oof_rmse.get)
best_single_idx = list(base_models.keys()).index(best_single_name)
best_single_test = test_preds_base[:, best_single_idx]

# Blend: stacking + best single model
blend_weights = [0.7, 0.3]
blended_test = blend_weights[0] * stacking_test + blend_weights[1] * best_single_test
blended_test = np.clip(blended_test, 0, 10)

# Check blend OOF RMSE
blended_oof = blend_weights[0] * stacking_oof + blend_weights[1] * oof_preds[:, best_single_idx]
blended_rmse = np.sqrt(mean_squared_error(y_train, blended_oof))
print(f"  Blended OOF RMSE (0.7*stack + 0.3*{best_single_name}): {blended_rmse:.6f}")

# Choose best final prediction
final_options = {
    'Stacking': (stacking_test, stacking_rmse),
    'Blended': (blended_test, blended_rmse),
    best_single_name: (np.clip(best_single_test, 0, 10), oof_rmse[best_single_name]),
}
best_final_name = min(final_options, key=lambda k: final_options[k][1])
best_final_preds = final_options[best_final_name][0]
best_final_rmse = final_options[best_final_name][1]

print(f"\n  En iyi final yontem: {best_final_name} (OOF RMSE: {best_final_rmse:.6f})")

# =============================================================
# SONUCLARI KAYDET
# =============================================================
print("\n" + "=" * 70)
print("SONUCLAR")
print("=" * 70)

# Summary table
print(f"\n{'Yontem':<30} {'OOF RMSE':>12}")
print("-" * 45)
print(f"{'Onceki en iyi (CatBoost)':<30} {'1.2201':>12}")
for name, rmse in sorted(oof_rmse.items(), key=lambda x: x[1]):
    print(f"{name:<30} {rmse:>12.6f}")
print(f"{'Stacking':<30} {stacking_rmse:>12.6f}")
print(f"{'Blended':<30} {blended_rmse:>12.6f}")
print("-" * 45)
print(f"{'FINAL -> ' + best_final_name:<30} {best_final_rmse:>12.6f}")
improvement = 1.2201 - best_final_rmse
print(f"\n  Iyilesme: {improvement:.6f} ({improvement/1.2201*100:.2f}%)")

# Save submission
submission = pd.DataFrame({'id': test_ids.astype(int), TARGET: best_final_preds})
sub_path = os.path.join(BASE_DIR, "submission_advanced.csv")
submission.to_csv(sub_path, index=False)
print(f"\n[DOSYA] Submission: {sub_path}")
print(f"  Ort: {best_final_preds.mean():.4f}, Std: {best_final_preds.std():.4f}")
print(f"  Min: {best_final_preds.min():.4f}, Max: {best_final_preds.max():.4f}")

# Save params
params_out = {
    'best_method': best_final_name,
    'best_oof_rmse': best_final_rmse,
    'previous_rmse': 1.2201,
    'improvement': improvement,
    'oof_rmse_all': oof_rmse,
    'stacking_rmse': stacking_rmse,
    'blended_rmse': blended_rmse,
    'optuna_params': {k: v['params'] for k, v in optuna_results.items()},
}
with open(os.path.join(OUTPUT_DIR, "advanced_params.json"), 'w') as f:
    json.dump(params_out, f, indent=4, default=str)

# =============================================================
# GORSELLESTIRME
# =============================================================
print("\n[GRAFIK] Gorselleştirme...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. RMSE comparison
methods = ['Onceki\n(CatBoost)'] + list(oof_rmse.keys()) + ['Stacking', 'Blended']
rmses = [1.2201] + list(oof_rmse.values()) + [stacking_rmse, blended_rmse]
colors = ['#e74c3c'] + ['#3498db'] * len(oof_rmse) + ['#2ecc71', '#9b59b6']
bars = axes[0].bar(range(len(methods)), rmses, color=colors, edgecolor='white')
axes[0].set_xticks(range(len(methods)))
axes[0].set_xticklabels(methods, fontsize=8, rotation=45, ha='right')
axes[0].set_ylabel('RMSE')
axes[0].set_title('RMSE Karsilastirma', fontweight='bold')
for b, v in zip(bars, rmses):
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Hedef: 1.0')
axes[0].legend()

# 2. Tahmin dagilimi
axes[1].hist(y_train, bins=50, alpha=0.5, color='#3498db', label='Train (Gercek)')
axes[1].hist(best_final_preds, bins=50, alpha=0.5, color='#2ecc71', label=f'Test ({best_final_name})')
axes[1].set_title('Deger Dagilimlari', fontweight='bold')
axes[1].legend()

# 3. OOF scatter
axes[2].scatter(y_train, stacking_oof, alpha=0.05, s=5, color='#3498db')
axes[2].plot([0, 10], [0, 10], 'r--', alpha=0.5)
axes[2].set_xlabel('Gercek')
axes[2].set_ylabel('Tahmin')
axes[2].set_title(f'Stacking OOF (RMSE: {stacking_rmse:.4f})', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_gelismis_sonuclar.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Kaydedildi: analiz_ciktilari/14_gelismis_sonuclar.png")

print("\n" + "=" * 70)
print("[OK] GELISMIS MODEL TAMAMLANDI!")
print("=" * 70)
