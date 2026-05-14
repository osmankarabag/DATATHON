import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import json
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'bilissel_performans_skoru'
N_TRIALS = 100  # Her model icin deneme sayisi
N_FOLDS = 5
RANDOM_STATE = 42

# =============================================================
# 1. VERİYİ YÜKLE VE ÖN İŞLE (model_egitim.py ile aynı pipeline)
# =============================================================
print("=" * 70)
print("OPTUNA HIPERPARAMETRE OPTIMIZASYONU")
print("=" * 70)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_x.csv"))

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Birlestir
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# --- Eksik deger doldurma ---
print("\n[1/4] Eksik deger doldurma...")
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

# --- Encoding ---
print("[2/4] Encoding...")
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

combined = pd.get_dummies(combined, columns=['meslek'], prefix='meslek', dtype=int)
combined = pd.get_dummies(combined, columns=['ulke'], prefix='ulke', dtype=int)
print("  Encoding tamamlandi.")

# --- Train/Test ayir ---
train_data = combined[combined['is_train'] == 1].copy()
test_data = combined[combined['is_train'] == 0].copy()
drop_cols = ['id', 'is_train', TARGET]
feature_cols = [c for c in combined.columns if c not in drop_cols]

X_train = train_data[feature_cols].values
y_train = train_data[TARGET].values
X_test = test_data[feature_cols].values
test_ids = test_data['id'].values

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================================
# 2. OPTUNA OBJECTIVE FONKSİYONLARI
# =============================================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_seed': RANDOM_STATE,
        'verbose': 0,
    }
    
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        y_pred = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    
    return np.mean(rmse_scores)


def objective_lightgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgbm_early_stopping(50)])
        y_pred = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    
    return np.mean(rmse_scores)


def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log=True),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  verbose=0)
        y_pred = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    
    return np.mean(rmse_scores)


# =============================================================
# 3. LightGBM early stopping callback
# =============================================================
from lightgbm import early_stopping as lgbm_early_stopping

# =============================================================
# 4. OPTIMIZASYON CALISTIR
# =============================================================
print("\n" + "=" * 70)
print(f"OPTUNA OPTIMIZASYONU BASLIYOR ({N_TRIALS} deneme / model)")
print("=" * 70)

all_results = {}

# --- CatBoost ---
print(f"\n{'='*30} CatBoost {'='*30}")
print(f"  {N_TRIALS} deneme basliyor...")
start = time.time()
study_cat = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study_cat.optimize(objective_catboost, n_trials=N_TRIALS, show_progress_bar=False)
cat_time = time.time() - start
print(f"  Tamamlandi! Sure: {cat_time:.1f}s")
print(f"  En iyi RMSE: {study_cat.best_value:.6f}")
print(f"  En iyi parametreler: {json.dumps(study_cat.best_params, indent=4)}")
all_results['CatBoost'] = {
    'best_rmse': study_cat.best_value,
    'best_params': study_cat.best_params,
    'study': study_cat,
    'time': cat_time
}

# --- LightGBM ---
print(f"\n{'='*30} LightGBM {'='*30}")
print(f"  {N_TRIALS} deneme basliyor...")
start = time.time()
study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study_lgb.optimize(objective_lightgbm, n_trials=N_TRIALS, show_progress_bar=False)
lgb_time = time.time() - start
print(f"  Tamamlandi! Sure: {lgb_time:.1f}s")
print(f"  En iyi RMSE: {study_lgb.best_value:.6f}")
print(f"  En iyi parametreler: {json.dumps(study_lgb.best_params, indent=4)}")
all_results['LightGBM'] = {
    'best_rmse': study_lgb.best_value,
    'best_params': study_lgb.best_params,
    'study': study_lgb,
    'time': lgb_time
}

# --- XGBoost ---
print(f"\n{'='*30} XGBoost {'='*30}")
print(f"  {N_TRIALS} deneme basliyor...")
start = time.time()
study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(objective_xgboost, n_trials=N_TRIALS, show_progress_bar=False)
xgb_time = time.time() - start
print(f"  Tamamlandi! Sure: {xgb_time:.1f}s")
print(f"  En iyi RMSE: {study_xgb.best_value:.6f}")
print(f"  En iyi parametreler: {json.dumps(study_xgb.best_params, indent=4)}")
all_results['XGBoost'] = {
    'best_rmse': study_xgb.best_value,
    'best_params': study_xgb.best_params,
    'study': study_xgb,
    'time': xgb_time
}

# =============================================================
# 5. SONUCLARI KARSILASTIR
# =============================================================
print("\n" + "=" * 70)
print("OPTIMIZASYON SONUCLARI")
print("=" * 70)

comparison = []
# Onceki default sonuclar
default_scores = {'CatBoost': 1.221298, 'LightGBM': 1.230191, 'XGBoost': 1.234663}

for name, res in sorted(all_results.items(), key=lambda x: x[1]['best_rmse']):
    default = default_scores.get(name, 0)
    improvement = default - res['best_rmse']
    pct = (improvement / default * 100) if default > 0 else 0
    comparison.append({
        'Model': name,
        'Onceki RMSE': default,
        'Optuna RMSE': res['best_rmse'],
        'Iyilesme': improvement,
        'Iyilesme (%)': pct,
        'Sure (s)': res['time']
    })
    print(f"\n  {name}:")
    print(f"    Onceki RMSE:  {default:.6f}")
    print(f"    Optuna RMSE:  {res['best_rmse']:.6f}")
    print(f"    Iyilesme:     {improvement:.6f} ({pct:.2f}%)")

comp_df = pd.DataFrame(comparison)
print("\n" + comp_df.to_string(index=False))

# En iyi modeli bul
best_name = comp_df.loc[comp_df['Optuna RMSE'].idxmin(), 'Model']
best_rmse = comp_df['Optuna RMSE'].min()
best_params = all_results[best_name]['best_params']
print(f"\n[EN IYI] {best_name} -> RMSE: {best_rmse:.6f}")

# =============================================================
# 6. GORSELLESTIRME
# =============================================================

# --- Grafik 1: Onceki vs Optuna karsilastirma ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.35
bars1 = ax.bar(x - width/2, comp_df['Onceki RMSE'], width, label='Onceki (Default)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, comp_df['Optuna RMSE'], width, label='Optuna (Optimize)', color='#2ecc71', alpha=0.8)
ax.set_xlabel('Model', fontsize=13)
ax.set_ylabel('RMSE', fontsize=13)
ax.set_title('Hiperparametre Optimizasyonu: Onceki vs Optuna', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comp_df['Model'], fontsize=12)
ax.legend(fontsize=12)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27ae60')

ax.set_ylim(bottom=comp_df[['Onceki RMSE', 'Optuna RMSE']].min().min() * 0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_optuna_karsilastirma.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[GRAFIK] Karsilastirma grafigi: analiz_ciktilari/10_optuna_karsilastirma.png")

# --- Grafik 2: Optuna optimizasyon gecmisi ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
studies = {'CatBoost': study_cat, 'LightGBM': study_lgb, 'XGBoost': study_xgb}
colors_map = {'CatBoost': '#e74c3c', 'LightGBM': '#2ecc71', 'XGBoost': '#3498db'}

for idx, (name, study) in enumerate(studies.items()):
    trials = study.trials
    trial_nums = [t.number for t in trials]
    trial_vals = [t.value for t in trials]
    
    # Her denemenin o ana kadarki en iyi degeri
    best_vals = []
    best_so_far = float('inf')
    for v in trial_vals:
        if v < best_so_far:
            best_so_far = v
        best_vals.append(best_so_far)
    
    axes[idx].scatter(trial_nums, trial_vals, alpha=0.3, s=15, color=colors_map[name], label='Deneme')
    axes[idx].plot(trial_nums, best_vals, color=colors_map[name], linewidth=2, label='En iyi')
    axes[idx].axhline(y=study.best_value, color='red', linestyle='--', alpha=0.5, label=f'Best: {study.best_value:.4f}')
    axes[idx].set_xlabel('Deneme No', fontsize=11)
    axes[idx].set_ylabel('RMSE', fontsize=11)
    axes[idx].set_title(f'{name} Optimizasyon Gecmisi', fontsize=13, fontweight='bold')
    axes[idx].legend(fontsize=9)

plt.suptitle('Optuna Optimizasyon Gecmisleri', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_optuna_gecmisi.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[GRAFIK] Optimizasyon gecmisi: analiz_ciktilari/11_optuna_gecmisi.png")

# --- Grafik 3: En iyi modelin parametre onemleri ---
fig, ax = plt.subplots(figsize=(12, 6))
best_study = all_results[best_name]['study']
importances = optuna.importance.get_param_importances(best_study)
params_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(params_sorted)))
ax.barh(list(params_sorted.keys())[::-1], list(params_sorted.values())[::-1], color=colors_imp)
ax.set_xlabel('Onem', fontsize=13)
ax.set_title(f'{best_name} - Parametre Onemleri', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_parametre_onemleri.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[GRAFIK] Parametre onemleri: analiz_ciktilari/12_parametre_onemleri.png")

# =============================================================
# 7. EN IYI MODEL ILE FINAL TAHMIN
# =============================================================
print("\n" + "=" * 70)
print(f"FINAL TAHMIN: {best_name} (Optimize edilmis)")
print("=" * 70)

# En iyi modeli tum veriyle egit
if best_name == 'CatBoost':
    final_model = CatBoostRegressor(**best_params, random_seed=RANDOM_STATE, verbose=0)
elif best_name == 'LightGBM':
    final_model = LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
else:
    final_model = XGBRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

final_model.fit(X_train, y_train)

# Test tahmini
predictions = final_model.predict(X_test)

# Submission kaydet
submission = pd.DataFrame({
    'id': test_ids.astype(int),
    TARGET: predictions
})
submission_path = os.path.join(BASE_DIR, "submission_optuna.csv")
submission.to_csv(submission_path, index=False)

print(f"\n[DOSYA] Submission: {submission_path}")
print(f"   Satir: {len(submission)}")
print(f"   Ort:   {predictions.mean():.4f}")
print(f"   Std:   {predictions.std():.4f}")
print(f"   Min:   {predictions.min():.4f}")
print(f"   Max:   {predictions.max():.4f}")

# En iyi parametreleri de kaydet
params_path = os.path.join(OUTPUT_DIR, "best_params.json")
with open(params_path, 'w') as f:
    json.dump({
        'best_model': best_name,
        'best_rmse': best_rmse,
        'best_params': best_params,
        'all_results': {k: {'rmse': v['best_rmse'], 'params': v['best_params']} for k, v in all_results.items()}
    }, f, indent=4)
print(f"[DOSYA] Parametreler: {params_path}")

# Tahmin dagilimi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(y_train, bins=50, alpha=0.7, color='#3498db', edgecolor='white', label='Train (Gercek)')
axes[0].axvline(y_train.mean(), color='red', linestyle='--', label=f'Ort: {y_train.mean():.2f}')
axes[0].set_title('Train - Gercek Deger Dagilimi', fontsize=13, fontweight='bold')
axes[0].set_xlabel(TARGET)
axes[0].legend()

axes[1].hist(predictions, bins=50, alpha=0.7, color='#2ecc71', edgecolor='white', label='Test (Optuna Tahmin)')
axes[1].axvline(predictions.mean(), color='red', linestyle='--', label=f'Ort: {predictions.mean():.2f}')
axes[1].set_title(f'Test - Optuna Tahmin Dagilimi ({best_name})', fontsize=13, fontweight='bold')
axes[1].set_xlabel(TARGET)
axes[1].legend()

plt.suptitle(f'Deger Dagilimlari - Optimize {best_name}', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_optuna_tahmin_dagilimi.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[GRAFIK] Tahmin dagilimi: analiz_ciktilari/13_optuna_tahmin_dagilimi.png")

print("\n" + "=" * 70)
print("[OK] OPTUNA OPTIMIZASYONU TAMAMLANDI!")
print("=" * 70)
print(f"\n  Ozet:")
print(f"  - En iyi model: {best_name}")
print(f"  - Onceki RMSE:  {default_scores[best_name]:.6f}")
print(f"  - Optuna RMSE:  {best_rmse:.6f}")
print(f"  - Iyilesme:     {default_scores[best_name] - best_rmse:.6f}")
print(f"  - Submission:   {submission_path}")
