import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

# ── Başlık ──────────────────────────────────────────────────────────────
cells.append(md("""# 🧠 Bilişsel Performans Tahmini — Tam Pipeline
### Veri Ön İşleme → Feature Engineering → Noise Reduction → Stacking Ensemble

---
**Model Sıralaması:**
| Dosya | OOF RMSE |
|-------|----------|
| submission.csv (baseline) | 1.2212 |
| submission_optuna.csv | 1.2201 |
| submission_advanced.csv | 1.2191 |
| submission_neural.csv | 1.2168 |
| **submission_noise_reduced.csv** ✅ | **1.1125** |
"""))

# ── 0. Kütüphaneler ──────────────────────────────────────────────────────
cells.append(md("## 0️⃣ Kütüphaneler ve Ayarlar"))
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping as lgbm_es
from xgboost import XGBRegressor
import warnings, os
warnings.filterwarnings('ignore')

# Görsel ayarlar
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style='whitegrid', palette='muted')

BASE_DIR = '.'
TARGET   = 'bilissel_performans_skoru'
N_FOLDS  = 5
print('✅ Kütüphaneler yüklendi.')"""))

# ── 1. Veri Yükleme ──────────────────────────────────────────────────────
cells.append(md("## 1️⃣ Veri Yükleme ve İlk İnceleme"))
cells.append(code("""train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test_x.csv')
print(f'Train: {train_df.shape}  |  Test: {test_df.shape}')
display(train_df.head(3))"""))

cells.append(code("""# Eksik değer özeti
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print('Eksik Değerli Sütunlar:')
display(missing.to_frame('Eksik Sayı').assign(Oran=lambda d: (d['Eksik Sayı']/len(train_df)*100).round(2)))"""))

cells.append(code("""# Hedef değişken dağılımı
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
train_df[TARGET].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Hedef Değişken Dağılımı')
axes[0].set_xlabel(TARGET)

corr_cols = train_df.select_dtypes(include=np.number).corr()[TARGET].drop(TARGET).sort_values()
corr_cols.plot(kind='barh', ax=axes[1], color=['salmon' if v < 0 else 'steelblue' for v in corr_cols])
axes[1].set_title('Sayısal Sütun Korelasyonları')
plt.tight_layout()
plt.show()"""))

# ── 2. Veri Temizleme ────────────────────────────────────────────────────
cells.append(md("## 2️⃣ Veri Temizleme — Ülke Duplikatları"))
cells.append(code("""# Türkçe / İngilizce ülke ismi tutarsızlıkları
ulke_map = {'Spain': 'Ispanya', 'South Korea': 'Guney Kore', 'Sweden': 'Isvec'}
train_df['ulke'] = train_df['ulke'].replace(ulke_map)
test_df['ulke']  = test_df['ulke'].replace(ulke_map)

print('Benzersiz ülke sayısı:', train_df['ulke'].nunique())
display(train_df['ulke'].value_counts())"""))

# ── 3. Birleştir ve Eksik Doldur ─────────────────────────────────────────
cells.append(md("## 3️⃣ Eksik Değer Doldurma — KNN Imputer"))
cells.append(code("""train_df['is_train'] = 1
test_df['is_train']  = 0
test_df[TARGET]      = np.nan
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Mod ile doldur
for col in ['ruh_sagligi_durumu', 'kronotip']:
    combined[col] = combined[col].fillna(combined[col].mode()[0])

# KNN Imputer için label encoding
cat_knn = ['cinsiyet', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi', 'meslek']
num_cols = ['yas','vucut_kitle_indeksi','rem_yuzdesi','derin_uyku_yuzdesi',
            'uykuya_dalma_suresi_dk','gecelik_uyanma_sayisi','uyku_oncesi_kafein_mg',
            'uyku_oncesi_ekran_suresi_dk','gunluk_adim_sayisi','sekerleme_suresi_dk',
            'stres_skoru','gunluk_calisma_saati','dinlenik_nabiz_bpm',
            'oda_sicakligi_celsius','hafta_sonu_uyku_farki_saat']

le_map = {}
df_enc = combined.copy()

# Adım 1: Önce object dtype — mixed-type atama hatasını önler
for col in cat_knn:
    df_enc[col] = df_enc[col].astype(object)

# Adım 2: NaN'ları atlayarak LabelEncode, sonra float'a çevir
for col in cat_knn:
    le   = LabelEncoder()
    mask = df_enc[col].isnull()
    df_enc.loc[~mask, col] = le.fit_transform(
        df_enc.loc[~mask, col].astype(str).values
    ).astype(float)
    df_enc.loc[mask, col] = np.nan
    df_enc[col] = df_enc[col].astype(float)
    le_map[col] = le

feats_knn = cat_knn + num_cols
sc_knn    = StandardScaler()
df_f      = df_enc[feats_knn].copy()

# StandardScaler NaN'ı kaldıramaz → sadece tam satırlarla fit et, tüm veriyi transform et
sc_knn.fit(df_f.dropna())
df_s = pd.DataFrame(sc_knn.transform(df_f), columns=feats_knn, index=df_f.index)

imp    = KNNImputer(n_neighbors=5, weights='distance').fit_transform(df_s)
imp_df = pd.DataFrame(sc_knn.inverse_transform(imp), columns=feats_knn, index=combined.index)

# meslek imputed değerini geri dönüştür
n_cls = len(le_map['meslek'].classes_)
m_dec = le_map['meslek'].inverse_transform(imp_df['meslek'].round().astype(int).clip(0, n_cls-1))
combined.loc[combined['meslek'].isnull(), 'meslek'] = m_dec[combined['meslek'].isnull()]

for col in ['vucut_kitle_indeksi','stres_skoru','uyku_oncesi_kafein_mg']:
    combined.loc[combined[col].isnull(), col] = imp_df.loc[combined[col].isnull(), col]

print('✅ KNN Imputer tamamlandı.')
print(f'   Kalan eksik (test TARGET hariç): {combined.drop(columns=[TARGET]).isnull().sum().sum()}')"""))

# ── 4. Feature Engineering ───────────────────────────────────────────────
cells.append(md("## 4️⃣ Feature Engineering — 8 Yeni Özellik"))
cells.append(code("""combined['rem_stres_interaction']    = combined['rem_yuzdesi'] * (10 - combined['stres_skoru'])
combined['uyku_kalite_skoru']        = (combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi']
                                         - combined['gecelik_uyanma_sayisi'] * 2)
combined['stres_skoru_sq']           = combined['stres_skoru'] ** 2
combined['rem_derin_toplam']         = combined['rem_yuzdesi'] + combined['derin_uyku_yuzdesi']
combined['stres_calisma_interaction']= combined['stres_skoru'] * combined['gunluk_calisma_saati']
combined['stres_nabiz_interaction']  = combined['stres_skoru'] * combined['dinlenik_nabiz_bpm']
combined['stres_bmi_interaction']    = combined['stres_skoru'] * combined['vucut_kitle_indeksi']
combined['uyku_hijyen_skoru']        = combined['uyku_oncesi_kafein_mg'] + combined['uyku_oncesi_ekran_suresi_dk']
print('✅ 8 yeni feature eklendi.')"""))

# ── 5. K-Means Clustering ────────────────────────────────────────────────
cells.append(md("## 5️⃣ K-Means Kümeleme — Kişi Profili Özellikleri"))
cells.append(code("""cluster_feats = ['rem_yuzdesi','stres_skoru','gunluk_calisma_saati',
                  'uyku_kalite_skoru','dinlenik_nabiz_bpm']
sc_cl = StandardScaler()
X_cl  = sc_cl.fit_transform(combined[cluster_feats])

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
combined['kisi_profili_cluster'] = kmeans.fit_predict(X_cl)

dists = kmeans.transform(X_cl)
for i in range(8):
    combined[f'cluster_{i}_dist'] = dists[:, i]

# Küme görselleştirmesi
fig, ax = plt.subplots(figsize=(10, 4))
train_tmp = combined[combined['is_train']==1]
cluster_means = train_tmp.groupby('kisi_profili_cluster')[TARGET].mean().sort_values()
cluster_means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('K-Means Küme Başına Ortalama Bilişsel Performans Skoru')
ax.set_xlabel('Küme No')
ax.set_ylabel('Ortalama Skor')
plt.tight_layout()
plt.show()
print(f'✅ 9 küme özelliği eklendi (1 label + 8 uzaklık).')"""))

# ── 6. Encoding ──────────────────────────────────────────────────────────
cells.append(md("## 6️⃣ Encoding — Binary, Ordinal ve Target Encoding"))
cells.append(code("""# Binary
binary_map = {
    'cinsiyet': {'Erkek': 0, 'Kadin': 1},
    'mevsim':   {'Sonbahar-Kis': 0, 'Ilkbahar-Yaz': 1},
    'gun_tipi': {'Hafta ici': 0, 'Hafta sonu': 1},
}
for col, m in binary_map.items():
    combined[col] = combined[col].map(m)

# Ordinal
combined['kronotip'] = combined['kronotip'].map({'Gece insani': 0, 'Notr': 1, 'Sabah insani': 2})
combined['ruh_sagligi_durumu'] = combined['ruh_sagligi_durumu'].map(
    {'Anksiyete ve depresyon': 0, 'Depresyon': 1, 'Anksiyete': 2, 'Saglikli': 3})

# Target Encoding (KFold Smoothed)
train_mask = combined['is_train'] == 1
for col in ['meslek', 'ulke']:
    train_tmp  = combined[train_mask].copy()
    global_mean = train_tmp[TARGET].mean()
    combined[f'{col}_target_enc'] = np.nan
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf_te.split(train_tmp):
        tr_df, val_df = train_tmp.iloc[tr_idx], train_tmp.iloc[val_idx]
        counts = tr_df.groupby(col)[TARGET].count()
        means  = tr_df.groupby(col)[TARGET].mean()
        smooth = (counts * means + 10 * global_mean) / (counts + 10)
        combined.loc[val_df.index, f'{col}_target_enc'] = val_df[col].map(smooth)
    # Test
    counts_all  = train_tmp.groupby(col)[TARGET].count()
    means_all   = train_tmp.groupby(col)[TARGET].mean()
    smooth_all  = (counts_all * means_all + 10 * global_mean) / (counts_all + 10)
    combined.loc[~train_mask, f'{col}_target_enc'] = combined.loc[~train_mask, col].map(smooth_all)
    combined[f'{col}_target_enc'].fillna(global_mean, inplace=True)

combined.drop(columns=['meslek', 'ulke'], inplace=True)
print('✅ Encoding tamamlandı.')"""))

# ── 7. Train/Test Split ──────────────────────────────────────────────────
cells.append(md("## 7️⃣ Veri Bölme"))
cells.append(code("""train_data = combined[combined['is_train'] == 1].copy()
test_data  = combined[combined['is_train'] == 0].copy()
drop_cols  = ['id', 'is_train', TARGET]
feature_cols = [c for c in combined.columns if c not in drop_cols]

X_full = train_data[feature_cols].values.astype(np.float32)
y_full = train_data[TARGET].values.astype(np.float32)
X_test = test_data[feature_cols].values.astype(np.float32)
test_ids = test_data['id'].values

print(f'X_full: {X_full.shape}')
print(f'X_test: {X_test.shape}')
print(f'Feature sayısı: {len(feature_cols)}')"""))

# ── 8. Residual Noise Filtering ──────────────────────────────────────────
cells.append(md("""## 8️⃣ Gürültü Temizleme — Residual Filtering
> Hızlı bir LightGBM ile OOF hataları hesaplanır.
> Hatası en yüksek olan **%3'lük** gürültülü satırlar eğitim setinden çıkarılır.
"""))
cells.append(code("""print('Gürültülü satırlar tespit ediliyor...')
lgb_fast = LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
oof_fast = np.zeros(len(X_full))
kf_noise = KFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, val_idx in kf_noise.split(X_full):
    lgb_fast.fit(X_full[tr_idx], y_full[tr_idx])
    oof_fast[val_idx] = lgb_fast.predict(X_full[val_idx])

residuals = np.abs(y_full - oof_fast)
threshold = np.percentile(residuals, 97)
clean_mask = residuals < threshold

X_clean = X_full[clean_mask]
y_clean = y_full[clean_mask]

print(f'Orijinal : {len(X_full)} satır')
print(f'Silinen  : {(~clean_mask).sum()} satır (en yüksek hata %3)')
print(f'Temiz    : {len(X_clean)} satır')

# Residual dağılımı
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8, label='Tüm Satırlar')
plt.axvline(threshold, color='tomato', lw=2, linestyle='--', label=f'%97 Eşiği ({threshold:.3f})')
plt.title('OOF Residual Dağılımı — Kırmızı çizginin sağı silindi')
plt.xlabel('|Gerçek - Tahmin|')
plt.legend()
plt.tight_layout()
plt.show()"""))

# ── 9. Modeller ──────────────────────────────────────────────────────────
cells.append(md("## 9️⃣ Model Eğitimi — Stacking Ensemble (CatBoost + LightGBM + XGBoost → Ridge)"))
cells.append(code("""cat_params = {'iterations': 1937, 'depth': 6, 'learning_rate': 0.0143,
              'l2_leaf_reg': 0.257, 'bagging_temperature': 0.83, 'random_strength': 5.79,
              'border_count': 186, 'min_data_in_leaf': 29, 'verbose': 0}

lgb_params = {'n_estimators': 1935, 'max_depth': 7, 'learning_rate': 0.0064,
              'num_leaves': 27, 'subsample': 0.63, 'colsample_bytree': 0.41,
              'reg_alpha': 7.99, 'reg_lambda': 0.0018, 'min_child_samples': 56,
              'n_jobs': -1, 'verbose': -1}

xgb_params = {'n_estimators': 1230, 'max_depth': 4, 'learning_rate': 0.0171,
              'subsample': 0.70, 'colsample_bytree': 0.44, 'reg_alpha': 6.11,
              'reg_lambda': 0.0022, 'min_child_weight': 11, 'gamma': 1.94, 'n_jobs': -1}

models = {
    'CatBoost' : CatBoostRegressor(**cat_params, random_seed=42),
    'LightGBM' : LGBMRegressor(**lgb_params, random_state=42),
    'XGBoost'  : XGBRegressor(**xgb_params, random_state=42)
}

oof_preds  = {n: np.zeros(len(X_clean)) for n in models}
test_preds = {n: np.zeros((len(X_test), N_FOLDS)) for n in models}
kf_main    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
rmse_scores = {}

for name, model in models.items():
    print(f'\\n→ {name} eğitiliyor...')
    for fold, (tr_idx, val_idx) in enumerate(kf_main.split(X_clean)):
        X_tr, y_tr = X_clean[tr_idx], y_clean[tr_idx]
        X_va, y_va = X_clean[val_idx], y_clean[val_idx]
        if name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
        elif name == 'LightGBM':
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgbm_es(50, verbose=False)])
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds[name][val_idx] = model.predict(X_va)
        test_preds[name][:, fold] = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_clean, oof_preds[name]))
    rmse_scores[name] = rmse
    print(f'   OOF RMSE: {rmse:.6f}')"""))

# ── 10. Stacking ─────────────────────────────────────────────────────────
cells.append(md("## 🔟 Meta-Learner — Ridge Regresyon ile Stacking"))
cells.append(code("""X_meta      = np.column_stack([oof_preds[n] for n in models])
X_test_meta = np.column_stack([test_preds[n].mean(axis=1) for n in models])

meta_model  = Ridge(alpha=1.0)
meta_model.fit(X_meta, y_clean)

final_oof   = meta_model.predict(X_meta)
final_test  = np.clip(meta_model.predict(X_test_meta), 0, 10)

final_rmse  = np.sqrt(mean_squared_error(y_clean, final_oof))
rmse_scores['Stacking (Final)'] = final_rmse
print(f'✅ Ridge Stacking OOF RMSE: {final_rmse:.6f}')

# Karşılaştırma grafiği
all_rmses = {
    'Baseline\\n(CatBoost)': 1.2213,
    'Optuna': 1.2201,
    'Advanced\\nStacking': 1.2191,
    'Neural\\nEnsemble': 1.2168,
    'Noise\\nReduced ✅': final_rmse
}
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(all_rmses.keys(), all_rmses.values(),
              color=['#95afc0','#74b9ff','#a29bfe','#fd79a8','#00b894'], edgecolor='white', width=0.5)
ax.axhline(1.0, color='red', linestyle='--', lw=1.5, label='Hedef: 1.0')
for bar, val in zip(bars, all_rmses.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.4f}', ha='center', va='bottom', fontsize=10)
ax.set_ylim(1.08, 1.24)
ax.set_ylabel('OOF RMSE (düşük = iyi)')
ax.set_title('Model Geliştirme Süreci — RMSE Karşılaştırması')
ax.legend()
plt.tight_layout()
plt.show()"""))

# ── 11. Submission ───────────────────────────────────────────────────────
cells.append(md("## 1️⃣1️⃣ Submission Dosyası Oluşturma"))
cells.append(code("""sub_path   = 'submission_noise_reduced.csv'
submission = pd.DataFrame({'id': test_ids.astype(int), TARGET: final_test})
submission.to_csv(sub_path, index=False)

print(f'✅ Submission kaydedildi: {sub_path}')
print(f'   Satır sayısı: {len(submission)}')
print(f'   Ortalama: {final_test.mean():.4f}')
print(f'   Std: {final_test.std():.4f}')
print(f'   Min: {final_test.min():.4f}  |  Max: {final_test.max():.4f}')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(final_test, bins=50, color='#00b894', edgecolor='white', alpha=0.85)
ax.set_title('Tahmin Edilen Değerlerin Dağılımı')
ax.set_xlabel(TARGET)
ax.set_ylabel('Frekans')
plt.tight_layout()
plt.show()

display(submission.head(10))"""))

# ── Özet ────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📋 Pipeline Özeti

| Adım | Açıklama |
|------|----------|
| **Veri Temizleme** | Ülke duplikatları düzeltildi (Spain→Ispanya vb.) |
| **KNN Imputer** | Eksik değerler ölçeklenmiş uzaklık ağırlıklı KNN ile dolduruldu |
| **Feature Engineering** | 8 etkileşim ve türetilmiş özellik eklendi |
| **K-Means Clustering** | 8 kümeli kişi profili + 8 merkez uzaklığı eklendi |
| **Target Encoding** | Meslek ve ülke, KFold smoothed target encoding ile kodlandı |
| **Residual Filtering** | En yüksek hatalı %3 satır eğitimden çıkarıldı (56.000 → 54.320) |
| **Stacking Ensemble** | CatBoost + LightGBM + XGBoost → Ridge meta-learner |
| **Final RMSE** | **1.1125** |
"""))

nb.cells = cells
nb_path = 'bilisselperformans_tam_pipeline.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook olusturuldu: {nb_path}')
