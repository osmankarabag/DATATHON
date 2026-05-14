# 🧠 Bilişsel Performans Tahmini — Datathon

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-yellow?style=for-the-badge&logo=yandex&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-brightgreen?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Opt-purple?style=for-the-badge)

**Uyku kalitesi, stres ve yaşam alışkanlıkları verilerinden bilişsel performansı tahmin eden makine öğrenmesi pipeline'ı.**

</div>

---

## 📌 Proje Özeti

Bu proje, bir datathon kapsamında geliştirilmiştir. Hedef değişken **`bilissel_performans_skoru`** (0–10 arası sürekli değer) olup; uyku düzeni, stres, yaş, meslek, kafein tüketimi ve benzeri özellikler kullanılarak regresyon modelleri ile tahmin edilmektedir.

**En iyi elde edilen OOF RMSE: `1.2191`**

---

## 📁 Proje Yapısı

```
DATATHON/
│
├── 📂 analiz_ciktilari/          # Grafik çıktıları ve parametre dosyaları
│   ├── 01_eksik_deger_dagilimi.png
│   ├── 02_eksik_deger_deseni.png
│   ├── 03_aykiri_deger_oranlari.png
│   ├── 04_boxplotlar.png
│   ├── 05_histogramlar.png
│   ├── 06_model_karsilastirma.png
│   ├── 07_model_heatmap.png
│   ├── 08_feature_importance.png
│   ├── 09_tahmin_dagilimi.png
│   ├── 10_optuna_karsilastirma.png
│   ├── 11_optuna_gecmisi.png
│   ├── 12_parametre_onemleri.png
│   ├── 13_optuna_tahmin_dagilimi.png
│   ├── 14_gelismis_sonuclar.png
│   ├── best_params.json          # Optuna en iyi parametreler
│   ├── advanced_params.json      # Stacking sonuçları
│   └── model_sonuclari.csv       # Tüm model karşılaştırması
│
├── 📓 bilisselperformans_tam_pipeline.ipynb   # Tam pipeline notebook
├── 📓 veri_on_isleme.ipynb                    # Ön işleme notebook
│
├── 🐍 eksik_aykiri_analiz.py     # EDA: Eksik ve aykırı değer analizi
├── 🐍 eksik_deger_doldurma.py    # KNN tabanlı eksik değer doldurma
├── 🐍 encoding_islemi.py         # Encoding stratejileri
├── 🐍 model_egitim.py            # 11 model karşılaştırması (5-Fold CV)
├── 🐍 optuna_optimizasyon.py     # Bayesian hiperparametre optimizasyonu
├── 🐍 gelismis_model.py          # Stacking ensemble + gelişmiş stratejiler
├── 🐍 noise_reduction_model.py   # Gürültü azaltma yaklaşımı
├── 🐍 neural_ensemble.py         # Sinir ağı tabanlı ensemble
│
├── 📊 submission.csv             # Temel model submission
├── 📊 submission_optuna.csv      # Optuna optimize submission
├── 📊 submission_advanced.csv    # Stacking ensemble submission
├── 📊 submission_noise_reduced.csv
└── 📊 submission_neural.csv
```

---

## 🔄 Pipeline Akışı

```
Ham Veri (train.csv + test_x.csv)
         │
         ▼
┌─────────────────────┐
│  1. EDA & Analiz    │  ← eksik_aykiri_analiz.py
│  Eksik & Aykırı     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Ön İşleme       │  ← eksik_deger_doldurma.py
│  KNN Imputation     │
│  Encoding           │  ← encoding_islemi.py
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Model Eğitimi   │  ← model_egitim.py
│  11 Model, 5-Fold   │
│  CV Karşılaştırması │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Optuna Opt.     │  ← optuna_optimizasyon.py
│  100 trial/model    │
│  CatBoost/LGB/XGB   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Stacking        │  ← gelismis_model.py
│  200 trial/model    │
│  Ridge Meta-Learner │
│  OOF Predictions    │
└────────┬────────────┘
         │
         ▼
   submission_advanced.csv
```

---

## ⚙️ Kullanılan Teknikler

### 🧹 Veri Ön İşleme
| Teknik | Uygulama |
|--------|----------|
| **KNN Imputer** (`k=5`, distance ağırlıklı) | `meslek`, `vucut_kitle_indeksi`, `stres_skoru`, `uyku_oncesi_kafein_mg` |
| **Mode Doldurma** | `ruh_sagligi_durumu`, `kronotip` |
| **Ülke Standardizasyonu** | `Spain→Ispanya`, `South Korea→Güney Kore`, `Sweden→İsveç` (duplikat birleştirme) |

### 🔡 Encoding
| Tür | Değişkenler |
|-----|------------|
| **Binary** | `cinsiyet`, `mevsim`, `gun_tipi` |
| **Ordinal** | `kronotip` (0-1-2), `ruh_sagligi_durumu` (0-3) |
| **K-Fold Target Encoding** | `meslek`, `ulke` (veri sızıntısı önleme) |
| **One-Hot** | Temel model için `meslek`, `ulke` |

### 🔧 Feature Engineering
Gelişmiş modelde **11 yeni özellik** türetilmiştir:

```python
rem_stres_interaction      = rem_yuzdesi × (10 - stres_skoru)
uyku_kalite_skoru          = rem_yuzdesi + derin_uyku_yuzdesi - uyanma_sayisi×2
stres_skoru_sq             = stres_skoru²
rem_derin_toplam           = rem_yuzdesi + derin_uyku_yuzdesi
stres_calisma_interaction  = stres_skoru × gunluk_calisma_saati
stres_nabiz_interaction    = stres_skoru × dinlenik_nabiz_bpm
stres_sicaklik_interaction = stres_skoru × oda_sicakligi_celsius
stres_bmi_interaction      = stres_skoru × vucut_kitle_indeksi
rem_sq                     = rem_yuzdesi²
uyku_hijyen_skoru          = kafein_mg + ekran_suresi_dk
calisma_sq                 = gunluk_calisma_saati²
```

### 🤖 Modeller
| Model | Açıklama |
|-------|----------|
| **CatBoost** | Gradient boosting (en iyi tek model) |
| **LightGBM** | Hızlı gradient boosting |
| **XGBoost** | Extreme gradient boosting |
| **ExtraTrees** | Stacking base model olarak |
| **Ridge** | Meta-learner (stacking) |
| Linear, Lasso, ElasticNet, KNN, RF, GBM, SVR, AdaBoost | Karşılaştırma amaçlı |

---

## 📊 Sonuçlar

### Model Karşılaştırması (5-Fold CV)

| Yöntem | OOF RMSE |
|--------|----------|
| Baseline (CatBoost default) | 1.2213 |
| Optuna — CatBoost (100 trial) | 1.2201 |
| Optuna — LightGBM (100 trial) | 1.2238 |
| Optuna — XGBoost (100 trial) | 1.2241 |
| **Stacking (200 trial/model)** | **1.2191** ✅ |

### En İyi Parametreler (Optuna — CatBoost)

```json
{
  "iterations": 1490,
  "depth": 5,
  "learning_rate": 0.02284,
  "l2_leaf_reg": 5.062,
  "bagging_temperature": 0.4597,
  "border_count": 176,
  "min_data_in_leaf": 49
}
```

---

## 📈 Görsel Analizler

| # | Grafik | Açıklama |
|---|--------|----------|
| 01 | Eksik Değer Dağılımı | Sütun bazında eksik sayı ve yüzde |
| 02 | Eksik Değer Deseni | 1000 örnek üzerinde heatmap |
| 03 | Aykırı Değer Oranları | IQR yöntemi ile aykırı değer barları |
| 04 | Boxplotlar | Tüm sayısal değişkenlerin dağılımı |
| 05 | Histogramlar | KDE ile histogram |
| 06 | Model RMSE Karşılaştırması | 11 model barplot |
| 07 | Performans Heatmap | RMSE, MAE, R² normalize heatmap |
| 08 | Feature Importance | Top-20 özellik |
| 09 | Tahmin Dağılımı | Train gerçek vs test tahmin |
| 10 | Optuna Karşılaştırma | Default vs Optuna RMSE |
| 11 | Optuna Geçmişi | 100 trial boyunca RMSE eğrisi |
| 12 | Parametre Önemleri | En etkili hiperparametreler |
| 13 | Optuna Tahmin Dağılımı | Optimize model tahmin dağılımı |
| 14 | Gelişmiş Sonuçlar | Stacking OOF scatter + karşılaştırma |

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn
```

### Adım Adım Çalıştırma

```bash
# 1. Keşifçi veri analizi
python eksik_aykiri_analiz.py

# 2. Eksik değer doldurma
python eksik_deger_doldurma.py

# 3. Encoding
python encoding_islemi.py

# 4. Model karşılaştırması (11 model)
python model_egitim.py

# 5. Optuna optimizasyonu (100 trial/model)
python optuna_optimizasyon.py

# 6. Gelişmiş stacking ensemble (200 trial/model)
python gelismis_model.py
```

> ⚠️ `train.csv` ve `test_x.csv` dosyalarının proje kök dizininde bulunması gerekmektedir.

---

## 🔬 Veri Seti

| Özellik | Değer |
|---------|-------|
| Görev | Regresyon |
| Hedef | `bilissel_performans_skoru` (0–10) |
| Özellik sayısı | ~20 ham + 11 türetilmiş |
| Kategorik | `cinsiyet`, `kronotip`, `ruh_sagligi_durumu`, `mevsim`, `gun_tipi`, `meslek`, `ulke` |
| Sayısal | `yas`, `rem_yuzdesi`, `derin_uyku_yuzdesi`, `stres_skoru`, `vucut_kitle_indeksi`, vb. |

---

## 🛠️ Teknolojiler

<div align="center">

| Kütüphane | Kullanım |
|-----------|----------|
| `pandas` / `numpy` | Veri işleme |
| `scikit-learn` | KNN Imputer, encoding, cross-validation, meta-learner |
| `catboost` | Ana model |
| `lightgbm` | Ensemble bileşeni |
| `xgboost` | Ensemble bileşeni |
| `optuna` | Bayesian hiperparametre optimizasyonu |
| `matplotlib` / `seaborn` | Görselleştirme |

</div>

---

<div align="center">
  <sub>Datathon projesi — Osman Karabağ</sub>
</div>
