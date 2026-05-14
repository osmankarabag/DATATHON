import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analiz_ciktilari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))

print("=" * 70)
print("VERİ SETİ GENEL BİLGİ")
print("=" * 70)
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")
print(f"\nSütunlar ve Tipler:")
print(df.dtypes.to_string())

# ============================================================
# 1. EKSİK DEĞER ANALİZİ
# ============================================================
print("\n" + "=" * 70)
print("1. EKSİK DEĞER ANALİZİ")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Eksik Sayı': missing,
    'Eksik Yüzde (%)': missing_pct,
    'Veri Tipi': df.dtypes
}).sort_values('Eksik Sayı', ascending=False)

print("\nEksik Değer Tablosu:")
print(missing_df.to_string())
print(f"\nToplam eksik hücre: {df.isnull().sum().sum()}")
print(f"Toplam hücre: {df.shape[0] * df.shape[1]}")
print(f"Genel eksiklik oranı: {(df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100):.2f}%")

# Eksik değer olan sütunları filtrele
missing_cols = missing_df[missing_df['Eksik Sayı'] > 0]
if len(missing_cols) > 0:
    print(f"\nEksik değer içeren sütun sayısı: {len(missing_cols)}")
    
    # Eksik değer grafiği
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    cols_with_missing = missing_cols.index.tolist()
    vals = missing_cols['Eksik Sayı'].values
    pcts = missing_cols['Eksik Yüzde (%)'].values
    
    colors = sns.color_palette("Reds_r", len(cols_with_missing))
    bars = axes[0].barh(cols_with_missing, vals, color=colors)
    axes[0].set_title("Eksik Değer Sayıları", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Eksik Sayı")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{int(v)}', va='center', fontsize=10)
    
    bars2 = axes[1].barh(cols_with_missing, pcts, color=colors)
    axes[1].set_title("Eksik Değer Yüzdeleri (%)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Yüzde (%)")
    for bar, v in zip(bars2, pcts):
        axes[1].text(bar.get_width() + max(pcts)*0.01, bar.get_y() + bar.get_height()/2,
                     f'{v:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eksik_deger_dagilimi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap - eksik değer pattern
    if len(cols_with_missing) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        sample_size = min(1000, len(df))
        sample_idx = np.random.RandomState(42).choice(len(df), sample_size, replace=False)
        sns.heatmap(df.iloc[sample_idx][cols_with_missing].isnull().astype(int), 
                    cbar_kws={'label': 'Eksik (1) / Mevcut (0)'}, 
                    cmap='YlOrRd', ax=ax, yticklabels=False)
        ax.set_title("Eksik Değer Deseni (1000 Örnek)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "02_eksik_deger_deseni.png"), dpi=150, bbox_inches='tight')
        plt.close()
else:
    print("\n✅ Veri setinde HİÇ eksik değer bulunmamaktadır!")

# Semantik eksik değer kontrolü
print("\n" + "-" * 50)
print("Semantik/Örtük Eksik Değer Kontrolü")
print("-" * 50)
suspect_values = ['?', 'NA', 'N/A', 'null', 'None', '-', '', ' ', 'Bilinmiyor', 'Belirtilmemiş']
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    unique_vals = df[col].unique()
    found = [v for v in suspect_values if v in unique_vals]
    if found:
        print(f"  ⚠ '{col}': Şüpheli değerler bulundu -> {found}")
    print(f"  '{col}' benzersiz değerler ({df[col].nunique()}): {list(unique_vals[:10])}")

# ============================================================
# 2. AYKIRI DEĞER ANALİZİ (IQR Yöntemi)
# ============================================================
print("\n" + "=" * 70)
print("2. AYKIRI DEĞER ANALİZİ (IQR YÖNTEMİ)")
print("=" * 70)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in num_cols:
    num_cols.remove('id')

outlier_summary = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    count = len(outliers)
    pct = count / len(df) * 100
    outlier_summary.append({
        'Sütun': col, 'Q1': round(Q1, 2), 'Q3': round(Q3, 2), 'IQR': round(IQR, 2),
        'Alt Sınır': round(lower, 2), 'Üst Sınır': round(upper, 2),
        'Aykırı Sayı': count, 'Aykırı Yüzde (%)': round(pct, 2)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Aykırı Sayı', ascending=False)
print("\nAykırı Değer Özet Tablosu (IQR):")
print(outlier_df.to_string(index=False))

# Aykırı değer bar grafiği
has_outliers = outlier_df[outlier_df['Aykırı Sayı'] > 0]
if len(has_outliers) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("OrRd_r", len(has_outliers))
    bars = ax.barh(has_outliers['Sütun'], has_outliers['Aykırı Yüzde (%)'], color=colors)
    ax.set_title("Aykırı Değer Oranları (IQR Yöntemi)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Aykırı Değer Yüzdesi (%)")
    for bar, v in zip(bars, has_outliers['Aykırı Yüzde (%)'].values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_aykiri_deger_oranlari.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Boxplot'lar
n = len(num_cols)
ncols_plot = 3
nrows_plot = (n + ncols_plot - 1) // ncols_plot
fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(18, nrows_plot * 4))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, y=col, ax=axes[i], color='#4ECDC4', flierprops=dict(marker='o', markersize=3, alpha=0.3))
    axes[i].set_title(col, fontsize=11, fontweight='bold')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Sayısal Değişkenlerin Boxplot Grafikleri", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_boxplotlar.png"), dpi=150, bbox_inches='tight')
plt.close()

# Histogram + KDE
fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(18, nrows_plot * 4))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='#3498db', alpha=0.7, bins=50)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Ort: {df[col].mean():.2f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='--', label=f'Med: {df[col].median():.2f}')
    axes[i].legend(fontsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Sayısal Değişken Dağılımları (Histogram + KDE)", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_histogramlar.png"), dpi=150, bbox_inches='tight')
plt.close()

# Temel istatistikler
print("\n" + "=" * 70)
print("SAYISAL DEĞİŞKEN İSTATİSTİKLERİ")
print("=" * 70)
desc = df[num_cols].describe().T
desc['skewness'] = df[num_cols].skew()
desc['kurtosis'] = df[num_cols].kurtosis()
print(desc.round(3).to_string())

print(f"\n✅ Tüm grafikler '{OUTPUT_DIR}' klasörüne kaydedildi.")
print("Üretilen dosyalar:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")
