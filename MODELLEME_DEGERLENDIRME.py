# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 23:09:44 2025

@author: TCSEKOC
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:00:00 2025

@author: TCSEKOC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Veri setini yükleyelim (önceki aşamalardan dışa aktarılmış final veri)
data = pd.read_excel("C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/Phyton/Dry_Bean_Final_Preprocessed.xlsx")

# Veri setinin bir kopyasını alalım
df = data.copy()

# Tüm aşamaların raporlarını birleştirmek için bir dosya açalım
with open('dimension_reduction_report.txt', 'w', encoding='utf-8') as f:
    f.write("Dry Bean Dataset - Özellik Seçimi ve Boyut İndirgeme Raporu\n")
    f.write("========================================================\n\n")

# 1. Ham Veri (Sadece Preprocessing Yapılmış)
# Bu veri seti zaten önceki aşamalarda işlenmiş (eksik veri doldurma, aykırı değer işleme, ölçekleme, kategorik kodlama)
# Ham veriyi kontrol edelim ve dışa aktaralım
with open('dimension_reduction_report.txt', 'a', encoding='utf-8') as f:
    f.write("**1. Ham Veri (Sadece Preprocessing Yapılmış)**\n")
    f.write("=====================================\n")
    f.write("Bu veri seti, önceki aşamalarda şu işlemlerden geçirilmiştir:\n")
    f.write("1. Eksik veriler dolduruldu (MajorAxisLength ve MinorAxisLength: medyan, Extent: KNN Imputation).\n")
    f.write("2. Aykırı değerler düzeltildi (Winsorization ile).\n")
    f.write("3. Sayısal sütunlar ölçeklendirildi (StandardScaler ile).\n")
    f.write("4. Class sütunu numerik hale getirildi (LabelEncoder ile).\n")
    f.write("Ham veri setinin ilk 5 satırı:\n")
    f.write(df.head().to_string() + "\n\n")

# Ham veriyi dışa aktar
output_path_raw = "C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/Phyton/Dry_Bean_Raw_Preprocessed.xlsx"
df.to_excel(output_path_raw, index=False)

with open('dimension_reduction_report.txt', 'a', encoding='utf-8') as f:
    f.write(f"Ham veri seti, '{output_path_raw}' dosyasına Excel formatında dışa aktarılmıştır.\n\n")

# 2. PCA ile Boyut İndirgeme
# Özellikler ve hedef değişkeni ayıralım
X = df.drop('Class', axis=1)
y = df['Class']

# PCA uygulayalım
# Önce tüm bileşenleri hesaplayalım ve açıklanan varyans oranlarını inceleyelim
pca = PCA()
X_pca = pca.fit_transform(X)

# Açıklanan varyans oranlarını ve kümülatif varyans oranlarını hesaplayalım
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Açıklanan varyans oranlarının ortalamasını hesaplayalım
mean_explained_variance = np.mean(explained_variance_ratio)

# Number of components: Açıklanan varyans oranı ortalamadan büyük olan bileşenleri seçelim
n_components = np.sum(explained_variance_ratio > mean_explained_variance)

# PCA’yı seçilen bileşen sayısıyla tekrar uygulayalım
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# PCA sonrası veri setini oluşturalım
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['Class'] = y

# Açıklanan varyans oranlarını tablo olarak gösterelim
variance_table = pd.DataFrame({
    'Bileşen': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
    'Açıklanan Varyans Oranı': explained_variance_ratio,
    'Kümülatif Varyans Oranı': cumulative_variance_ratio
}).to_string(index=False)

# Açıklanan varyans oranlarını grafikle gösterelim
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', label='Kümülatif Varyans Oranı')
plt.axhline(y=cumulative_variance_ratio[n_components-1], color='r', linestyle='--', label=f'Seçilen Bileşen Sayısı (n={n_components})')
plt.axvline(x=n_components, color='r', linestyle='--')
plt.title('PCA - Açıklanan Varyans Oranları')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Kümülatif Varyans Oranı')
plt.legend()
plt.grid(True)
plt.savefig('pca_variance_ratio.png')
plt.close()

# En iyi iki öznitelik için discrimination power’ı 2 boyutlu grafikle gösterelim
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Class'], palette='Set1', s=100)
plt.title('PCA - En İyi İki Bileşen ile Sınıf Ayrımı (PC1 ve PC2)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Class')
plt.savefig('pca_scatter.png')
plt.close()

# PCA öncesi ve sonrası tablo (örnek olarak birkaç sütun)
pca_comparison = pd.DataFrame({
    'Öncesi (Area)': X['Area'].head(),
    'Öncesi (Perimeter)': X['Perimeter'].head(),
    'Sonrası (PC1)': df_pca['PC1'].head(),
    'Sonrası (PC2)': df_pca['PC2'].head()
}).to_string()

# PCA sonrası veri setini dışa aktar
output_path_pca = "C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/Phyton/Dry_Bean_PCA_Transformed.xlsx"
df_pca.to_excel(output_path_pca, index=False)

# PCA Raporu
with open('dimension_reduction_report.txt', 'a', encoding='utf-8') as f:
    f.write("**2. PCA ile Boyut İndirgeme**\n")
    f.write("================================\n")
    f.write("**Yöntem:** PCA (Principal Component Analysis)\n")
    f.write(f"**Seçilen Bileşen Sayısı:** {n_components}\n")
    f.write("Seçim kriteri: Açıklanan varyans oranı, tüm bileşenlerin açıklanan varyans oranlarının ortalamasından büyük olanlar seçildi.\n")
    f.write(f"Ortalama açıklanan varyans oranı: {mean_explained_variance:.4f}\n\n")
    
    f.write("**Açıklanan Varyans Oranları:**\n")
    f.write(variance_table + "\n\n")
    
    f.write("**Görselleştirmeler:**\n")
    f.write("1. Açıklanan varyans oranları: 'pca_variance_ratio.png'\n")
    f.write("2. En iyi iki bileşen ile sınıf ayrımı: 'pca_scatter.png'\n\n")
    
    f.write("**Öncesi ve Sonrası Karşılaştırma Tablosu (Örnek):**\n")
    f.write(pca_comparison + "\n\n")
    
    f.write(f"PCA ile dönüştürülmüş veri seti, '{output_path_pca}' dosyasına Excel formatında dışa aktarılmıştır.\n\n")

# 3. LDA ile Boyut İndirgeme
# LDA uygulayalım (n_components = 3)
n_components_lda = 3
lda = LDA(n_components=n_components_lda)
X_lda = lda.fit_transform(X, y)

# LDA sonrası veri setini oluşturalım
df_lda = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_components_lda)])
df_lda['Class'] = y

# Sınıflar arası ayrımı iki boyutlu grafikle gösterelim (LD1 ve LD2)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_lda['LD1'], y=df_lda['LD2'], hue=df_lda['Class'], palette='Set1', s=100)
plt.title('LDA - Sınıflar Arası Ayrım (LD1 ve LD2)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(title='Class')
plt.savefig('lda_scatter.png')
plt.close()

# LDA öncesi ve sonrası tablo (örnek olarak birkaç sütun)
lda_comparison = pd.DataFrame({
    'Öncesi (Area)': X['Area'].head(),
    'Öncesi (Perimeter)': X['Perimeter'].head(),
    'Sonrası (LD1)': df_lda['LD1'].head(),
    'Sonrası (LD2)': df_lda['LD2'].head()
}).to_string()

# LDA sonrası veri setini dışa aktar
output_path_lda = "C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/Phyton/Dry_Bean_LDA_Transformed.xlsx"
df_lda.to_excel(output_path_lda, index=False)

# LDA Raporu
with open('dimension_reduction_report.txt', 'a', encoding='utf-8') as f:
    f.write("**3. LDA ile Boyut İndirgeme**\n")
    f.write("================================\n")
    f.write("**Yöntem:** LDA (Linear Discriminant Analysis)\n")
    f.write(f"**Seçilen Bileşen Sayısı:** {n_components_lda}\n")
    f.write("Seçim kriteri: Kullanıcı tarafından belirlenmiştir (n_components = 3).\n\n")
    
    f.write("**Görselleştirme:**\n")
    f.write("Sınıflar arası ayrım (LD1 ve LD2): 'lda_scatter.png'\n\n")
    
    f.write("**Öncesi ve Sonrası Karşılaştırma Tablosu (Örnek):**\n")
    f.write(lda_comparison + "\n\n")
    
    f.write(f"LDA ile dönüştürülmüş veri seti, '{output_path_lda}' dosyasına Excel formatında dışa aktarılmıştır.\n")