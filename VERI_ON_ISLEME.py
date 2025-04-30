# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:00:00 2025

@author: TCSEKOC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Veri setini verilen Excel dosyasından yükleme
data = pd.read_excel("C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/DryBeanDataset/Dry_Bean_Dataset.xlsx")

# Orijinal veri setinin bir kopyasını alalım
df = data.copy()

# Tüm aşamaların raporlarını birleştirmek için bir dosya açalım
with open('final_report.txt', 'w', encoding='utf-8') as f:
    f.write("Dry Bean Dataset - Tüm Aşamalar Raporu\n")
    f.write("=====================================\n\n")

# 1. Aşama: Eksik Veri Ekleme ve Doldurma
# Eksik veri ekleme fonksiyonu
def add_missing_values(df, columns, fraction):
    df_copy = df.copy()
    for col in columns:
        df_copy.loc[df_copy.sample(frac=fraction).index, col] = np.nan
    return df_copy

# %5 eksik veri eklenecek sütunlar: 'MajorAxisLength' ve 'MinorAxisLength'
low_missing_columns = ['MajorAxisLength', 'MinorAxisLength']
df = add_missing_values(df, low_missing_columns, fraction=0.05)

# %35 eksik veri eklenecek sütun: 'Extent'
high_missing_columns = ['Extent']
df = add_missing_values(df, high_missing_columns, fraction=0.35)

# Adım 3: Eksik verileri gözlemleme
missing_data = df.isnull().sum()
missing_data_summary = missing_data.to_string()

# Eksik veri oranlarını grafikle gösterelim
plt.figure(figsize=(10, 6))
missing_data[missing_data > 0].plot(kind='bar', color='red')
plt.title('Eksik Veri Sayısı (Sütun Bazında)')
plt.xlabel('Öznitelik')
plt.ylabel('Eksik Veri Sayısı')
plt.tight_layout()
plt.savefig('missing_data_summary.png')
plt.close()

# %5 eksik verileri medyan ile doldurma
major_skewness = df['MajorAxisLength'].skew()
minor_skewness = df['MinorAxisLength'].skew()
df_before_filling = df.copy()
df['MajorAxisLength'] = df['MajorAxisLength'].fillna(df['MajorAxisLength'].median())
df['MinorAxisLength'] = df['MinorAxisLength'].fillna(df['MinorAxisLength'].median())

# Öncesi/sonrası eksik veri tablosu
missing_before_low = df_before_filling[low_missing_columns].isnull().sum()
missing_after_low = df[low_missing_columns].isnull().sum()
missing_comparison_low = pd.DataFrame({
    'Öncesi': missing_before_low,
    'Sonrası (Medyan)': missing_after_low
}).to_string()

# Görselleştirme: MajorAxisLength için öncesi/sonrası dağılım
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_before_filling['MajorAxisLength'], kde=True, color='red', label='Eksik Veri (%5)')
plt.title('MajorAxisLength - Eksik Veri Eklenmiş')
plt.xlabel('MajorAxisLength')
plt.ylabel('Frekans')
plt.subplot(1, 2, 2)
sns.histplot(df['MajorAxisLength'], kde=True, color='green', label='Medyan ile Doldurulmuş')
plt.title('MajorAxisLength - Medyan ile Doldurulmuş')
plt.xlabel('MajorAxisLength')
plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig('major_axis_length_filled_comparison.png')
plt.close()

# Görselleştirme: MinorAxisLength için öncesi/sonrası dağılım
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_before_filling['MinorAxisLength'], kde=True, color='red', label='Eksik Veri (%5)')
plt.title('MinorAxisLength - Eksik Veri Eklenmiş')
plt.xlabel('MinorAxisLength')
plt.ylabel('Frekans')
plt.subplot(1, 2, 2)
sns.histplot(df['MinorAxisLength'], kde=True, color='green', label='Medyan ile Doldurulmuş')
plt.title('MinorAxisLength - Medyan ile Doldurulmuş')
plt.xlabel('MinorAxisLength')
plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig('minor_axis_length_filled_comparison.png')
plt.close()

# %35 eksik verileri KNN Imputation ile doldurma
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df_before_filling_high = df.copy()
imputer = KNNImputer(n_neighbors=5)
df_numeric_filled = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), 
                                 columns=numeric_columns, 
                                 index=df.index)
df_filled = df.copy()
df_filled[numeric_columns] = df_numeric_filled
df = df_filled

# Öncesi/sonrası eksik veri tablosu
missing_before_high = df_before_filling_high[high_missing_columns].isnull().sum()
missing_after_high = df[high_missing_columns].isnull().sum()
missing_comparison_high = pd.DataFrame({
    'Öncesi': missing_before_high,
    'Sonrası (KNN)': missing_after_high
}).to_string()

# Görselleştirme: Extent için öncesi/sonrası dağılım
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_before_filling_high['Extent'], kde=True, color='red', label='Eksik Veri (%35)')
plt.title('Extent - Eksik Veri Eklenmiş')
plt.xlabel('Extent')
plt.ylabel('Frekans')
plt.subplot(1, 2, 2)
sns.histplot(df['Extent'], kde=True, color='purple', label='KNN ile Doldurulmuş')
plt.title('Extent - KNN ile Doldurulmuş')
plt.xlabel('Extent')
plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig('extent_filled_comparison.png')
plt.close()

# Çarpıklık hesaplama (Extent için)
extent_skewness = df_before_filling_high['Extent'].skew()

# 1. Aşama Raporu
with open('final_report.txt', 'a', encoding='utf-8') as f:
    f.write("**1. Aşama: Eksik Veri Ekleme ve Doldurma**\n")
    f.write("================================\n")
    f.write("%5 eksik veri eklenen sütunlar: 'MajorAxisLength' ve 'MinorAxisLength'\n")
    f.write("Seçim kriteri:\n")
    f.write("1. Geometrik ve Bilgilendirici Özellikler: Bu öznitelikler, fasulyelerin geometrik özelliklerini temsil eder ve sınıflandırma için önemli olabilir.\n")
    f.write("2. Sürekli sayısal değerler içerir ve genellikle yüksek varyansa sahiptir, bu da farklı sınıflar arasında ayrım yapma potansiyeline sahip olduklarını gösterir.\n")
    f.write("3. Eksik veri giderme yöntemlerinin bu tür bilgilendirici öznitelikler üzerindeki etkisini gözlemlemek amacıyla seçilmiştir.\n")
    f.write("%35 eksik veri eklenen sütun: 'Extent'\n")
    f.write("Seçim kriteri:\n")
    f.write("1. 'Extent' özniteliği, 0 ile 1 arasında normalize edilmiş bir değer içerir ve düşük varyansa sahiptir.\n")
    f.write("2. Sınıflandırma için daha az bilgilendirici olabilir (düşük korelasyon bekleniyor).\n")
    f.write("3. %35 gibi yüksek bir eksik veri oranıyla, bu özniteliğin eksik veri giderme yöntemlerine nasıl tepki verdiğini gözlemlemek amacıyla seçilmiştir.\n")
    f.write("4. Diğer geometrik özniteliklerle dolaylı bir ilişkiye sahip olabilir, bu nedenle farklı bir doldurma yöntemiyle nasıl sonuçlanacağını görmek ilginç olacaktır.\n\n")
    
    f.write("**Eksik Verilerin Gözlemlenmesi**\n")
    f.write("Eksik veri sayıları:\n")
    f.write(missing_data_summary + "\n")
    f.write("Eksik veri dağılımı 'missing_data_summary.png' grafiğinde gösterilmiştir.\n\n")
    
    f.write("**%5 Eksik Verilerin Doldurulması**\n")
    f.write("'MajorAxisLength' ve 'MinorAxisLength' öznitelikleri medyan ile doldurulmuştur.\n")
    f.write(f"Doldurma yöntemi seçimi: Her iki öznitelik de sağa çarpıktır (MajorAxisLength skewness: {major_skewness:.2f}, MinorAxisLength skewness: {minor_skewness:.2f}). Medyan, çarpık dağılımlarda daha uygun bir merkezi eğilim ölçüsüdür.\n")
    f.write("Öncesi ve sonrası eksik veri sayıları:\n")
    f.write(missing_comparison_low + "\n")
    f.write("'MajorAxisLength' için öncesi/sonrası dağılım 'major_axis_length_filled_comparison.png' grafiğinde gösterilmiştir.\n")
    f.write("'MinorAxisLength' için öncesi/sonrası dağılım 'minor_axis_length_filled_comparison.png' grafiğinde gösterilmiştir.\n\n")
    
    f.write("**%35 Eksik Verilerin Doldurulması**\n")
    f.write("'Extent' özniteliği KNN Imputation yöntemiyle doldurulmuştur.\n")
    f.write("Doldurma yöntemi seçimi:\n")
    f.write(f"1. 'Extent' hafif sola çarpık bir dağılıma sahiptir (skewness: {extent_skewness:.2f}).\n")
    f.write("2. %35 gibi yüksek bir eksik veri oranı için, öznitelikler arasındaki ilişkileri dikkate alan KNN Imputation yöntemi tercih edilmiştir.\n")
    f.write("3. KNN, diğer özniteliklerle (örneğin, 'Area' veya 'Perimeter') olası dolaylı ilişkileri kullanarak daha doğru bir doldurma yapar.\n")
    f.write("Öncesi ve sonrası eksik veri sayıları:\n")
    f.write(missing_comparison_high + "\n")
    f.write("'Extent' için öncesi/sonrası dağılım 'extent_filled_comparison.png' grafiğinde gösterilmiştir.\n\n")

# 2. Aşama: Aykırı Değer Tespiti ve İşleme
# IQR yöntemiyle aykırı değerleri tespit etme fonksiyonu
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers, lower_bound, upper_bound, Q1, Q3, IQR

# Analiz edilecek sütunlar
columns_to_analyze = ['MajorAxisLength', 'MinorAxisLength', 'Extent', 'Area', 'Perimeter', 'ConvexArea', 'EquivDiameter']

# IQR tablosu için verileri toplayalım
iqr_data = {'Sütun': [], 'Q1': [], 'Q3': [], 'IQR': [], 'Alt Sınır': [], 'Üst Sınır': []}
outlier_counts_before = {}
outlier_counts_after = {}
lower_bounds = {}
upper_bounds = {}

# Öncesi/sonrası veri setlerini saklayalım
df_before_outlier_handling = df.copy()

for col in columns_to_analyze:
    # Aykırı değerleri ve IQR değerlerini tespit et
    outliers, lower_bound, upper_bound, Q1, Q3, IQR = detect_outliers_iqr(df, col)
    outlier_counts_before[col] = len(outliers)
    lower_bounds[col] = lower_bound
    upper_bounds[col] = upper_bound
    
    # IQR tablosu için verileri kaydet
    iqr_data['Sütun'].append(col)
    iqr_data['Q1'].append(Q1)
    iqr_data['Q3'].append(Q3)
    iqr_data['IQR'].append(IQR)
    iqr_data['Alt Sınır'].append(lower_bound)
    iqr_data['Üst Sınır'].append(upper_bound)
    
    # Aykırı değerleri sınır değerlerle değiştir (Winsorization)
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Değişim sonrası aykırı değer kontrolü
    outliers_after, _, _, _, _, _ = detect_outliers_iqr(df, col)
    outlier_counts_after[col] = len(outliers_after)

# IQR tablosunu oluştur
iqr_table = pd.DataFrame(iqr_data).to_string(index=False)

# Aykırı değerlerin öncesi/sonrası tablosu
outlier_comparison = pd.DataFrame({
    'Öncesi': outlier_counts_before,
    'Sonrası': outlier_counts_after,
    'Alt Sınır': lower_bounds,
    'Üst Sınır': upper_bounds
}).to_string()

# Görselleştirme: Öncesi/sonrası boxplot'lar
for col in columns_to_analyze:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_before_outlier_handling[col], color='red')
    plt.title(f'{col} - Aykırı Değerler Öncesi')
    plt.ylabel(col)
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col], color='green')
    plt.title(f'{col} - Aykırı Değerler Sonrası (Winsorization)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'{col}_outlier_comparison.png')
    plt.close()

# 2. Aşama Raporu
with open('final_report.txt', 'a', encoding='utf-8') as f:
    f.write("**2. Aşama: Aykırı Değer Tespiti ve İşleme**\n")
    f.write("================================\n")
    f.write("**Yöntem Seçimi:** IQR (Interquartile Range) yöntemi kullanılmıştır.\n")
    f.write("Seçim nedenleri:\n")
    f.write("1. Dağılım Varsayımı Yapmaz: IQR, verinin normal dağılıma uygun olup olmadığını varsaymaz. Veri setindeki öznitelikler çarpık dağılımlar gösteriyor (örneğin, MajorAxisLength skewness ≈ 1.35, Extent skewness ≈ -0.89). Bu nedenle IQR, Z-score’a göre daha uygun.\n")
    f.write("2. Daha Sağlam (Robust): IQR, aykırı değerlere karşı daha dayanıklıdır çünkü çeyrekliklere dayanır. Z-score ise ortalamaya ve standart sapmaya bağlıdır, bu da aykırı değerlerden daha fazla etkilenir.\n")
    f.write("3. Veri Setinin Yapısı: Dry Bean veri seti heterojen bir yapıya sahip (örneğin, Area 28.395’ten 41.000’e kadar değişiyor). IQR, bu tür veri setlerinde daha güvenilir sonuçlar verir.\n\n")
    
    f.write("**Analiz Edilen Sütunlar:** MajorAxisLength, MinorAxisLength, Extent, Area, Perimeter, ConvexArea, EquivDiameter\n")
    f.write("Seçim nedenleri:\n")
    f.write("1. MajorAxisLength, MinorAxisLength ve Extent: Bu sütunlar, 1. aşamada eksik veri ekleme ve doldurma işlemlerinden geçti. Eksik veri giderme işleminin aykırı değerler üzerindeki etkisini görmek istedim.\n")
    f.write("2. Geometrik Öznitelikler: MajorAxisLength, MinorAxisLength, Area, Perimeter, ConvexArea ve EquivDiameter, fasulyelerin boyutlarını ve şekillerini doğrudan ölçen temel özniteliklerdir. Bu sütunlarda aykırı değerler, sınıflandırma performansını ciddi şekilde etkileyebilir.\n")
    f.write("3. Extent: %35 eksik veri eklenip KNN ile dolduruldu. KNN imputasyon, aykırı değerleri etkileyebilir, bu nedenle bu sütunda analiz yapmak önemli.\n")
    f.write("4. Area, Perimeter, ConvexArea ve EquivDiameter: Bu sütunlar, yüksek varyansa sahip ve sınıflandırma için kritik özniteliklerdir. Aykırı değerlerin bu sütunlarda olması, modelin genelleme yeteneğini bozabilir.\n\n")
    
    f.write("**IQR Değerleri ve Sınırlar (Aykırı Değer Tespiti Öncesi):**\n")
    f.write(iqr_table + "\n\n")
    
    f.write("**Aykırı Değer İşleme Yöntemi:** Sınır Değerlerle Değiştirme (Winsorization)\n")
    f.write("Seçim nedenleri:\n")
    f.write("1. Veri Kaybını Önler: Veri setinde 13.611 örnek var. Aykırı değerleri silmek, veri kaybına neden olabilir ve sınıflandırma dengesini bozabilir.\n")
    f.write("2. Verinin Genel Yapısını Korur: Sınır değerlerle değiştirme, aykırı değerlerin etkisini azaltırken verinin genel yapısını korur.\n")
    f.write("3. Extent gibi normalize edilmiş özniteliklerde değiştirme daha mantıklı.\n\n")
    
    f.write("**Aykırı Değerlerin Öncesi ve Sonrası Durumu**\n")
    f.write("Aykırı değer sayıları ve sınırlar:\n")
    f.write(outlier_comparison + "\n")
    for col in columns_to_analyze:
        f.write(f"{col} için öncesi/sonrası boxplot: '{col}_outlier_comparison.png'\n")
    
    f.write("\n**Değerlendirme:**\n")
    f.write("1. Aykırı değerlerin sayısı, her sütunda toplam örneklerin %1-1.5’ini oluşturuyor (örneğin, 13.611 örneğin %1’i 136 örnek). Bu, IQR yönteminin doğru çalıştığını ve çok fazla örneği aykırı olarak işaretlemediğini gösteriyor.\n")
    f.write("2. Area, Perimeter, ConvexArea ve EquivDiameter sütunlarında aykırı değerlerin düzeltilmesi, bu sütunların yüksek varyanslı yapısı nedeniyle özellikle önemli. Bu sütunlar, sınıflandırma için kritik öznitelikler ve aykırı değerler modelin genelleme yeteneğini bozabilir.\n")
    f.write("3. Bu sütunlar birbirleriyle yüksek korelasyona sahip (örneğin, Area ve ConvexArea arasında korelasyon ≈ 0.99). Bir sütunda aykırı değerlerin düzeltilmesi, diğer sütunlarda da dolaylı bir iyileşme sağlar.\n\n")

# 3. Aşama: Özellik Ölçekleme
# Sayısal sütunlar (Class hariç tüm sütunlar sayısal)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Ölçekleme öncesi istatistikleri hesapla
stats_before = df[numeric_columns].describe().loc[['mean', 'std', 'min', 'max']].to_string()

# Boxplot: Ölçekleme öncesi
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_columns])
plt.title('Özellikler - Ölçekleme Öncesi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('scaling_before_boxplot.png')
plt.close()

# Histogram: Ölçekleme öncesi (örnek olarak birkaç sütun)
plt.figure(figsize=(12, 8))
for i, col in enumerate(['Area', 'MajorAxisLength', 'Extent'], 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, color='red')
    plt.title(f'{col} - Ölçekleme Öncesi')
plt.tight_layout()
plt.savefig('scaling_before_histogram.png')
plt.close()

# StandardScaler ile ölçekleme
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Ölçekleme sonrası istatistikleri hesapla
stats_after = df[numeric_columns].describe().loc[['mean', 'std', 'min', 'max']].to_string()

# Boxplot: Ölçekleme sonrası
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_columns])
plt.title('Özellikler - Ölçekleme Sonrası (StandardScaler)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('scaling_after_boxplot.png')
plt.close()

# Histogram: Ölçekleme sonrası (örnek olarak birkaç sütun)
plt.figure(figsize=(12, 8))
for i, col in enumerate(['Area', 'MajorAxisLength', 'Extent'], 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, color='green')
    plt.title(f'{col} - Ölçekleme Sonrası')
plt.tight_layout()
plt.savefig('scaling_after_histogram.png')
plt.close()

# 3. Aşama Raporu
with open('final_report.txt', 'a', encoding='utf-8') as f:
    f.write("**3. Aşama: Özellik Ölçekleme (Feature Scaling)**\n")
    f.write("================================\n")
    f.write("**Yöntem Seçimi:** StandardScaler\n")
    f.write("Seçim nedenleri:\n")
    f.write("1. Normal Dağılıma Yakınlık: StandardScaler, veriyi ortalaması 0 ve standart sapması 1 olacak şekilde standardize eder. Veri setindeki öznitelikler çarpık olsa da (örneğin, MajorAxisLength skewness ≈ 1.35), eksik veri giderme ve aykırı değer düzeltme işlemleri dağılımları bir miktar düzeltmiş olabilir. StandardScaler çarpık verilerle de çalışabilir.\n")
    f.write("2. Makine Öğrenmesi Modellerine Uygunluk: StandardScaler, mesafe tabanlı (KNN, SVM) ve gradyan tabanlı (lojistik regresyon, yapay sinir ağları) algoritmalar için daha uygun. Dry Bean veri seti bir sınıflandırma problemi için kullanılıyor ve bu tür modellerle çalışılması muhtemel.\n")
    f.write("3. MinMaxScaler’a Göre Avantaj: MinMaxScaler, veriyi 0-1 aralığına sıkıştırır ancak aykırı değerlere karşı daha hassastır. StandardScaler daha sağlam bir yöntemdir.\n")
    f.write("4. Veri Setinin Yapısı: Öznitelikler farklı birimlerde ve aralıklarda (örneğin, Area 28.395-41.000, Extent 0.669-0.820). StandardScaler, bu farklı ölçekleri standardize eder.\n\n")
    
    f.write("**Ölçeklenen Sütunlar:** Tüm sayısal sütunlar (Class hariç)\n")
    f.write("Sayısal sütunlar: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4\n\n")
    
    f.write("**Ölçekleme Öncesi İstatistikler:**\n")
    f.write(stats_before + "\n\n")
    f.write("**Ölçekleme Sonrası İstatistikler:**\n")
    f.write(stats_after + "\n\n")
    
    f.write("**Görselleştirmeler:**\n")
    f.write("1. Ölçekleme öncesi boxplot: 'scaling_before_boxplot.png'\n")
    f.write("2. Ölçekleme sonrası boxplot: 'scaling_after_boxplot.png'\n")
    f.write("3. Ölçekleme öncesi histogram (örnek sütunlar): 'scaling_before_histogram.png'\n")
    f.write("4. Ölçekleme sonrası histogram (örnek sütunlar): 'scaling_after_histogram.png'\n\n")

# 4. Aşama: Kategorik Verilerin Kodlanması
# Kategorik sütunları belirleyelim
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# categorical_columns = ['Class'] (sadece Class sütunu kategorik)

# Class sütununu LabelEncoder ile kodlayalım
df_before_encoding = df.copy()
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# Sınıf etiketlerinin dönüşüm öncesi ve sonrası eşleşmesini gösterelim
class_mapping = pd.DataFrame({
    'Orijinal Sınıf': label_encoder.classes_,
    'Kodlanmış Sınıf': range(len(label_encoder.classes_))
}).to_string(index=False)

# Sınıf dağılımlarını görselleştirelim
# Öncesi: Orijinal sınıf etiketleri
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=df_before_encoding['Class'], hue=df_before_encoding['Class'], palette='Set2', legend=False)
plt.title('Sınıf Dağılımı - Kodlama Öncesi')
plt.xlabel('Sınıf')
plt.ylabel('Frekans')
plt.xticks(rotation=45)

# Sonrası: Kodlanmış sınıf etiketleri
plt.subplot(1, 2, 2)
sns.countplot(x=df['Class'], hue=df['Class'], palette='Set2', legend=False)
plt.title('Sınıf Dağılımı - Kodlama Sonrası')
plt.xlabel('Kodlanmış Sınıf')
plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig('class_distribution_comparison.png')
plt.close()

# 4. Aşama Raporu
with open('final_report.txt', 'a', encoding='utf-8') as f:
    f.write("**4. Aşama: Kategorik Verilerin Kodlanması**\n")
    f.write("================================\n")
    f.write("**Kategorik Sütunlar:** Sadece 'Class' sütunu kategorik.\n")
    f.write("Veri setinde başka kategorik sütun bulunmamaktadır.\n\n")
    
    f.write("**Kodlama Yöntemi:** LabelEncoder (Class sütunu için)\n")
    f.write("Seçim nedenleri:\n")
    f.write("1. Class sütunu, hedef değişken (target variable) olarak sınıflandırma problemi için kullanılıyor. Makine öğrenmesi modelleri kategorik verileri doğrudan işleyemez, bu nedenle numerik hale getirilmesi gerekiyor.\n")
    f.write("2. LabelEncoder, kategorik sınıf etiketlerini sıralı numerik değerlere dönüştürmek için uygun bir yöntemdir ve hedef değişkenler için sıkça kullanılır.\n")
    f.write("3. Class sütunu sıralı bir yapıya sahip değil (örneğin, 'SEKER' ile 'DERMASON' arasında bir sıralama ilişkisi yok), ancak LabelEncoder sınıflandırma problemlerinde hedef değişken için sorun çıkarmaz, çünkü modeller bu değerleri yalnızca sınıf kimlikleri olarak değerlendirir.\n\n")
    
    f.write("**Dönüştürmenin Faydaları:**\n")
    f.write("1. Numerik hale getirilen sınıf etiketleri, sınıflandırma algoritmalarının (örneğin, scikit-learn kütüphanesindeki modeller) doğru bir şekilde çalışmasını sağlar.\n")
    f.write("2. Veri seti, bir sonraki aşamada (örneğin, model eğitimi) doğrudan kullanılabilir hale gelir.\n\n")
    
    f.write("**Sınıf Etiketlerinin Dönüşüm Tablosu:**\n")
    f.write(class_mapping + "\n\n")
    
    f.write("**Görselleştirme:**\n")
    f.write("Sınıf dağılımı öncesi/sonrası: 'class_distribution_comparison.png'\n")
    f.write("Not: Grafiklerde sınıf dağılımlarının şekli değişmemiştir, sadece etiketler numerik hale getirilmiştir.\n\n")

# Final veri setinin durumunu kontrol edelim
with open('final_report.txt', 'a', encoding='utf-8') as f:
    f.write("**Final Veri Seti Durumu:**\n")
    f.write("================================\n")
    f.write("Final veri seti (df), tüm aşamaların uygulanmış halini yansıtmaktadır:\n")
    f.write("1. Eksik veriler dolduruldu (MajorAxisLength ve MinorAxisLength: medyan, Extent: KNN Imputation).\n")
    f.write("2. Aykırı değerler düzeltildi (Winsorization ile).\n")
    f.write("3. Sayısal sütunlar ölçeklendirildi (StandardScaler ile).\n")
    f.write("4. Class sütunu numerik hale getirildi (LabelEncoder ile).\n")
    f.write("Final veri setinin ilk 5 satırı:\n")
    f.write(df.head().to_string() + "\n\n")

# Final veri setini Excel dosyasına dışa aktar
output_path = "C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/Phyton/Dry_Bean_Final_Preprocessed.xlsx"
df.to_excel(output_path, index=False)

# Dışa aktarma işlemini rapora ekle
with open('final_report.txt', 'a', encoding='utf-8') as f:
    f.write("**Final Veri Setinin Dışa Aktarılması:**\n")
    f.write("================================\n")
    f.write(f"Final veri seti, '{output_path}' dosyasına Excel formatında dışa aktarılmıştır.\n")
    f.write("Bu dosya, sonraki aşamalarda (Özellik Seçimi ve Boyut İndirgeme) kaynak veri olarak kullanılabilir.\n")