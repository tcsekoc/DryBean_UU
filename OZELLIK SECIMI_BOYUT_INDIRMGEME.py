# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:00:00 2025

@author: TCSEKOC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Veri setlerinin dosya yolları
data_path = "C:/Users/tcsekoc/Desktop/UU/ML/Ödev/Vize/Phyton/"
datasets = {
    'Raw': f"{data_path}Dry_Bean_Raw_Preprocessed.xlsx",
    'PCA': f"{data_path}Dry_Bean_PCA_Transformed.xlsx",
    'LDA': f"{data_path}Dry_Bean_LDA_Transformed.xlsx"
}

# Sınıf etiketleri (LabelEncoder ile kodlanmış)
class_names = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
n_classes = len(class_names)

# Performans metriklerini saklamak için sözlük
results = {
    'Raw': {},
    'PCA': {},
    'LDA': {}
}

# Tüm sonuçları birleştirmek için rapor dosyası
with open('classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("Dry Bean Dataset - Sınıflandırma Raporu\n")
    f.write("=====================================\n\n")

# Her bir veri temsili için işlemleri ayrı bloklar halinde gerçekleştir
for data_type in datasets.keys():
    # --------------------------------------
    # Veri Setini Yükleme ve İlk Analiz
    # --------------------------------------
    df = pd.read_excel(datasets[data_type])
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Veri setine özgü rapor bölümü
    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(f"**{data_type} Veri Temsili**\n")
        f.write("=====================\n")
        f.write(f"Veri seti: {datasets[data_type]}\n")
        f.write(f"Özellik sayısı: {X.shape[1]}\n")
        f.write(f"Örnek sayısı: {X.shape[0]}\n\n")

        # Veri ölçeklendirme kontrolü (ortalama ve standart sapma)
        f.write(f"**Veri Ölçeklendirme Durumu ({data_type})**\n")
        f.write("------------------------------------\n")
        f.write("Sayısal sütunların istatistikleri (ölçeklendirme kontrolü):\n")
        f.write(X.describe().loc[['mean', 'std']].to_string() + "\n\n")

    # Sınıf dağılımını görselleştir (Öncesi durumu)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y, hue=y, palette='Set2', legend=False)
    plt.title(f'{data_type} Veri Temsili - Sınıf Dağılımı (Öncesi)')
    plt.xlabel('Sınıf')
    plt.ylabel('Frekans')
    plt.xticks(ticks=range(n_classes), labels=class_names, rotation=45)
    plt.savefig(f'class_distribution_before_{data_type}.png')
    plt.close()

    # --------------------------------------
    # Nested Cross-Validation Yapısı
    # --------------------------------------
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Dış döngü: 5 katmanlı CV
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=123)  # İç döngü: 3 katmanlı CV

    # Dış döngü için eğitim/test seti boyutlarını sakla
    outer_fold_sizes = {'Fold': [], 'Eğitim Seti Boyutu': [], 'Test Seti Boyutu': []}
    fold_idx = 0

    # Sınıflandırıcılar ve hiperparametre aralıkları
    classifiers = {
        'Logistic Regression': {
            'model': LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000),  # max_iter artırıldı
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}  # Naive Bayes için hiperparametre ayarı yok
        }
    }

    # Her sınıflandırıcı için sonuçları sakla
    for clf_name in classifiers.keys():
        results[data_type][clf_name] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'best_params': [],
            'best_roc': None,
            'best_roc_auc': None
        }

    # Dış döngü için sınıf dağılımlarını sakla (sonrası durumu için)
    train_distributions = []
    test_distributions = []

    # Dış döngü
    for train_idx, test_idx in outer_cv.split(X):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Eğitim ve test seti boyutlarını kaydet
        outer_fold_sizes['Fold'].append(fold_idx)
        outer_fold_sizes['Eğitim Seti Boyutu'].append(len(train_idx))
        outer_fold_sizes['Test Seti Boyutu'].append(len(test_idx))

        # Eğitim ve test seti sınıf dağılımlarını sakla
        train_dist = pd.Series(y_train).value_counts().sort_index()
        test_dist = pd.Series(y_test).value_counts().sort_index()
        train_distributions.append(train_dist)
        test_distributions.append(test_dist)

        # Her sınıflandırıcı için iç döngü ve hiperparametre ayarlama
        for clf_name, clf_info in classifiers.items():
            model = clf_info['model']
            params = clf_info['params']

            # İç döngü: Hiperparametre ayarlama
            if params:  # Eğer hiperparametre ayarı varsa
                grid_search = GridSearchCV(
                    model,
                    params,
                    cv=inner_cv,
                    scoring='f1_macro',
                    n_jobs=-1,
                    return_train_score=True
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:  # Hiperparametre ayarı yoksa (Naive Bayes)
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}

            # Test seti üzerinde tahmin yap
            y_pred = best_model.predict(X_test)

            # Performans metriklerini hesapla
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            # Sonuçları kaydet
            results[data_type][clf_name]['accuracy'].append(accuracy)
            results[data_type][clf_name]['precision'].append(precision)
            results[data_type][clf_name]['recall'].append(recall)
            results[data_type][clf_name]['f1'].append(f1)
            results[data_type][clf_name]['best_params'].append(best_params)

            # ROC eğrisi ve AUC skoru (One-vs-All)
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            if hasattr(best_model, "predict_proba"):
                y_score = best_model.predict_proba(X_test)
            else:  # Naive Bayes gibi modeller için decision_function yoksa
                y_score = best_model.predict(X_test)
                y_score = label_binarize(y_score, classes=range(n_classes))

            # ROC eğrisi ve AUC skoru hesapla
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # En iyi ROC eğrisini sakla (F1 skoru baz alınarak)
            current_f1 = f1
            if (results[data_type][clf_name]['best_roc'] is None or
                current_f1 > results[data_type][clf_name]['best_roc']['f1']):
                results[data_type][clf_name]['best_roc'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': roc_auc,
                    'f1': current_f1
                }
                results[data_type][clf_name]['best_roc_auc'] = roc_auc

    # --------------------------------------
    # Dış Döngü Analizi: Eğitim/Test Seti Boyutları
    # --------------------------------------
    outer_fold_table = pd.DataFrame(outer_fold_sizes).to_string(index=False)
    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(f"**{data_type} Veri Temsili - Dış Döngü: Eğitim/Test Seti Boyutları**\n")
        f.write("--------------------------------------------------\n")
        f.write(outer_fold_table + "\n\n")

    # Eğitim/test seti boyutlarını görselleştir
    plt.figure(figsize=(10, 6))
    plt.plot(outer_fold_sizes['Fold'], outer_fold_sizes['Eğitim Seti Boyutu'], marker='o', label='Eğitim Seti')
    plt.plot(outer_fold_sizes['Fold'], outer_fold_sizes['Test Seti Boyutu'], marker='o', label='Test Seti')
    plt.title(f'{data_type} Veri Temsili - Dış Döngü: Eğitim/Test Seti Boyutları')
    plt.xlabel('Dış Döngü Fold')
    plt.ylabel('Örnek Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outer_fold_sizes_{data_type}.png')
    plt.close()

    # --------------------------------------
    # Eğitim/Test Seti Sınıf Dağılımları (Sonrası Durumu)
    # --------------------------------------
    # Eğitim ve test seti sınıf dağılımlarını görselleştir
    for fold in range(5):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        train_dist = train_distributions[fold]
        sns.barplot(x=train_dist.index, y=train_dist.values, hue=train_dist.index, palette='Set2', legend=False)
        plt.title(f'{data_type} - Fold {fold+1}: Eğitim Seti Sınıf Dağılımı')
        plt.xlabel('Sınıf')
        plt.ylabel('Frekans')
        plt.xticks(ticks=range(n_classes), labels=class_names, rotation=45)

        plt.subplot(1, 2, 2)
        test_dist = test_distributions[fold]
        sns.barplot(x=test_dist.index, y=test_dist.values, hue=test_dist.index, palette='Set2', legend=False)
        plt.title(f'{data_type} - Fold {fold+1}: Test Seti Sınıf Dağılımı')
        plt.xlabel('Sınıf')
        plt.ylabel('Frekans')
        plt.xticks(ticks=range(n_classes), labels=class_names, rotation=45)

        plt.tight_layout()
        plt.savefig(f'class_distribution_fold_{fold+1}_{data_type}.png')
        plt.close()

    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(f"**{data_type} Veri Temsili - Eğitim/Test Seti Sınıf Dağılımları**\n")
        f.write("--------------------------------------------------\n")
        for fold in range(5):
            f.write(f"Fold {fold+1} Eğitim Seti Dağılımı:\n")
            f.write(train_distributions[fold].to_string() + "\n")
            f.write(f"Fold {fold+1} Test Seti Dağılımı:\n")
            f.write(test_distributions[fold].to_string() + "\n\n")
            f.write(f"Görselleştirme: 'class_distribution_fold_{fold+1}_{data_type}.png'\n\n")

    # --------------------------------------
    # Performans Metrikleri Tablosu
    # --------------------------------------
    metrics_table = {
        'Sınıflandırıcı': [],
        'Accuracy (Ort ± Std)': [],
        'Precision (Ort ± Std)': [],
        'Recall (Ort ± Std)': [],
        'F1 Score (Ort ± Std)': []
    }

    for clf_name in classifiers.keys():
        metrics = results[data_type][clf_name]
        metrics_table['Sınıflandırıcı'].append(clf_name)
        metrics_table['Accuracy (Ort ± Std)'].append(
            f"{np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}"
        )
        metrics_table['Precision (Ort ± Std)'].append(
            f"{np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}"
        )
        metrics_table['Recall (Ort ± Std)'].append(
            f"{np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}"
        )
        metrics_table['F1 Score (Ort ± Std)'].append(
            f"{np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}"
        )

    metrics_df = pd.DataFrame(metrics_table)
    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(f"**{data_type} Veri Temsili - Performans Metrikleri Tablosu**\n")
        f.write("--------------------------------------------------\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")

    # Performans metriklerini görselleştir (Boxplot)
    plt.figure(figsize=(12, 8))
    metrics_data = []
    labels = []
    for clf_name in classifiers.keys():
        metrics = results[data_type][clf_name]
        metrics_data.extend(metrics['accuracy'])
        metrics_data.extend(metrics['precision'])
        metrics_data.extend(metrics['recall'])
        metrics_data.extend(metrics['f1'])
        labels.extend([f'{clf_name}\nAccuracy'] * 5)
        labels.extend([f'{clf_name}\nPrecision'] * 5)
        labels.extend([f'{clf_name}\nRecall'] * 5)
        labels.extend([f'{clf_name}\nF1 Score'] * 5)

    metrics_df_plot = pd.DataFrame({'Değer': metrics_data, 'Metrik': labels})
    sns.boxplot(x='Metrik', y='Değer', hue='Metrik', data=metrics_df_plot, palette='Set3')
    plt.title(f'{data_type} Veri Temsili - Performans Metrikleri Dağılımı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'performance_metrics_boxplot_{data_type}.png')
    plt.close()

    # --------------------------------------
    # Hiperparametre Seçim Sonuçları
    # --------------------------------------
    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(f"**{data_type} Veri Temsili - İç Döngü: Hiperparametre Seçim Sonuçları**\n")
        f.write("--------------------------------------------------\n")
        for clf_name in classifiers.keys():
            f.write(f"**{clf_name}**\n")
            f.write("En İyi Hiperparametreler:\n")
            for fold, params in enumerate(results[data_type][clf_name]['best_params'], 1):
                f.write(f"Fold {fold}: {params}\n")
            f.write("\n")

    # --------------------------------------
    # ROC Eğrileri ve AUC Skorları
    # --------------------------------------
    roc_auc_table = {'Sınıflandırıcı': [], 'Ortalama ROC-AUC': []}
    for i, class_name in enumerate(class_names):
        roc_auc_table[class_name] = []

    for clf_name in classifiers.keys():
        plt.figure(figsize=(10, 8))
        best_roc = results[data_type][clf_name]['best_roc']
        roc_auc = best_roc['roc_auc']
        for i in range(n_classes):
            plt.plot(best_roc['fpr'][i], best_roc['tpr'][i],
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{data_type} - {clf_name} ROC Eğrileri (En İyi Dış Döngü)')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{data_type}_{clf_name.lower().replace(" ", "_")}.png')
        plt.close()

        # ROC-AUC tablosu için verileri topla
        roc_auc_table['Sınıflandırıcı'].append(clf_name)
        roc_auc_table['Ortalama ROC-AUC'].append(np.mean(list(roc_auc.values())))
        for i, class_name in enumerate(class_names):
            roc_auc_table[class_name].append(roc_auc[i])

    # ROC-AUC tablosunu oluştur
    roc_auc_df = pd.DataFrame(roc_auc_table)
    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(f"**{data_type} Veri Temsili - ROC-AUC Skorları Tablosu**\n")
        f.write("--------------------------------------------------\n")
        f.write(roc_auc_df.to_string(index=False) + "\n\n")

        f.write(f"**{data_type} Veri Temsili - Görselleştirmeler**\n")
        f.write("--------------------------------------------------\n")
        f.write(f"1. Sınıf Dağılımı (Öncesi): 'class_distribution_before_{data_type}.png'\n")
        f.write(f"2. Dış Döngü Eğitim/Test Boyutları: 'outer_fold_sizes_{data_type}.png'\n")
        for fold in range(5):
            f.write(f"3. Fold {fold+1} Eğitim/Test Sınıf Dağılımları: 'class_distribution_fold_{fold+1}_{data_type}.png'\n")
        f.write(f"4. Performans Metrikleri Dağılımı: 'performance_metrics_boxplot_{data_type}.png'\n")
        for clf_name in classifiers.keys():
            f.write(f"5. {clf_name} ROC Eğrisi: 'roc_curve_{data_type}_{clf_name.lower().replace(' ', '_')}.png'\n")
        f.write("\n")