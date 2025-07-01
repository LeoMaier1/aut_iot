#!/usr/bin/env python3
"""
reg_classification.py

Führt eine Klassifikation defekter Flaschen auf Basis von Drop-Vibrationsdaten durch.
Extrahiert Zeit- und Frequenzfeatures, trainiert verschiedene Modelle, und gibt F1-Scores aus.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


def extract_features(signal):
    """
    Extrahiert statistische und FFT-basierte Features aus einem 1D-Array.
    """
    arr = np.array(signal)
    feats = {
        'mean': arr.mean(),
        'std': arr.std(),
        'max': arr.max(),
        'min': arr.min(),
        'skew': skew(arr),
        'kurtosis': kurtosis(arr)
    }
    # 5 höchste Peaks im FFT-Spektrum
    fft_vals = np.abs(np.fft.fft(arr))
    peaks = np.sort(fft_vals)[-5:]
    for i, val in enumerate(peaks, 1):
        feats[f'fft_peak_{i}'] = val
    return feats


def main():
    # 1. Rohdaten laden
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'vibration.csv')
    df = pd.read_csv(csv_path)
    # Falls es statt 'status' bereits eine binäre Spalte gibt, passe hier an
    if 'status' in df.columns:
        df['defective'] = (df['status'] == 'defekt').astype(int)
    elif 'defective' not in df.columns:
        raise ValueError("Keine Zielspalte 'status' oder 'defective' gefunden.")

    # 2. Feature-Engineering
    records = []
    for _, row in df.iterrows():
        # Annahme: 'vibration_series' ist eine stringkodierte Liste von Werten
        signal = row['vibration_series']
        if isinstance(signal, str):
            sig = np.fromstring(signal, sep=',')
        else:
            sig = np.array(signal)
        feats = extract_features(sig)
        feats['defective'] = row['defective']
        records.append(feats)
    feat_df = pd.DataFrame(records)

    # 3. Train/Test-Split
    X = feat_df.drop('defective', axis=1)
    y = feat_df['defective']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Modelle definieren
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5)
    }

    # 5. Trainieren & Evaluieren
    results = []
    print("\nModell-Evaluation (F1-Score)\n" + "="*30)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        f1_train = f1_score(y_train, y_train_pred)
        f1_test  = f1_score(y_test,  y_test_pred)

        print(f"{name}:")
        print(f"  F1-Score (Train): {f1_train:.2f}")
        print(f"  F1-Score (Test) : {f1_test:.2f}\n")

        results.append({
            'Genutzte Features': ', '.join(X.columns),
            'Modell-Typ':       name,
            'F1-Score (Training)': f"{f1_train:.2f}",
            'F1-Score (Test)':     f"{f1_test:.2f}"
        })

    # 6. Ergebnistabelle
    results_df = pd.DataFrame(results)
    print("Ergebnistabelle für Report:")
    print(results_df.to_markdown(index=False))

    # 7. Konfusionsmatrix für bestes Modell (hier Decision Tree)
    best_model = DecisionTreeClassifier(max_depth=5)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ok", "defekt"])
    disp.plot()
    plt.title("Confusion Matrix (Decision Tree, alle Features)")
    plt.show()
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_best, target_names=["ok", "defekt"]))

    # Verteilung der Zielvariable in den Features
    print("Verteilung aller Daten:")
    print(feat_df['defective'].value_counts())
    print("Train:", y_train.value_counts())
    print("Test :", y_test.value_counts())


if __name__ == '__main__':
    main()
