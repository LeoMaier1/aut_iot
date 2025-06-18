# Kapitel: Regressionsmodell für Endgewicht

## Aufgabe 12.3: Regressionsmodell für Endgewicht (20%)

### Zielsetzung
Entwicklung eines linearen Regressionsmodells zur Vorhersage des Endgewichts (`final_weight`) von Flaschen basierend auf IoT-Sensordaten aus der Produktionslinie.

### Datengrundlage
- **Trainingsdaten**: `database/data.csv` mit Sensormessungen von 367 Flaschen
- **Vorhersagedaten**: `X.csv` mit 274 neuen Flaschen für Prognosen
- **Zielvariable**: `final_weight` (Endgewicht der Flaschen in Gramm)

### Verwendete Features
Das Modell nutzt alle verfügbaren Sensordaten als Eingabefeatures:

| Feature | Beschreibung | Sensor-Typ |
|---------|--------------|------------|
| `vibration_index_red` | Vibrationsmessung Rot | Vibrationssensor |
| `fill_level_grams_red` | Füllstand Rot | Füllstandssensor |
| `vibration_index_blue` | Vibrationsmessung Blau | Vibrationssensor |
| `fill_level_grams_blue` | Füllstand Blau | Füllstandssensor |
| `vibration_index_green` | Vibrationsmessung Grün | Vibrationssensor |
| `fill_level_grams_green` | Füllstand Grün | Füllstandssensor |
| `temperature_green` | Temperatur Grün | Temperatursensor |
| `temperature_red` | Temperatur Rot | Temperatursensor |
| `temperature_blue` | Temperatur Blau | Temperatursensor |

### Modell-Implementierung
```python
# Einfache lineare Regression mit scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Features und Zielvariable
y = df['final_weight']
X = df.drop(['bottle', 'final_weight'], axis=1)

# Train-Test Split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)
```

### Ergebnistabelle
| Genutzte Spalten | Modell-Typ | MSE-Wert (Training) | MSE-Wert (Test) |
|------------------|------------|---------------------|-----------------|
| All features | Linear | [Wird bei Ausführung generiert] | [Wird bei Ausführung generiert] |

*Hinweis: Die exakten MSE-Werte werden bei der Ausführung des Modells in `reg_model.py` ausgegeben.*

### Modellformel
Das lineare Regressionsmodell hat die allgemeine Form:

```
final_weight = β₀ + β₁×vibration_index_red + β₂×fill_level_grams_red + 
               β₃×vibration_index_blue + β₄×fill_level_grams_blue + 
               β₅×vibration_index_green + β₆×fill_level_grams_green + 
               β₇×temperature_green + β₈×temperature_red + β₉×temperature_blue
```

Die konkreten Koeffizienten (β-Werte) werden bei der Modellausführung berechnet und ausgegeben.

### Vorhersageergebnisse
- **Anzahl Vorhersagen**: 274 Flaschen aus X.csv
- **Ausgabedatei**: `reg_student1-student2-student3.csv`
- **Format**: Zwei Spalten - `Flaschen_ID` und `y_hat`

### Modellcharakteristiken
- **Algorithmus**: Lineare Regression (Ordinary Least Squares)
- **Features**: 9 Sensormessungen pro Flasche
- **Datensplit**: 70% Training, 30% Test
- **Preprocessing**: Fehlende Werte durch Mittelwert ersetzt
- **Evaluation**: Mean Squared Error (MSE)

### Implementierungsdetails
```python
# Vorhersagen für neue Daten
X_pred = X_pred_df.drop(['bottle'], axis=1)
X_pred = X_pred.fillna(X_pred.mean())
final_predictions = model.predict(X_pred)

# Ergebnis speichern
predictions_df = pd.DataFrame({
    'Flaschen_ID': X_pred_df['bottle'],
    'y_hat': final_predictions
})
```

### Dateien der Implementierung
- **Quellcode**: `linear_reg/reg_model.py`
- **Vorhersagen**: `linear_reg/reg_student1-student2-student3.csv`
- **Dokumentation**: `docs/regression_dokumentation.md`
- **Jupyter Notebook**: `docs/regression_analysis.ipynb`

### Fazit
Das entwickelte lineare Regressionsmodell nutzt alle verfügbaren IoT-Sensordaten zur präzisen Vorhersage des Endgewichts von Flaschen. Die Implementierung ist einfach, transparent und gut reproduzierbar. Das Modell liefert quantitative Vorhersagen für die Produktionsqualitätskontrolle.

### Anweisungen zur Nutzung
1. Führen Sie `python reg_model.py` aus
2. Benennen Sie die Ausgabedatei mit Ihren Matrikelnummern um
3. Dokumentieren Sie die MSE-Werte aus der Ausgabe in Ihrem Report
4. Die Modellformel mit konkreten Koeffizienten finden Sie in der Programmausgabe
