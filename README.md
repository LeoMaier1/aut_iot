# aut_iot - IoT Flaschenfüllstation Analyse

Dieses Projekt implementiert ein vollständiges IoT-System zur Überwachung und Analyse einer Flaschenfüllstation. Es umfasst MQTT-Datensammlung, Datenverarbeitung, Machine Learning-Modelle für Regression und Klassifikation sowie Visualisierung.

## Projektstruktur

### Hauptmodule

- **`mqtt_client/`** - MQTT-Client für Datensammlung von IoT-Sensoren
- **`database/`** - Datenverarbeitung und -speicherung
- **`linear_reg/`** - Lineare Regression für Gewichtsvorhersage
- **`Classification/`** - Klassifikation defekter Flaschen
- **`visualisierung/`** - Datenvisualisierung und Plots

### Datendateien

- **`data.csv`** - Hauptdatensatz mit Sensor- und Produktionsdaten
- **`vibration.csv`** - Vibrationsdaten für Defekterkennung
- **`X.csv`** - Testdaten für Vorhersagen
- **`config.json`** - Konfigurationsdatei für MQTT und andere Einstellungen

## Installation

### Voraussetzungen
- Python 3.8 oder höher
- pip (Python Package Manager)

### Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

Die wichtigsten Pakete:
- `paho-mqtt` - MQTT-Client
- `pandas` - Datenverarbeitung
- `numpy` - Numerische Berechnungen
- `scikit-learn` - Machine Learning
- `matplotlib` - Visualisierung
- `scipy` - Wissenschaftliche Berechnungen

## Konfiguration

Bearbeiten Sie `config.json` für Ihre MQTT-Einstellungen:
```json
{
    "mqtt": {
        "broker": "158.180.44.197",
        "port": 1883,
        "username": "bobm",
        "password": "letmein",
        "topic": "iot1/teaching_factory/#"
    }
}
```

## Ausführung

### 1. MQTT-Datensammlung starten
```bash
cd mqtt_client
python mqtt_client.py
```
Sammelt kontinuierlich Daten von der IoT-Flaschenfüllstation und speichert sie in `database/data.csv`.

### 2. Lineare Regression für Gewichtsvorhersage
```bash
cd linear_reg
python reg_model.py
```
- Trainiert ein lineares Regressionsmodell
- Sagt Endgewichte von Flaschen vorher
- Speichert Vorhersagen in `reg_52315857-52310501.csv`
- **Modell-Performance**: MSE ≈ 0 (bei allen Features), MSE ≈ 40 (bei reduzierten Features)

### 3. Defekt-Klassifikation
```bash
cd Classification
python class.py
```
- Klassifiziert defekte Flaschen basierend auf Vibrationsdaten
- Nutzt verschiedene ML-Algorithmen (Logistic Regression, k-NN, Decision Tree)
- Extrahiert Zeit- und Frequenzfeatures aus Vibrationssignalen

### 4. Datenvisualisierung
```bash
cd visualisierung
python visualisierung.py
```
Erstellt Zeitreihen-Plots der Sensordaten (z.B. Füllstand, Temperatur).

## Datenfluss

1. **Sensordaten** → MQTT Topics → `mqtt_client.py`
2. **Rohdaten** → `database/data.csv` (strukturierte Speicherung)
3. **Vibrationsdaten** → `database/vibration.csv` (für Defekterkennung)
4. **ML-Modelle** → Vorhersagen und Klassifikationen
5. **Visualisierung** → Plots und Analysen

## Module im Detail

### MQTT Client (`mqtt_client/mqtt_client.py`)
- Verbindet sich mit MQTT-Broker
- Sammelt Daten von verschiedenen Dispensern (rot, blau, grün)
- Verarbeitet Füllstand, Vibration, Temperatur und Endgewicht
- Speichert Drop-Oscillation-Daten separat für Defekterkennung

### Datenverarbeitung (`database/`)
- **`transform.py`**: Konvertiert MQTT-Nachrichten in strukturierte Daten
- **`save_drop_oscillation.py`**: Speichert Vibrationsdaten für Klassifikation

### Regression (`linear_reg/reg_model.py`)
- Trainiert lineares Modell zur Gewichtsvorhersage
- Features: Vibrations-Indizes, Füllstände, Temperaturen
- Evaluiert mit MSE (Mean Squared Error)
- Erstellt Vorhersagen für neue Daten

### Klassifikation (`Classification/class.py`)
- Erkennt defekte Flaschen anhand von Vibrationssignalen
- Extrahiert statistische und FFT-Features
- Vergleicht verschiedene Klassifikationsalgorithmen
- Evaluiert mit F1-Score

### Visualisierung (`visualisierung/visualisierung.py`)
- Erstellt Zeitreihen-Plots von Sensordaten
- Konfigurierbar über `config.json`
- Speichert Plots als PNG-Dateien

## Ergebnisse

- **Regression**: MSE ≈ 0 bei Vollfeatures, MSE ≈ 40 bei reduzierten Features
- **Klassifikation**: F1-Scores variieren je nach Algorithmus und Features
- **Visualisierung**: Zeitreihen zeigen Produktionsmuster und Anomalien