import pandas as pd
import matplotlib.pyplot as plt
import os

# Daten laden
csv_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data.csv')
df = pd.read_csv(csv_path)

# Konfiguration
column = 'fill_level_grams_red'  # Welche Spalte plotten
start_row = 0                    # Start-Zeile
end_row = 200                    # End-Zeile

print(f"Erstelle Plot für {column} (Zeilen {start_row}-{end_row})")

# Datentypen überprüfen und konvertieren
print(f"Datentyp von {column}: {df[column].dtype}")

# Alle numerischen Spalten zu float konvertieren (außer bottle)
numeric_columns = ['vibration_index_red', 'fill_level_grams_red', 'vibration_index_blue', 
                   'fill_level_grams_blue', 'vibration_index_green', 'fill_level_grams_green',
                   'temperature_green', 'temperature_red', 'temperature_blue', 'final_weight']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Nach Konvertierung - Datentyp von {column}: {df[column].dtype}")

# Daten filtern
data_slice = df.iloc[start_row:end_row]

# Plot erstellen
plt.figure(figsize=(12, 6))
plt.plot(data_slice[column], linewidth=2, marker='o', markersize=3)
plt.title(f'{column.replace("_", " ").title()} - Zeitreihe (Zeilen {start_row}-{end_row})')
plt.xlabel('Datenpunkt')
plt.ylabel(column.replace("_", " ").title())
plt.grid(True, alpha=0.3)

# Speichern
filename = f'{column}_zeitreihe.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot gespeichert als: {filename}")

# Statistiken ausgeben (mit richtiger Formatierung)
print(f"Statistiken für {column}:")
print(f"   Min: {data_slice[column].min():.2f}")
print(f"   Max: {data_slice[column].max():.2f}")
print(f"   Mittelwert: {data_slice[column].mean():.2f}")
print(f"   Standardabweichung: {data_slice[column].std():.2f}")
print(f"   Anzahl Datenpunkte: {len(data_slice)}")

# Zeige erste und letzte Werte
print(f"\nErste 5 Werte:")
print(data_slice[column].head().tolist())
print(f"Letzte 5 Werte:")
print(data_slice[column].tail().tolist())