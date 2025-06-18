import paho.mqtt.client as mqtt
import json
import csv
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_config
from database.transform import transform_data
from database.save_drop_oscillation import save_drop_oscillation

config = load_config()
broker = config["mqtt"]["broker"]
port = config["mqtt"]["port"]
topic = config["mqtt"]["topic"]
username = config["mqtt"]["username"]
password = config["mqtt"]["password"]
csv_path = config["storage"]["csv_path"]
CSV_FILE = os.path.join(os.path.dirname(__file__), csv_path)

# Bestehende Daten lesen
with open('database/data.csv', 'r') as file:
    existing_data = file.read()

# Neue Datei mit Header schreiben
FIELDNAMES = [
    "bottle",
    "vibration_index_red", "fill_level_grams_red", 
    "vibration_index_blue", "fill_level_grams_blue",
    "vibration_index_green", "fill_level_grams_green",
    "temperature_green", "temperature_red", "temperature_blue",
    "final_weight"
]

with open('database/data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(FIELDNAMES)  # Header schreiben
    file.write(existing_data)    # Bestehende Daten anhängen

# Sammle Daten pro bottle
bottle_data = {}
last_bottle_per_color = {"red": None, "blue": None, "green": None}

def write_csv(row):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()  # Schreibt automatisch die Titelzeile
        writer.writerow(row)

def on_connect(client, userdata, flags, rc, properties=None):
    print("Verbunden mit Code:", rc)
    client.subscribe(topic)

def on_message(client, userdata, msg):
    # --- Drop-Oscillation: Direkt speichern, keine transform_data! ---
    if msg.topic.endswith("drop_oscillation"):
        try:
            data = json.loads(msg.payload.decode())
            bottle = data.get("bottle")
            drop_osc = data.get("drop_oscillation")
            if bottle and drop_osc:
                save_drop_oscillation(bottle, drop_osc)  # Übergib Liste, nicht String!
                print(f"Drop-Oscillation gespeichert für bottle {bottle}")
            else:
                print("Drop-Oscillation: bottle oder drop_oscillation fehlt!")
        except Exception as e:
            print(f"Fehler beim Speichern von drop_oscillation: {e}")
        return  # Danach keine weitere Verarbeitung!

    # --- Normale Verarbeitung für alle anderen Topics ---
    timestamp = datetime.utcnow().isoformat()
    bottle, values = transform_data(timestamp, msg.topic, msg.payload.decode())
    print(f"Empfangen: topic={msg.topic}, bottle={bottle}, values={values}")

    # Spezialbehandlung für Temperature ohne bottle-ID
    if "temperature" in msg.topic and (not bottle or bottle == ""):
        # Bestimme Farbe und verwende die zuletzt aktive Flasche dieser Farbe
        if "temperature_red" in values and last_bottle_per_color["red"]:
            bottle = last_bottle_per_color["red"]
            values["bottle"] = bottle
        elif "temperature_blue" in values and last_bottle_per_color["blue"]:
            bottle = last_bottle_per_color["blue"]
            values["bottle"] = bottle
        elif "temperature_green" in values and last_bottle_per_color["green"]:
            bottle = last_bottle_per_color["green"]
            values["bottle"] = bottle

    if not bottle or not values:
        print("Kein bottle oder keine Werte extrahiert!")
        return

    # Merke dir die letzte Flasche pro Farbe
    if "fill_level_grams_red" in values:
        last_bottle_per_color["red"] = bottle
    elif "fill_level_grams_blue" in values:
        last_bottle_per_color["blue"] = bottle
    elif "fill_level_grams_green" in values:
        last_bottle_per_color["green"] = bottle

    if bottle not in bottle_data:
        bottle_data[bottle] = {"bottle": bottle}
    bottle_data[bottle].update(values)
    print(f"Aktueller Stand für bottle {bottle}: {bottle_data[bottle]}")

    required = set(FIELDNAMES) - {"bottle"}
    if required.issubset(bottle_data[bottle].keys()):
        write_csv(bottle_data[bottle])
        print(f"Datensatz geschrieben: {bottle_data[bottle]}")
        del bottle_data[bottle]

client = mqtt.Client()
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port)
client.loop_forever()
