import paho.mqtt.client as mqtt
import json
import csv
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_config


config = load_config()

broker = config["mqtt"]["broker"]
port = config["mqtt"]["port"]
topic = config["mqtt"]["topic"]
username = config["mqtt"]["username"]
password = config["mqtt"]["password"]

# Nutze csv_path aus storage, nicht mqtt
csv_path = config["storage"]["csv_path"]

# === CSV Setup ===
# Absoluter Pfad zur CSV-Datei
CSV_FILE = os.path.join(os.path.dirname(__file__), csv_path)
FIELDNAMES = ["timestamp", "topic", "value"]

def write_csv(timestamp, topic, value):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"timestamp": timestamp, "topic": topic, "value": value})

# === MQTT Setup ===
def on_connect(client, userdata, flags, rc, properties=None):
    print("Verbunden mit Code:", rc)
    client.subscribe(topic)  # Topic aus config

def on_message(client, userdata, msg):
    timestamp = datetime.utcnow().isoformat()
    try:
        payload = msg.payload.decode()
        write_csv(timestamp, msg.topic, payload)
        print(f"[{timestamp}] {msg.topic}: {payload}")
    except Exception as e:
        print("Fehler beim Verarbeiten der Nachricht:", e)

client = mqtt.Client()
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port)
client.loop_forever()
