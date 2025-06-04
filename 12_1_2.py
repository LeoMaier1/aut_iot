import paho.mqtt.client as mqtt
import time
import random  # Nur als Platzhalter für echte Sensorwerte

# === Konfiguration ===
broker = "158.180.44.197"
port = 1883
gruppe = "FauLe"  # <-- Eigene Gruppe einsetzen
namen = "Augschoell, Maier"
messgröße = "füllstand"
einheit = "cm"

# === Topics ===
topic_groupsname = f"aut/{gruppe}/$groupsname"
topic_names = f"aut/{gruppe}/names"
topic_fuellstand = f"aut/{gruppe}/{messgröße}"
topic_fuellstand_unit = f"aut/{gruppe}/{messgröße}/$unit"

# === Callback (optional) ===
def on_message(client, userdata, message):
    print("Nachricht empfangen:")
    print(f"{message.topic}: {message.payload.decode()}")

# === MQTT-Client einrichten ===
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.username_pw_set("bobm", "letmein")
mqttc.on_message = on_message
mqttc.connect(broker, port)
mqttc.loop_start()

# === Einmalige Daten beim Start senden (retain=True) ===
mqttc.publish(topic_groupsname, gruppe, retain=True)
mqttc.publish(topic_names, namen, retain=True)
mqttc.publish(topic_fuellstand_unit, einheit, retain=True)

# === Periodisch Füllstand senden (alle 10 Sek.) ===
try:
    while True:
        fuellstand = random.randint(0, 100)  # z.B. Ultraschallsensorwert
        mqttc.publish(topic_fuellstand, fuellstand, retain=True)
        print(f"Gesendet: {fuellstand} {einheit}")
        time.sleep(10)
except KeyboardInterrupt:
    print("Beendet durch Benutzer.")
    mqttc.loop_stop()
    mqttc.disconnect()
