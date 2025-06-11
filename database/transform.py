import json

def transform_data(timestamp, topic, payload):
    """
    Transformiert die empfangenen Daten und gibt alle relevanten Felder zurück.
    """
    try:
        # JSON-Daten parsen
        data = json.loads(payload)
    except Exception:
        return None, None

    # Extrahiere bottle id
    bottle = str(data.get("bottle") or data.get("bottle_id") or "")

    # Je nach Topic die Werte extrahieren
    result = {"bottle": bottle}
    
    if "dispenser_red" in topic:
        result["fill_level_grams_red"] = data.get("fill_level_grams")
        result["vibration_index_red"] = data.get("vibration-index")  # Aus dispenser, nicht drop_oscillation
    elif "dispenser_blue" in topic:
        result["fill_level_grams_blue"] = data.get("fill_level_grams")
        result["vibration_index_blue"] = data.get("vibration-index")  # Aus dispenser
    elif "dispenser_green" in topic:
        result["fill_level_grams_green"] = data.get("fill_level_grams")
        result["vibration_index_green"] = data.get("vibration-index")  # Aus dispenser
    elif "temperature" in topic:
        color = data.get("dispenser") or data.get("color")
        if color == "red":
            result["temperature_red"] = data.get("temperature_C")
        elif color == "blue":
            result["temperature_blue"] = data.get("temperature_C")
        elif color == "green":
            result["temperature_green"] = data.get("temperature_C")
    elif "scale/final_weight" in topic:
        result["final_weight"] = data.get("final_weight")
    else:
        return None, None

    return bottle, result