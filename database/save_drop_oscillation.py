import csv
import os

VIBRATION_CSV = os.path.join(os.path.dirname(__file__), "vibration.csv")
FIELDNAMES = ["bottle", "vibration_series", "status"]

def save_drop_oscillation(bottle, drop_oscillation, status="unknown"):
    """
    Speichert bottle, drop_oscillation (als Komma-String) und Status in vibration.csv.
    """
    file_exists = os.path.isfile(VIBRATION_CSV)
    with open(VIBRATION_CSV, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "bottle": bottle,
            "vibration_series": ",".join(map(str, drop_oscillation)),
            "status": status
        })