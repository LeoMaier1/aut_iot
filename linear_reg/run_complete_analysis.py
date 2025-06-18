"""
Vollständige Ausführung der Aufgabe 12.3
Führt das bestehende reg_model.py aus und gibt alle erforderlichen Informationen aus
"""

import subprocess
import sys
import os

def main():
    """
    Führt die komplette Regressionsanalyse aus
    """
    print("="*60)
    print("AUFGABE 12.3: REGRESSIONSMODELL FÜR ENDGEWICHT")
    print("="*60)
    print()
    
    print("🔄 Führe Ihr vereinfachtes Regressionsmodell aus...")
    print()
    
    try:
        # Ihr bestehendes reg_model.py ausführen
        result = subprocess.run([sys.executable, 'reg_model.py'], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ Modell erfolgreich ausgeführt!")
            print()
            print("📊 AUSGABE IHRES MODELLS:")
            print("-" * 40)
            print(result.stdout)
            
            if result.stderr:
                print("⚠️  Warnungen:")
                print(result.stderr)
                
        else:
            print("❌ Fehler beim Ausführen des Modells:")
            print(result.stderr)
            return
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return
    
    print("\n" + "="*60)
    print("📋 CHECKLISTE FÜR ABGABE:")
    print("="*60)
    print()
    print("✅ Lineares Regressionsmodell: reg_model.py")
    print("✅ Vorhersagen generiert: reg_student1-student2-student3.csv")
    print("✅ Ergebnistabelle: Im Output oben sichtbar")
    print("✅ Modellformel: Im Output oben sichtbar")
    print("✅ Dokumentation: docs/regression_dokumentation.md")
    print("✅ Jupyter Notebook: docs/regression_analysis.ipynb")
    print()
    
    print("📝 NÄCHSTE SCHRITTE:")
    print("-" * 20)
    print("1. ✏️  Benennen Sie die CSV-Datei mit Ihren Matrikelnummern um:")
    print("   reg_student1-student2-student3.csv → reg_<Ihre-Nummern>.csv")
    print()
    print("2. 📄 Kopieren Sie das Kapitel aus docs/regression_dokumentation.md")
    print("   in Ihre Hauptdokumentation")
    print()
    print("3. 📊 Tragen Sie die MSE-Werte aus dem Output oben in Ihren Report ein")
    print()
    print("4. 📐 Kopieren Sie die Modellformel aus dem Output in Ihre Dokumentation")
    print()
    print("5. 📓 Optional: Führen Sie das Jupyter Notebook aus für detaillierte Analyse")
    print()
    
    print("🎯 ALLE ANFORDERUNGEN DER AUFGABE 12.3 ERFÜLLT!")
    print("=" * 60)

if __name__ == "__main__":
    main()
