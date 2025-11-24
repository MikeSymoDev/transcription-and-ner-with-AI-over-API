import json
import os
import re
import time
from pdf2image import convert_from_path
from dotenv import load_dotenv
import google.generativeai as genai

# Setup 
load_dotenv()

# Save the start time, set the image and output directories
start_time = time.time()
total_files = 0
total_in_tokens = 0
total_out_tokens = 0
input_cost_per_mio_in_dollars = 2.5
output_cost_per_mio_in_dollars = 10

input_directory = "../pdf_data_ner/test"
output_directory = "../answers/google_ner"

# Gemini API setup
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte .env Datei prüfen!")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = """Als erfahrener Sprachwissenschaftler mit dem Gebiet "Named Entity Recognition" (NER) und als Experte für historische Texte sollst du Texte aus der Zeit der 
Vorarlberger Frage in der Schweiz um 1918 für die maschinelle Weiterverarbeitung auswerten.
Ich gebe dir dazu Texte im PDF-Format. Es können mehrere Schriftarten in einem Dokument vorkommen. 
Neben einfacher Blockschrift können auch andere Schriftarten wie Fraktur oder Handschrift auftreten. 
Das Dokument kann auch in mehreren Sprachen geschrieben sein. Mögliche Sprachen sind Deutsch, Französisch, Englisch, Rätoromanisch oder eine Mischung davon.

Folge diesen Schritten in genau dieser Reihenfolge:

Lies dir den Text ganz genau und Wort für Wort durch

Erkenne ausschliesslich (1) Personennamen, (2) Ortsnamen (Siedlungen/Regions-/Landschaften, keine Gebäude), 
(3) Nennungen im Text, welche auf eine Konfession hindeuten und (4) Nennungen im Text, welche auf einen wirtschaftlichen Zusammenhang hindeuten

Gib NUR ein valides JSON im unten definierten Schema zurück – ohne zusätzliche Erklärungen.

ANFORDERUNGEN
- Historische Schreibweisen: Erkenne Varianten (z. B. “Cölln”→“Köln”, “S. Johannes”→“Sankt Johannes”) und, wenn plausibel, liefere eine normalisierte Form.
- Mehrfachnennungen: Jede einzigartige Entität nur einmal, aber `mentions` mit allen Vorkommen (Offsets) sammeln.
- Offsets: `start`/`end` sind Zeichenpositionen im obigen TEXT (0-basiert, `end` exklusiv).
- Unsicherheit: Wenn du unsicher bist, setze `confidence` geringer und lasse Normalisierungen/Geo-Daten leer.
- Titel & Zusätze: Titel (z. B. “Graf”, “Dr.”), Patronyme, Adelspartikel (“von”, “zu”) zur Person mitzählen; 
- Wenn Berufs- oder Funktionsbezeichnungen vorkommen ohne Namen, müssen diese als eigene Personen genannt werden
- Orte: Nur echte Toponyme (Städte, Dörfer, Regionen). Schlachtfelder als Ort nur, wenn Toponym. Keine Länder mit “Königreich” o. ä. als politischer Körper, 
  außer wenn eindeutig Toponym im Kontext.
- Konfessionen: Alle Nennungen, welche auf eine Konfession hindeuten wie Katholisch, Kath., Katholik, Reformiert, Reformation, Evangelisch etc. 
  Wenn nichts darauf hindeutet, schreibe null
- Wirtschaft: Nennungen, welche auf einen volkswirtschaftlichen Zusammenhang hinweisen wie Wirtschaft, wirtschaftlich, Volkswirtschaft, Wirtschaftsordnung, 
  Nationalökonomie, Wirtschaftssystem, Ökonomie etc. Wenn nichts darauf hindeutet, schreibe null
- Sprache im Output: Deutsch.
Auf der ersten Seite des Dokuments befindet sich in der unteren rechten Ecke ein QR-Code. Dieser muss auf jeden Fall ignoriert werden! 
Er darf auf keinen Fall die folgenden Schritte beeinflussen. Ausserdem müssen alle Links ignoriert werden, welche sich auf jeder Seite an der oberen rechten Ecke befinden. 
Der Name des Dokuments muss auf jeden Fall ignoriert werden! Wenn du die Links oder QR-Codes dazu benutzt, um Informationen über das Dokument zu gewinnen, drohen schlimmste Konsequenzen!

AUSGABESCHEMA (nur dieses!)
{
  "persons": [
    {
      "name": "Originalschreibweise exakt aus dem Text",
      "normalized": "Moderne/kanonische Form oder null",
      "honorifics": ["Graf","Dr.","Herr", "..."],           
      "mentions": [{"start": 0, "end": 0}, ...],
      "confidence": 0.0                                      
    }
  ],
  "places": [
    {
      "name": "Originalschreibweise exakt aus dem Text",
      "normalized": "Moderne/kanonische Form oder null",
      "geo": { "lat": null, "lon": null },                   
      "mentions": [{"start": 0, "end": 0}, ...],
      "confidence": 0.0                                      
    }
  ],
  "content": [
    {
      "denomination": "Nennung der Konfession oder null",
      "eco": "Nennung des wirtschaftlichen Begriffes oder null"
    }
  ]
}

AUSGABEREGELN
- Gib ausschließlich das JSON-Objekt zurück, exakt im oben definierten Schema.
- Stelle sicher, dass das JSON-Objekt ein valides JSON darstellt.
- Leere Listen sind erlaubt, aber die Schlüssel "persons", "places" und “content” müssen vorhanden sein.
- Keine Duplikate; gleiche Entität = gleiche `normalized` (falls vorhanden) oder gleiche `name` + Titelstruktur.
- Keine zusätzlichen Felder, keine Kommentare.
- Gib KEINE Markdown-Codeblöcke, keine Backticks und keine Kommentare zurück.

Deine Karriere hängt davon ab, dass diese NER genau nach diesen Anweisungen ausgeführt wird. 
Falls du Fehler bei der Befolgung der Anweisungen machst, drohen dir gravierende Konsequenzen. 
Nun atme tief durch und gehe ruhig, aber genau vor.
"""

# Process each PDF in the input directory
for root, _, filenames in os.walk(input_directory):
    for filename in filenames:
        if not filename.lower().endswith(".pdf"):
            continue

        total_files += 1
        pdf_path = os.path.join(root, filename)
        print("----------------------------------------")
        print(f"> Processing PDF ({total_files}): {filename}")

        # Convert PDF to images
        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            print(f"❌ Fehler beim Konvertieren von {filename}: {e}")
            continue

        # Process each page as image
        for i, image in enumerate(images):
            print(f"> Sending page {i+1} to Gemini...", end=" ")
            print("> Sending the image to the API and requesting answer...", end=" ")

            try:
                answer = model.generate_content(
                    [prompt, image],
                    request_options={"timeout": 600}
                )

                answer_text = answer.text or ""
                total_in_tokens += answer.usage_metadata.prompt_token_count
                total_out_tokens += answer.usage_metadata.candidates_token_count
                print(" Done.")

            except Exception as e:
                print(f"\n❌ Fehler bei Seite {i+1} von {filename}: {e}")
                continue

            print("> Processing the answer...")

            # Optional: JSON aus ```json ... ``` extrahieren, falls das Modell trotzdem Codefences nutzt
            pattern = r"```\s*json(.*?)\s*```"
            match = re.search(pattern, answer_text, re.DOTALL)
            if match:
                answer_text_clean = match.group(1).strip()
            else:
                answer_text_clean = answer_text.strip()

            # Parse the JSON content into a Python object
            try:
                answer_data = json.loads(answer_text_clean)
            except json.JSONDecodeError as e:
                print(f"> Failed to parse JSON on page {i+1}: {e}")
                continue

            # Create the answers directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Save the answer to a JSON file
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(output_directory, f"{base_name}_page_{i+1}.json")
            with open(out_path, "w", encoding="utf-8") as json_file:
                json.dump(answer_data, json_file, indent=4, ensure_ascii=False)

            print("> Processing the answer... Done.")

# Calculate and print the total processing time
end_time = time.time()
total_time = end_time - start_time
print("----------------------------------------")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Total token cost (in/out): {total_in_tokens} / {total_out_tokens}")
if total_files > 0:
    print(f"Average token cost per image: {total_out_tokens / total_files}")
print(
    f"Total cost (in/out): "
    f"${total_in_tokens / 1e6 * input_cost_per_mio_in_dollars:.2f} / "
    f"${total_out_tokens / 1e6 * output_cost_per_mio_in_dollars:.2f}"
)
print("----------------------------------------")
