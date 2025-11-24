import os
import time
import json
from pdf2image import convert_from_path
from dotenv import load_dotenv
import google.generativeai as genai

# --- Setup ---
load_dotenv()

start_time = time.time()
total_files = 0
total_in_tokens = 0
total_out_tokens = 0
input_cost_per_mio_in_dollars = 2.5
output_cost_per_mio_in_dollars = 10

input_dir = "../pdf_data_ner"
output_dir = "../answers/google_ner"
os.makedirs(output_dir, exist_ok=True)

# --- Gemini API setup ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte .env Datei prüfen!")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Process PDFs ---
for root, _, filenames in os.walk(input_dir):
    for filename in filenames:
        if filename.lower().endswith(".pdf"):
            total_files += 1
            pdf_path = os.path.join(root, filename)
            print("----------------------------------------")
            print(f"> Processing PDF ({total_files}): {filename}")

            try:
                images = convert_from_path(pdf_path)
            except Exception as e:
                print(f"❌ Fehler beim Konvertieren von {filename}: {e}")
                continue

            # Process each page
            for i, image in enumerate(images):
                print(f"> Sending page {i+1} to Gemini...", end=" ")

                prompt =  """
                    

Du bist ein NER-Extractor für historische deutschsprachige Texte (ca. 15.–20. Jh.). Erkenne ausschließlich
(1) Personennamen und (2) Ortsnamen (Siedlungen/Regions-/Landschaften, keine Gebäude). Gib NUR ein valides JSON im unten definierten Schema zurück – ohne zusätzliche Erklärungen.


ANFORDERUNGEN
- Historische Schreibweisen: Erkenne Varianten (z. B. “Cölln”→“Köln”, “S. Johannes”→“Sankt Johannes”) und, wenn plausibel, liefere eine normalisierte Form.
- Mehrfachnennungen: Jede einzigartige Entität nur einmal, aber `mentions` mit allen Vorkommen (Offsets) sammeln.
- Offsets: `start`/`end` sind Zeichenpositionen im obigen TEXT (0-basiert, `end` exklusiv).
- Unsicherheit: Wenn du unsicher bist, setze `confidence` geringer und lasse Normalisierungen/Geo-Daten leer.
- Titel & Zusätze: Titel (z. B. “Graf”, “Dr.”), Patronyme, Adelspartikel (“von”, “zu”) zur Person mitzählen; Berufs- oder Funktionsbezeichnungen NICHT als eigene Personen.
- Orte: Nur echte Toponyme (Städte, Dörfer, Regionen). Schlachtfelder als Ort nur, wenn Toponym. Keine Länder mit “Königreich” o. ä. als politischer Körper, außer wenn eindeutig Toponym im Kontext.
- Keine Datums-/Ereignis- oder Institutions-Extraktion.
- Sprache im Output: Deutsch.

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
  ]
}

AUSGABEREGELN
- Gib ausschließlich das JSON-Objekt zurück, exakt im oben definierten Schema.
- Leere Listen sind erlaubt, aber die Schlüssel "persons" und "places" müssen vorhanden sein.
- Keine Duplikate; gleiche Entität = gleiche `normalized` (falls vorhanden) oder gleiche `name` + Titelstruktur.
- Keine zusätzlichen Felder, keine Kommentare.
- Gib KEINE Markdown-Codeblöcke, keine Backticks und keine Kommentare zurück.

                    ."""

                try:
                    answer = model.generate_content([prompt, image])
                    answer_text = answer.text or ""

                    total_in_tokens += answer.usage_metadata.prompt_token_count
                    total_out_tokens += answer.usage_metadata.candidates_token_count

                    print("Done.")

                    # --- JSON ---
                    base_name = os.path.splitext(filename)[0]
                    out_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.json")

                    try:
                        parsed = json.loads(answer_text)
                    except json.JSONDecodeError:
                        parsed = {
                            "error": "LLM output is not valid JSON",
                            "raw_text": answer_text
                        }

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"\n❌ Fehler bei Seite {i+1} von {filename}: {e}")

# --- Summary ---
end_time = time.time()
total_time = end_time - start_time
print("----------------------------------------")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Total token cost (in/out): {total_in_tokens} / {total_out_tokens}")

if total_files > 0:
    print(f"Average token cost per file: {total_out_tokens / total_files:.2f}")
else:
    print("No files were processed — check input directory or file types.")

print(f"Total cost (in/out): ${total_in_tokens / 1e6 * input_cost_per_mio_in_dollars:.2f} / "
      f"${total_out_tokens / 1e6 * output_cost_per_mio_in_dollars:.2f}")
print("----------------------------------------")
