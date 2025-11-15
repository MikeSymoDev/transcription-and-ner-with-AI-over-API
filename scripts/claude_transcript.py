"""
Batch-Transkription von PDF-Seiten mit Anthropic Claude (Vision).
- Liest PDFs aus input_dir
- Rendert jede Seite als PNG (pdf2image)
- Sendet Prompt + Bild (Base64) an Claude
- Speichert Antwort pro Seite als .txt
- Summiert Tokenverbrauch und schätzt Kosten
"""

import base64
import json
import os
import re
import time
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_path
from anthropic import Anthropic

load_dotenv()

# Verzeichnisse
input_dir = "../pdf_data_transcript"
output_dir = "../answers/anthropic_transcript"
os.makedirs(output_dir, exist_ok=True)

# Optional: Ausgabeordner leeren
for root, _, filenames in os.walk(output_dir):
    for filename in filenames:
        try:
            os.remove(os.path.join(root, filename))
        except Exception:
            pass

# API-Client
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise RuntimeError("Kein API-Key gefunden. Bitte ANTHROPIC_API_KEY oder CLAUDE_API_KEY in .env setzen.")
client = Anthropic(api_key=api_key)

# Modell & Parameter
MODEL_NAME = "claude-sonnet-4-5-20250929"  
TEMPERATURE = 0.5
MAX_OUTPUT_TOKENS = 8192  

# Token-/Kosten-Tracking (Dollar pro 1 Mio. Tokens)
input_cost_per_mio_in_dollars = 3.00
output_cost_per_mio_in_dollars = 15.00


PROMPT = """
Als erfahrener Historiker mit der Spezialisierung auf die Vorarlberger Frage in der Schweiz im Jahr 1918 sollst du das angehängte Dokument transkribieren. Im folgenden Absatz sind Informationen über den geschichtlichen Kontext, damit du weisst, in welchem Kontext der Inhalt des angehängten Dokuments steht. Dieser Absatz darf aber auf keinen Fall als Informationsquelle für die Transkription dienen.

Die Volksabstimmung in Vorarlberg am 11. Mai 1919 entschied über die Frage, ob die Vorarlberger Landesregierung Beitrittsverhandlungen mit der Schweiz aufnehmen sollte. Nach dem verlorenen Ersten Weltkrieg herrschte in Vorarlberg zu Jahresbeginn 1919 eine drückende wirtschaftliche Not. Zugleich war das politische Schicksal Deutschösterreichs während der zeitgleich laufenden Pariser Friedensverhandlungen ungewiss. Vor diesem Hintergrund entwickelte sich in Vorarlberg, dem unmittelbar an den Schweizer Kanton St. Gallen grenzenden westlichsten österreichischen Kronland, eine starke Bewegung für einen Anschluss an die Schweizer Eidgenossenschaft. Wenngleich bei der Abstimmung 81 Prozent des Stimmvolks für die Aufnahme von Beitrittsverhandlungen stimmte, wurde dieser Anschluss letztlich nicht vollzogen. Vorarlberg wurde 1920 zu einem Land Österreichs und die Anschlusspläne waren nur wenige Jahre später politisch bedeutungslos. Auf Schweizer Seite gab es gegen das Vorhaben teils erhebliche Vorbehalte, weil ein Beitritt Vorarlbergs zu einer katholischen Konfessionsmehrheit geführt und das deutschsprachige Übergewicht verstärkt hätte. Der Bundesrat sprach sich schließlich für den Status quo aus. Zugleich formierte sich um den Freiburger rechtskonservativen Intellektuellen Gonzague de Reynold eine Bewegung, die die Aufnahme Vorarlbergs nachdrücklich befürwortete. Sie ließen eine Vielzahl von Plakaten, Flugblättern und Propagandamaterialien drucken, um die Schweizer Öffentlichkeit zu einer Befürwortung der Aufnahme Vorarlbergs zu bewegen. Die Schweizer Bundesregierung nahm währenddessen eine ausdrücklich neutrale Position ein. Einerseits stand sie einer Aufnahme Vorarlbergs nicht ablehnend gegenüber, wollte jedoch das diplomatische Verhältnis zu den Siegermächten des Ersten Weltkriegs darüber nicht belasten.

Befolge folgende Schritte unbedingt und nur in dieser Reihenfolge:
Auf der ersten Seite des Dokuments befindet sich in der unteren rechten Ecke ein QR-Code. Dieser muss auf jeden Fall ignoriert werden! Er darf auf keinen Fall die folgenden Schritte beeinflussen. Ausserdem müssen alle Links ignoriert werden, welche sich auf jeder Seite an der oberen rechten Ecke befinden. Die Links dürfen auf keinen Fall transkribiert werden! Der Name des Dokuments muss auf jeden Fall ignoriert werden! Wenn du die Links oder QR-Codes dazu benutzt, um Informationen über das Dokument zu gewinnen, drohen schlimmste Konsequenzen!
Analysiere das Dokument auf seine verwendeten Sprachen. Sollte es sich nicht um ein rein deutschsprachiges Dokument handeln, merke dir die Sprache. Mögliche Sprachen können Deutsch, Englisch und Französisch sein.
Analysiere das Dokument auf seine verwendete Schriftarten. Neben einfacher Blockschrift können auch andere Schriftarten wie Fraktur oder Handschrift auftreten. Merke dir die Schriftart, falls das Dokument in Fraktur oder Handschrift geschrieben ist. Es können mehrere Schriftarten in einem Dokument vorkommen.
Analysiere das Dokument auf Beschädigungen. Es könnte sein, dass gewisse Wörter durchgestrichen sind oder der Text übermalt wurde. Merke dir die beschädigten Textstellen.
Transkribiere das Dokument Wort für Wort. Der Text muss unverändert reproduziert werden. Links dürfen auf keinen Fall transkribiert werden. Schreibfehler dürfen auf keinen Fall korrigiert werden. Deine Karriere hängt davon ab, dass bei diesem Schritt keine Fehler gemacht werden.
befolge die folgenden Schritte:
Wenn das Dokument nicht deutschsprachig ist, übersetze die Transkription ins deutsche. Wenn die verwendete Schriftart Fraktur ist, überprüfe die Transkription auf Fehler, welche du bei der Transkription gemacht haben könntest.
Wenn du bei deiner Analyse Beschädigungen entdeckt hast, prüfe ob die beschädigten Stellen entzifferbar sind. Sind die Stellen entzifferbar, transkribiere das gesamte Dokument. Sind die beschädigten Stellen nicht entzifferbar, lasse die beschädigten Wörter oder Buchstaben in der Transkription aus. Es ist nicht schlimm, zuzugeben, wenn du etwas nicht entziffern kannst. Deine Ehrlichkeit ist von zentraler Bedeutung.
Überprüfe, dass alle Informationen für Transkription nur aus dem angehängten Dokument stammen. Jede andere Informationsquelle ist strengstens verboten!

Überprüfe, ob alle Schritte in der richtigen Reihenfolge eingehalten und ausgeführt wurden.
Ignorieren aller Links und QR-Codes
Analyse der Sprachen des Dokuments
Analyse der Schriftarten des Dokuments
Analyse der Beschädigungen des Dokuments
Transkription des Dokuments
Überprüfungen bezüglich Sprachen, Schriftarten und Beschädigungen
Überprüfung der Herkunft der Informationen

Deine Karriere hängt davon ab, dass diese Transkription genau nach diesen Anweisungen ausgeführt wird. Falls du Fehler bei der Befolgung der Anweisungen machst, drohen dir gravierende Konsequenzen!
Nun atme tief durch und gehe Schritt für Schritt vor.
"""


# Hilfsfunktionen

def pil_to_base64_png(img: Image.Image) -> str:
    """Wandelt ein PIL-Image in Base64(PNG) um."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_text_from_response(resp) -> str:
    """
    Extrahiert Text aus der Anthropic-Response.
    resp.content ist eine Liste von Content-Blocks (meist Text).
    """
    text_parts = []
    try:
        # v1 SDK: content = [TextBlock(type='text', text='...'), ...]
        for block in (resp.content or []):
            # Bei SDK-Objekten:
            txt = getattr(block, "text", None)
            if txt:
                text_parts.append(txt)
            # Fallback: falls dicts
            elif isinstance(block, dict):
                t = block.get("text")
                if t:
                    text_parts.append(t)
    except Exception:
        pass
    return "".join(text_parts).strip()


def get_usage_tokens(resp) -> tuple[int, int]:
    """
    Liefert (input_tokens, output_tokens), wenn vorhanden.
    """
    in_toks = 0
    out_toks = 0
    try:
        in_toks = int(getattr(resp.usage, "input_tokens", 0) or 0)
        out_toks = int(getattr(resp.usage, "output_tokens", 0) or 0)
    except Exception:
        pass
    return in_toks, out_toks


def send_page_to_claude(image: Image.Image, prompt: str) -> tuple[str, int, int]:
    """
    Sendet eine Seite (PIL.Image) + Prompt an Claude.
    Gibt (antwort_text, input_tokens, output_tokens) zurück.
    """
    image_b64 = pil_to_base64_png(image)
    resp = client.messages.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
            ],
        }],
    )
    answer_text = extract_text_from_response(resp)
    in_toks, out_toks = get_usage_tokens(resp)
    return answer_text, in_toks, out_toks


# Hauptlogik

def main():
    start_time = time.time()
    total_files = 0
    total_in_tokens = 0
    total_out_tokens = 0

    print("----------------------------------------")
    print(f"Suche PDFs in: {os.path.abspath(input_dir)}")

    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if not filename.lower().endswith(".pdf"):
                continue

            total_files += 1
            pdf_path = os.path.join(root, filename)
            print("----------------------------------------")
            print(f"> Verarbeite PDF ({total_files}): {filename}")

            # PDF -> Bilder (DPI ggf. erhöhen für bessere Erkennung)
            try:
                images = convert_from_path(pdf_path, dpi=300)  # dpi=300 meist sinnvoll
            except Exception as e:
                print(f"❌ Fehler beim Konvertieren von {filename}: {e}")
                continue

            # Jede Seite senden
            for i, image in enumerate(images):
                print(f"> Sende Seite {i+1} an Claude...", end=" ", flush=True)
                try:
                    answer_text, in_toks, out_toks = send_page_to_claude(image, PROMPT)
                    total_in_tokens += in_toks
                    total_out_tokens += out_toks

                    # Ergebnis speichern
                    base_name = os.path.splitext(filename)[0]
                    out_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(answer_text or "")

                    print("Done.")
                except Exception as e:
                    print(f"\n❌ Fehler bei Seite {i+1} von {filename}: {e}")

    # Zusammenfassung
    end_time = time.time()
    duration = end_time - start_time

    print("----------------------------------------")
    print(f"Total processing time: {duration:.2f} seconds")
    print(f"Total token cost (in/out): {total_in_tokens} / {total_out_tokens}")

    if total_files > 0:
        avg_out = total_out_tokens / total_files
        print(f"Average output tokens per file: {avg_out:.2f}")
    else:
        print("No files were processed — check input directory or file types.")

    total_in_cost = total_in_tokens / 1e6 * input_cost_per_mio_in_dollars
    total_out_cost = total_out_tokens / 1e6 * output_cost_per_mio_in_dollars
    print(f"Estimated cost (in/out): ${total_in_cost:.2f} / ${total_out_cost:.2f}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
