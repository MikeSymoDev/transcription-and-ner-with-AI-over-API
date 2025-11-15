"""
This script uses the Google Gemini API to process PDF files.
It converts each PDF page into an image, sends it to Gemini for transcription,
and saves the extracted text as .txt files.
"""

import os
import time
from pdf2image import convert_from_path
from dotenv import load_dotenv
import google.generativeai as genai

# Setup 
load_dotenv()

start_time = time.time()
total_files = 0
total_in_tokens = 0
total_out_tokens = 0
input_cost_per_mio_in_dollars = 2.5
output_cost_per_mio_in_dollars = 10

input_dir = "../pdf_data_transcript/fraktur"
output_dir = "../answers/google_transcript"
os.makedirs(output_dir, exist_ok=True)

# Gemini API setup
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte .env Datei prüfen!")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")


'''
# Clear output directory (optional) 
for root, _, filenames in os.walk(output_dir):
    for filename in filenames:
        os.remove(os.path.join(root, filename))
'''


# Process PDFs 
for root, _, filenames in os.walk(input_dir):
    for filename in filenames:
        if filename.lower().endswith(".pdf"):
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
                prompt = (
                    """
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

              
                    ."""


                )

                try:
                    answer = model.generate_content([prompt, image])
                    answer_text = answer.text or ""
                    total_in_tokens += answer.usage_metadata.prompt_token_count
                    total_out_tokens += answer.usage_metadata.candidates_token_count
                    print("Done.")

                    # Save transcription
                    base_name = os.path.splitext(filename)[0]
                    out_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(answer_text)

                except Exception as e:
                    print(f"\n❌ Fehler bei Seite {i+1} von {filename}: {e}")

#  Summary 
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
