import re
import os
import argparse
from PyPDF2 import PdfReader

def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    current_speaker = None
    current_speech = []

    for line in lines:
        line = re.sub(r'\s+', ' ', line.strip())
        line = re.sub(r'^\d+\s*', '', line)
        line = re.sub(r'\s+\d+$', '', line)
        line = re.sub(r'\s\d+\s', ' ', line)

        if re.match(r'^[A-Z .]+:', line):
            if current_speaker:
                cleaned_lines.append(current_speaker)
                cleaned_lines.append(' '.join(current_speech).strip())
                cleaned_lines.append('')
            speaker = re.sub(r'\s+(:)', r'\1', line)
            current_speaker = speaker
            current_speech = []
        else:
            current_speech.append(line)

    if current_speaker:
        cleaned_lines.append(current_speaker)
        cleaned_lines.append(' '.join(current_speech).strip())

    return '\n'.join(cleaned_lines)

def clean_transcript_grouped_lines(text):
    lines = text.split('\n')
    speaker_pattern = re.compile(r'^([A-Z .]+):')

    unique_speakers = set()
    cleaned_lines = []

    current_speaker = None
    current_speech_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper() == "END OF THIS PROCEEDING":
            continue

        match = speaker_pattern.match(line)
        if match:
            if current_speaker and current_speech_lines:
                speech = ' '.join(current_speech_lines)
                speech = re.sub(r'\s+', ' ', speech).strip()
                cleaned_lines.append(speech)
            current_speaker = match.group(1).strip()
            unique_speakers.add(current_speaker)
            speech_part = line[match.end():].strip()
            current_speech_lines = [speech_part] if speech_part else []
        else:
            current_speech_lines.append(line)

    if current_speaker and current_speech_lines:
        speech = ' '.join(current_speech_lines)
        speech = re.sub(r'\s+', ' ', speech).strip()
        cleaned_lines.append(speech)

    return cleaned_lines, unique_speakers

def process_pdfs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            pdf_base = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{pdf_base}.txt")

            try:
                reader = PdfReader(pdf_path)
                if len(reader.pages) < 2:
                    print(f"⚠️ Skipping {filename}: less than 2 pages.")
                    continue

                all_text = ''
                for i in range(1, len(reader.pages)):
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        all_text += page_text + '\n'

                cleaned_text = clean_text(all_text)
                cleaned_lines, unique_speakers = clean_transcript_grouped_lines(cleaned_text)

                with open(output_path, 'w', encoding='utf-8') as f:
                    for line in cleaned_lines:
                        f.write(line + '\n')

                print(f"✅ Processed {filename} — Unique speakers: {len(unique_speakers)} — Saved as {output_path}")

            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and extract speaker-diarized text from PDF court transcripts.")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing PDF files.")
    parser.add_argument("--output", type=str, required=True, help="Output folder to save cleaned .txt files.")

    args = parser.parse_args()
    process_pdfs(args.input, args.output)