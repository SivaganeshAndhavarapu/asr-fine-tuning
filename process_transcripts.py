import os
import re
import argparse
from num2words import num2words

def normalize_text_line(line):
    line = line.lower()
    line = re.sub(r"\[.*?\]|\(.*?\)|<.*?>", "", line)  # Remove annotations
    line = re.sub(r"[^\w\s]", "", line)  # Remove punctuation
    line = re.sub(r"\d+", lambda x: num2words(x.group()), line)  # Expand numbers
    return line

def split_into_chunks(text, max_words=30):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def remove_unwanted_phrases(lines):
    remove_patterns = [
        re.compile(r"transcribed by teres", re.IGNORECASE),
        re.compile(r"end of (the )?days proceedings", re.IGNORECASE)
    ]
    cleaned = []
    for line in lines:
        for pattern in remove_patterns:
            line = pattern.sub("", line)
        cleaned.append(line)
    return cleaned

def process_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            # Normalize + split
            processed_lines = []
            for line in lines:
                normalized = normalize_text_line(line.strip())
                chunks = split_into_chunks(normalized)
                processed_lines.extend(chunk + "\n" for chunk in chunks)

            # Remove unwanted phrases
            cleaned_lines = remove_unwanted_phrases(processed_lines)

            # Save
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(cleaned_lines)

            print(f"✅ Processed: {filename}")

    print(f"\n🎉 All files processed and saved to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize, chunk, and clean transcript .txt files.")
    parser.add_argument("--input", type=str, required=True, help="Path to input folder containing .txt files")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder to save processed files")

    args = parser.parse_args()
    process_files(args.input, args.output)