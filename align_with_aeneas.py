# save as align_with_aeneas.py
import os
import subprocess
import argparse

def align_files(wav_folder, txt_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(wav_folder):
        if filename.endswith(".wav"):
            base_name = os.path.splitext(filename)[0]
            wav_path = os.path.join(wav_folder, filename)
            txt_path = os.path.join(txt_folder, f"{base_name}.txt")
            csv_path = os.path.join(output_folder, f"{base_name}.csv")

            if os.path.exists(txt_path):
                print(f"⏳ Aligning: {filename} + {base_name}.txt")

                command = [
                    "python", "-m", "aeneas.tools.execute_task",
                    wav_path,
                    txt_path,
                    "task_language=eng|os_task_file_format=csv|is_text_type=plain",
                    csv_path
                ]

                try:
                    subprocess.run(command, check=True)
                    print(f"✅ Saved: {csv_path}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error processing {filename}: {e}")
            else:
                print(f"⚠️ No matching TXT for {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Aeneas alignment for WAV-TXT pairs.")
    parser.add_argument("--wav_folder", required=True, help="Folder containing WAV files")
    parser.add_argument("--txt_folder", required=True, help="Folder containing text transcripts")
    parser.add_argument("--output_folder", required=True, help="Folder to save CSV alignments")

    args = parser.parse_args()
    align_files(args.wav_folder, args.txt_folder, args.output_folder)