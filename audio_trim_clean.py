import os
import argparse
from pydub import AudioSegment
import librosa
import soundfile as sf
import pandas as pd

def try_load_audio(file_path):
    try:
        return AudioSegment.from_file(file_path, format="mp3")
    except Exception:
        try:
            return AudioSegment.from_file(file_path, format="mp4")
        except Exception as e:
            print(f"❌ Could not load {file_path}: {e}")
            return None

def process_audio_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            file_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            wav_path = os.path.join(output_folder, f"{base_name}.wav")

            audio = try_load_audio(file_path)
            if audio is None:
                continue

            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")

            y, sr = librosa.load(wav_path, sr=16000)
            original_duration = librosa.get_duration(y=y, sr=sr)

            trimmed, index = librosa.effects.trim(y, top_db=20)
            trimmed_duration = librosa.get_duration(y=trimmed, sr=sr)

            start_trim_sec = index[0] / sr
            end_trim_sec = (len(y) - index[1]) / sr

            trimmed_path = os.path.join(output_folder, f"{base_name}_trimmed.wav")
            sf.write(trimmed_path, trimmed, sr)

            results.append({
                "filename": filename,
                "original_duration_sec": round(original_duration, 2),
                "trimmed_duration_sec": round(trimmed_duration, 2),
                "leading_silence_sec": round(start_trim_sec, 2),
                "trailing_silence_sec": round(end_trim_sec, 2),
            })

    # Save summary CSV
    report_path = os.path.join(output_folder, "audio_trim_report.csv")
    df = pd.DataFrame(results)
    df.to_csv(report_path, index=False)
    print(f"\n✅ Processing complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mp3 to mono 16kHz .wav and trim silence for ASR.")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing .mp3 files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for .wav and trimmed files")

    args = parser.parse_args()
    process_audio_folder(args.input, args.output)