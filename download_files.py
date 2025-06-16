import os
import csv
import argparse
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_base_filename_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path).split('?')[0]
    return os.path.splitext(filename)[0]

def get_extension_from_url(url):
    path = urlparse(url).path
    return os.path.splitext(os.path.basename(path))[1]

def download_file(url, save_path):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Downloaded: {save_path}")
    except Exception as e:
        print(f"❌ Failed: {url} | Error: {e}")

def prepare_download_tasks(csv_path, output_dir):
    tasks = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            transcript_url = row['transcript']
            audio_url = row['audio_path']

            base_name = get_base_filename_from_url(transcript_url)
            pdf_ext = get_extension_from_url(transcript_url)

            pdf_save_path = os.path.join(output_dir, f"{base_name}{pdf_ext}")
            audio_save_path = os.path.join(output_dir, f"{base_name}.mp3")

            tasks.append((transcript_url, pdf_save_path))
            tasks.append((audio_url, audio_save_path))
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Download audio and transcript files from a CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file (with 'transcript' and 'audio_path' columns).")
    parser.add_argument("--out", type=str, required=True, help="Directory to save downloaded files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel download threads.")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    download_tasks = prepare_download_tasks(args.csv, args.out)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_file, url, path) for url, path in download_tasks]
        for future in as_completed(futures):
            future.result()  # raises any exceptions caught during execution

if __name__ == "__main__":
    main()