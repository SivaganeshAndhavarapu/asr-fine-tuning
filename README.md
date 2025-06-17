# ASR Fine-Tuning Environment Setup

This guide explains how to create a Python virtual environment and install required dependencies for fine-tuning ASR models.

---

##  Prerequisites

- Python 3.8 or later
- `git` installed
- GPU with CUDA for training

---

## Clone the Repository

git clone https://github.com/SivaganeshAndhavarapu/asr-fine-tuning.git <br>
cd asr-fine-tuning

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
---
### Download the pdf tanscrpit and audio files

```bash
python download_files.py --csv files.csv --out file_downloads --workers 4

 Arguments:

    --csv: Path to the CSV file containing download URLs. (path to both transcpit and audio files path in the csv)
    --out: Output directory where downloaded files will be saved.
    --workers: Number of parallel download threads (default is 4).
```
---
###  Clean and Process PDF Files

Use the script below to clean raw PDFs and save the processed text to an output directory.

```bash
python clean_pdfs.py --input /path/to/pdf_folder --output /path/to/output_folder

 Arguments:

    --input: Path to the folder containing raw PDF files.
    --output: Destination folder where cleaned and processed outputs will be saved.
```
---
### Process Transcript Text Files

This script is used to clean or normalize raw `.txt` transcript files.

```bash
python process_transcripts.py --input /path/to/raw_txts --output /path/to/cleaned_txts

 Arguments:

    --input: Directory containing raw transcript .txt files.
    --output: Directory where cleaned transcript files will be saved.
```
---
##  Trim and Convert Audio Files

Use this script to trim, clean, and convert `.mp3` audio files into `.wav` format (mono, 16kHz).

```bash
python audio_trim_clean.py --input /path/to/mp3s --output /path/to/save_wavs

Arguments:

    --input: Directory containing input .mp3 audio files.
    --output: Destination directory to save processed .wav files.
```
---
## Forced Alignment with Aeneas

This script performs forced alignment between `.wav` audio files and their corresponding `.txt` transcript files using the Aeneas aligner.

```bash
python align_with_aeneas.py \
  --wav_folder /home/sivaganesh/ada_exp/file_downloads_4 \
  --txt_folder /home/sivaganesh/ada_exp/file_download_4_txt_pro \
  --output_folder /home/sivaganesh/ada_exp/aligned_output_1

Arguments:

    --wav_folder: Directory containing input .wav files (mono, 16kHz recommended).
    --txt_folder: Directory containing corresponding .txt transcript files.
    --output_folder: Directory where alignment results will be saved.
```
---


## Start ASR Fine-Tuning

```bash
python train.py

Arguments:

    --csv: audio path with text contains hard coded in the scrpit for train data.

```
---
## Evaluate Fine-Tuned Whisper Model

```bash
python evaluate_whisper.py

Arguments:

    --csv: audio path with text contains hard coded in the scrpit for test data.

```
---
