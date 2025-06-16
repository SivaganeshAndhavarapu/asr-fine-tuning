# 🧪 ASR Fine-Tuning Environment Setup

This guide explains how to create a Python virtual environment and install required dependencies for fine-tuning Whisper models.

---

##  Prerequisites

- Python 3.8 or later
- `git` installed
- GPU with CUDA for training

---

## Clone the Repository

```bash
git https://github.com/SivaganeshAndhavarapu/asr-fine-tuning.git
cd asr-fine-tuning

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
