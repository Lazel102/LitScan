# LitScan – A GPT-Assisted Tool for Systematic Literature 

**LitScan** is a semi-automated pipeline for conducting systematic reviews on scientific papers using Large Language Models (LLMs). Originally developed during a research internship at ENS-PSL, it is tailored to support meta-analyses on Theory of Mind (ToM) capabilities in LLMs.

## 🚀 Key Features

- Automated screening of scientific PDFs using GPT (3.5 or 4-turbo)
- Structured extraction of study metadata and quantitative results
- Iterative analysis with a persistent global reflection (cumulative reasoning across papers)
- Output in CSV and JSON formats for further analysis
- Ready-to-use for funnel plots, forest plots, and bar charts
- Debug mode with mock responses or GPT-3.5

---

## 📁 Project Structure

```
LitScan/
├── articles/             # Raw PDF articles
├── data/
│   ├── csv/              # Extracted tabular data
│   └── json/             # Per-paper GPT responses + parsed content
├── figures/              # Visualizations (e.g. model/task distributions)
├── scripts/              # Python scripts for screening and deep analysis
├── paper/                # LaTeX report document
└── venv/                 # Virtual environment
```

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/LitScan.git
cd LitScan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

---

## 🧪 Usage

### Step 1: Initial Screening

```bash
python scripts/screening.py --pdf_dir ./data/articles --output ./data/csv/reviewed_papers.csv --model gpt-3.5-turbo
```

### Step 2: In-Depth Extraction (for selected papers)

```bash
python scripts/deep_analysis.py --input ./data/csv/reviewed_papers.csv --json_out ./data/json/
```

> All GPT prompts, responses, and structured data are saved for transparency and reuse.

---


---

## 📄 Output

- `reviewed_papers.csv` – Screening results
- `final_review_processed.csv` – Cleaned dataset for meta-analysis
- `json/*.json` – Structured per-paper evaluations

---

## 🧠 About the Project

LitScan was built to explore hybrid workflows between AI and human reasoning in systematic reviews, particularly where high-volume or fast-moving literature makes manual screening infeasible.

Author: Yannick Zelle  
Institution: École Normale Supérieure (ENS-PSL), Paris  
Year: 2025

---

## 📜 License

This repository is currently private and intended for academic research and prototyping. For collaboration or licensing inquiries, please get in touch.
