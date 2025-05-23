from pathlib import Path
import csv
import os
import openai
import fitz  # PyMuPDF
import uuid
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
client = openai.OpenAI(api_key=api_key)

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
pdf_dir = BASE_DIR / "data" / "articles"
csv_path = BASE_DIR / "data" / "csv" / "final_review.csv"
input_screened_csv = BASE_DIR / "data" / "csv" / "reviewed_papers_checked.csv"
json_dir = BASE_DIR / "data" / "json"
json_dir.mkdir(parents=True, exist_ok=True)

# CSV schema (added 'Findings')
CSV_COLUMNS = [
    "ID", "Title", "Authors", "Year", "PublicationType", "Models", "ModelAccessDetails",
    "ComparedToHumans", "TaskTypes", "TaskOrder", "ToMTaskDescriptions",
    "QuantitativeMetrics", "SampleSize", "StatisticalSignificance",
    "Summary", "Findings", "AddToFinalSet", "Justification", "GlobalReflection"
]

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc]).strip()

def load_global_reflection(csv_path):
    if not csv_path.exists():
        return "No prior reflection."
    df = pd.read_csv(csv_path)
    reflections = df["GlobalReflection"].dropna().tolist()
    return reflections[-1] if reflections else "No prior reflection."

def summarize_prior_studies(csv_path):
    if not csv_path.exists():
        return "No prior included studies."
    df = pd.read_csv(csv_path)
    included = df[df["AddToFinalSet"].str.lower() == "yes"]
    summaries = []
    for _, row in included.iterrows():
        summaries.append(
            f"- Title: {row['Title']}\n"
            f"  Models: {row['Models']}\n"
            f"  TaskTypes: {row['TaskTypes']}\n"
            f"  Metrics: {row['QuantitativeMetrics']}\n"
            f"  Justification: {row['Justification'][:200]}..."
        )
    return "Previously Included Studies:\n" + "\n".join(summaries) if summaries else "No prior included studies."

def create_prompt(previous_studies, reflection, paper_text):
    return f"""
    {previous_studies}

    Previous GlobalReflection:
    {reflection}
    
    Full Paper Text:
    {paper_text}
    
    Now extract and fill the following fields (in plain text, no markdown). Please normalize the names for Tasks and Models for consistency:
    
    ID:
    Title:
    Authors:
    Year:
    PublicationType: (e.g., journal, preprint, conference, thesis)
    Models:
    ModelAccessDetails: (e.g., API use, fine-tuned, open-source)
    ComparedToHumans: yes / no / not reported
    TaskTypes:
    TaskOrder: (e.g., first-order, second-order, mixed)
    ToMTaskDescriptions:
    QuantitativeMetrics: (e.g., accuracy, pass rate, p-values — please extract values explicitly in structured form. If multiple p-values or results are present, list them all in a structured JSON-like format, such as:
      {{
        "model": "GPT-3",
        "task": "False Belief",
        "ToM-order": "2",
        "accuracy": 0.82,
        "pass_rate": "85%",
        "p_values": "p=0.03"
      }}
    These values will be used in forest and funnel plots. You may adapt this format if it improves clarity or completeness, for instance if there are different important values such as t-values or different effect sizes or confidence intervalls please include them , but if you do so, please document the change in GlobalReflection so future iterations know which format is being used and why.)
    SampleSize:
    StatisticalSignificance: yes / no / not reported
    
    Summary:
    (3–5 sentence summary of study’s goal, methods, and conclusions)
    
    Findings:
    (Summarize the most important experimental findings, clearly and concisely)
    
    AddToFinalSet: yes / no
    
    Justification:
    (2–4 sentences explaining inclusion/exclusion decision, especially in relation to existing included work)
    
    GlobalReflection:
    (This is an evolving, cumulative reflection across all included papers. You should pass on the reflection you received and add on to it when you consider this useful for the review process. Use this field to pass on learned generalizations, standardizations, or coding heuristics to future iterations — e.g., how to interpret recurring task types, model labels, or inclusion patterns. It should provide guidance, not summary — like a memory thread between reasoning agents. Do not use it to describe particular studies but include only knowledge that you consider important in the review process for following papers. )
    """

#def call_gpt(prompt, model="gpt-4o", debug=False):
def call_gpt(prompt, model="gpt-4o", debug=False):
    if debug:
        print("🧪 [DEBUG MODE] Loading mock response...")
        mock_path = BASE_DIR / "data" / "debugging" / "mock_response.txt"
        if mock_path.exists():
            with open(mock_path, encoding="utf-8") as f:
                return f.read()
        else:
            raise FileNotFoundError("Mock response file not found at data/debugging/mock_response.txt")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant helping build a systematic review dataset for meta-analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def parse_gpt_response(response):
    result = {col: "" for col in CSV_COLUMNS}
    current_key = None
    lines = response.splitlines()

    for line in lines:
        if ":" in line and any(line.lower().startswith(col.lower() + ":") for col in CSV_COLUMNS):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            for col in CSV_COLUMNS:
                if key.lower() == col.lower():
                    current_key = col
                    result[col] = val
                    break
        elif current_key:
            result[current_key] += "" + line.strip()

    result["ID"] = str(uuid.uuid4())[:8]
    return result

def append_to_csv(data, path):
    file_exists = path.exists()
    with open(path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def save_json(data, pdf_path):
    output_path = json_dir / f"{pdf_path.stem}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_pdf_names_from_screened_csv():
    if not input_screened_csv.exists():
        return set()
    df = pd.read_csv(input_screened_csv, sep=";")
    return set(df[df["Include"].str.lower() == "yes"]["PDF"].str.lower())

def get_processed_pdfs(json_dir):
    if not json_dir.exists():
        return set()
    return set(f.stem.lower() for f in json_dir.glob("*.json"))

def process_next_unreviewed_pdf(debug=False):
    processed_pdfs = get_processed_pdfs(json_dir)
    included_pdfs = get_pdf_names_from_screened_csv()
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        if pdf_path.name.lower() in included_pdfs and pdf_path.stem.lower() not in processed_pdfs:
            print(f"🔍 Processing {pdf_path.name}")
            text = extract_text(pdf_path)
            previous_studies = summarize_prior_studies(csv_path)
            reflection = load_global_reflection(csv_path)
            prompt = create_prompt(previous_studies, reflection, text)
            gpt_response = call_gpt(prompt, debug=debug)
            parsed = parse_gpt_response(gpt_response)
            append_to_csv(parsed, csv_path)
            static_prompt = create_prompt(previous_studies, reflection, "")
            save_json({"pdf": pdf_path.name,"prompt":static_prompt, "response": gpt_response, "parsed": parsed}, pdf_path)
            print(f"✅ Finished: {parsed['Title']}")

    else:
        print("🎉 All eligible papers in data/articles/ have been processed.")

if __name__ == "__main__":
    # Toggle debug mode here
    DEBUG_MODE = False
    process_next_unreviewed_pdf(debug=DEBUG_MODE)
