from openai import OpenAI
import fitz  # PyMuPDF
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import argparse
import sys
import hashlib

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

def get_mock_gpt_response():
    with open("../data/debugging/mock_gpt_text.txt", "r", encoding="utf-8") as f:
        content = f.read()

    return content

def extract_text_from_pdf(pdf_path, max_chars=3000):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    return text.strip()

def screen_text_with_gpt(text, model="gpt-3.5-turbo"):
    system_prompt = (
        "You are assisting with a systematic review titled "
        "'Theory of Mind in Large Language Models: A Systematic Review of Evaluation Paradigms and Cognitive Claims'.\n\n"
        "Your task is to assess whether a given scientific paper is suitable for inclusion in this review. "
        "The review only includes *empirical studies* that evaluate Theory of Mind (ToM) abilities in Large Language Models (LLMs).\n\n"
        "To be included, the study must:\n"
        "- Be an empirical paper published in a journal, conference, or on a preprint server (e.g., arXiv)\n"
        "- Evaluate general-purpose LLMs (e.g., GPT-3.5, GPT-4, PaLM), not only fine-tuned or specialized models\n"
        "- Use clearly defined ToM tasks (e.g., false-belief, recursive belief modeling, second-order inference)\n"
        "- Provide quantitative performance outcomes (e.g., accuracy, pass/fail rate)\n"
        "- Go beyond purely narrative comprehension or vague social reasoning claims\n"
        "- Not be a thesis (e.g., bachelor’s or master’s thesis) or unpublished coursework\n"
        "- Not be purely conceptual, benchmark-proposing, or theoretical\n\n"
        " Important: You may only see part of the paper (e.g., abstract or early sections). "
        "If the paper clearly describes an empirical setup and is likely to include quantitative results later, mark it as 'Include: yes'. "
        "From the given text (typically an abstract or full-text excerpt), extract the following fields:\n"
        "- Title: e.g., Emergent Theory of Mind in Large Language Models\n"
        "- Authors: e.g., Terentev et al.\n"
        "- Include: write 'yes' if the study meets all the criteria or likely does; write 'no' only if clearly not suitable\n"
        "- Reason: short explanation (only if excluded; e.g., no LLMs tested, vague ToM concept, thesis, no results)\n"
        "- Task type: e.g., false belief, recursive belief modeling, narrative inference, interaction-based\n"
        "- Model type: e.g., GPT-3.5, GPT-4, PaLM, custom fine-tuned model\n"
        "- Notes: any additional relevant details (e.g., task complexity, human baseline, prompt sensitivity)\n\n"
        "Return only clean, labeled lines for each field. If a field is not available, write 'Not reported'. "
        "Always include ':' between the field name and the value (e.g., 'Title: Example Title')."

    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:6000]}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def parse_response(raw):
    result = {
        "Title": "",
        "Authors": "",
        "Include": "",
        "Reason": "",
        "TaskType": "",
        "ModelType": "",
        "Notes": ""
    }
    lines = raw.strip().split("\n")
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip().lower()
            val = val.strip()
            if "title" in key:
                result["Title"] = val
            elif "author" in key:
                result["Authors"] = val
            elif "include" in key:
                result["Include"] = val
            elif "reason" in key:
                result["Reason"] = val
            elif "task" in key:
                result["TaskType"] = val
            elif "model" in key:
                result["ModelType"] = val
            elif "notes" in key:
                result["Notes"] = val
    return result

def generate_id(pdf_name):
    # You can also hash the filename or use a UUID if preferred
    return hashlib.md5(pdf_name.encode()).hexdigest()[:8]

def main(pdf_dir, output_file, model="gpt-3.5-turbo"):
    BASE_DIR = Path(__file__).resolve().parent.parent
    pdf_dir = (BASE_DIR / pdf_dir).resolve()
    output_file = (BASE_DIR / output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(columns=["ID", "PDF", "Title", "Authors", "Include", "Reason", "TaskType", "ModelType", "Notes"])

    for pdf in pdf_dir.glob("*.pdf"):
        print(f"Processing: {pdf.name}")
        try:
            text = extract_text_from_pdf(pdf)
            gpt_response = screen_text_with_gpt(text, model)
            #gpt_response = get_mock_gpt_response()
            parsed = parse_response(gpt_response)
            parsed["PDF"] = pdf.name
            parsed["ID"] = generate_id(pdf.name)
            df = pd.concat([df, pd.DataFrame([parsed])], ignore_index=True)
        except Exception as e:
            print(f"Error processing {pdf.name}: {e}")

    df.to_csv(output_file, index=False)
    print(f" Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./data/articles", help="Directory containing PDFs")
    parser.add_argument("--output", default="./data/csv/reviewed_papers.csv", help="Output CSV file")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    args = parser.parse_args()
    main(args.pdf_dir, args.output, args.model)
