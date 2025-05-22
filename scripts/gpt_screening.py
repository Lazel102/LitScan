import openai
import fitz  # PyMuPDF
import os
import csv
from pathlib import Path

openai.api_key_path = ".env"

def extract_text_from_pdf(pdf_path, max_chars=3000):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    return text.strip()

def screen_text_with_gpt(text, model="gpt-4"):
    system_prompt = (
        "You are assisting with a systematic review titled "
        "'Theory of Mind in Large Language Models: A Systematic Review of Evaluation Paradigms and Cognitive Claims'.\n\n"
        "Your task is to assess whether a given scientific paper is suitable for inclusion in this review. "
        "The review only includes *empirical studies* that evaluate Theory of Mind (ToM) abilities in Large Language Models (LLMs).\n\n"
        "From the given text (typically an abstract or full-text excerpt), extract the following fields:\n"
        "- Include: Yes or No\n"
        "- Reason (if No): short explanation (e.g., theoretical only, no LLM, no ToM task)\n"
        "- Task type: e.g., false belief, recursive belief modeling, narrative inference, or interaction-based\n"
        "- Model type: e.g., GPT-3.5, GPT-4, PaLM, custom fine-tuned model\n"
        "- Performance metric: what was measured (e.g., accuracy, pass/fail rate)\n"
        "- P-value or statistical significance (if mentioned)\n"
        "- Notes: any additional relevant details (e.g., task complexity, prompt sensitivity, human comparison)\n\n"
        "Return only clean, labeled lines for each field. If a field is not available, write 'Not reported'."
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:3000]}  # safety cut
        ],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

def parse_response(raw):
    result = {"Include": "", "Reason": "", "TaskType": "", "ModelType": "", "Notes": ""}
    lines = raw.split("\n")
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            key, val = key.strip(), val.strip()
            if key.lower().startswith("include"):
                result["Include"] = val
            elif key.lower().startswith("reason"):
                result["Reason"] = val
            elif key.lower().startswith("task"):
                result["TaskType"] = val
            elif key.lower().startswith("model"):
                result["ModelType"] = val
            elif key.lower().startswith("notes"):
                result["Notes"] = val
    return result

def main(pdf_dir, output_file, model="gpt-4"):
    pdf_dir = Path(pdf_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as out_csv:
        fieldnames = ["PDF", "Include", "Reason", "TaskType", "ModelType", "Notes"]
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()

        for pdf in pdf_dir.glob("*.pdf"):
            print(f"Processing: {pdf.name}")
            try:
                text = extract_text_from_pdf(pdf)
                gpt_response = screen_text_with_gpt(text, model)
                parsed = parse_response(gpt_response)
                parsed["PDF"] = pdf.name
                writer.writerow(parsed)
            except Exception as e:
                print(f"Error processing {pdf.name}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="articles", help="Directory containing PDFs")
    parser.add_argument("--output", default="data/csv/reviewed_papers.csv", help="Output CSV file")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    args = parser.parse_args()
    main(args.pdf_dir, args.output, args.model)