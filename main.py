import os
import fitz
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import json
import re 

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "renamed_pdfs"
LOG_FILE = "logs/results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load prompt
def load_prompt():
    with open("prompts/prompt.txt", "r") as f:
        return f.read()
    
PROMPT_TEMPLATE = load_prompt()


def extract_text_from_pdf(pdf_path):
    pages_content = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pages_content.append(page.get_text())
    except Exception as e:
        print(f"Error: {e}")

    return "".join(pages_content)[:3000]

def call_llm(text):
    prompt = PROMPT_TEMPLATE.format(extracted_text=text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content) if content else None
    except:
        return None

def clean_filename(s):
    if not s:
        return "Unknown"
    s = re.sub(r'[^a-zA-Z0-9_ ]', '', s)
    return s.replace(" ", "_")

def generate_filename(data):
    if not data:
        return "Unknown_File.pdf"

    parts = [
        clean_filename(data.get("project")),
        clean_filename(data.get("drawing_type")),
        clean_filename(data.get("level")),
        clean_filename(data.get("revision"))
    ]

    return "_".join([p for p in parts if p and p != "Unknown"]) + ".pdf"

def process_pdfs():
    results = []

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    for file in tqdm(files):
        input_path = os.path.join(INPUT_DIR, file)

        text = extract_text_from_pdf(input_path)

        if not text.strip():
            new_name = "NO_TEXT_" + file
            os.rename(input_path, os.path.join(OUTPUT_DIR, new_name))
            continue

        data = call_llm(text)
        new_name = generate_filename(data)

        output_path = os.path.join(OUTPUT_DIR, new_name)

        # Avoid overwrite
        counter = 1
        base_name = new_name.replace(".pdf", "")
        while os.path.exists(output_path):
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{base_name}_{counter}.pdf"
            )
            counter += 1

        os.rename(input_path, output_path)

        results.append({
            "original": file,
            "new": os.path.basename(output_path),
            "extracted": data
        })

    df = pd.DataFrame(results)
    df.to_csv(LOG_FILE, index=False)

if __name__ == "__main__":
    process_pdfs()