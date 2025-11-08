"""
================================================================================
RAG-Coder: A Framework for Augmenting Qualitative Analysis in ESE
================================================================================

* **File:** rag_coder.py
* **Author:** Lidiany Cerqueira
* **Date:** October 31, 2025
* **Version:** 1.0.1

---
**Description**

This script implements the RAG-Coder framework, a Python-based tool for 
semi-automating the qualitative analysis of open-ended survey data. It uses 
a Retrieval-Augmented Generation (RAG) strategy with the Google Gemini API 
to apply a formal codebook to new, unseen textual data.

This framework was developed for the paper:
"RAG-Coder: A Framework for Augmenting Qualitative Analysis in Empirical 
Software Engineering"

The system is designed to be auditable and transparent, generating
comprehensive logs (audit trail, model outputs, errors) to ensure
research validity and reproducibility.

---
**Usage**

1.  Ensure you have a 'config.json' file in the same directory.
2.  Populate your input CSV files (codebook.csv, etc.) as defined in 'config.json'.
3.  Set your Google API Key as an environment variable:
    (Windows):      set GOOGLE_API_KEY="your_key"
    (macOS/Linux):  export GOOGLE_API_KEY="your_key"
4.  Run the script:
    python rag_coder.py

**Requirements:**
- pandas
- google-generativeai
- tqdm

---
**License**

MIT License

Copyright (c) 2025 Lidiany Cerqueira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================================================================
"""


import os
import pandas as pd
import google.generativeai as genai
import json
import re
import time
import sys
from tqdm import tqdm

# --- Configuration Loader ---

def load_config(filename="config.json"):
    """Loads the configuration from a JSON file."""
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Configuration file '{filename}' contains invalid JSON.")
        sys.exit(1)

# --- API Setup ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

# --- Helper Functions ---

def load_data(filepath, delimiter=';'):
    """Loads a CSV file into a pandas DataFrame."""
    try: return pd.read_csv(filepath, delimiter=delimiter)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found."); return None

def format_codebook(df):
    """Formats the codebook DataFrame into a readable string for the prompt."""
    lines = []
    for _, row in df.iterrows():
        label = f"{row['category']}-{row['factor']}"
        lines.append(f"- Label: `{label}`\n  Description: {row['description']}")
    return "\n".join(lines)

def format_examples(df1, df2):
    """Formats the example studies into a readable string for the prompt."""
    examples_df = pd.concat([df1, df2], ignore_index=True)
    lines = []
    for _, row in examples_df.iterrows():
        lines.append(f"Response: \"{row['response_text']}\"")
        lines.append(f"Correct Label: `{row['label']}`")
        lines.append("-" * 10)
    return "\n".join(lines)

def create_prompt(codebook_str, examples_str, new_response):
    """Creates the full prompt with instructions, codebook, examples, and the new response."""
    return f"""
    You are a meticulous qualitative coding assistant for an academic study. Your job is to assign one or more labels from a given codebook to a user survey response.

    Follow these constraints strictly:
    1.  **Output Format**: You MUST output a valid JSON list `[...]`.
    2.  **Allowed Labels**: You may ONLY choose labels from the "ALLOWED LABELS (CODEBOOK)" section. Do not invent new labels.
    3.  **Evidence**: Use "span_evidence" to quote the shortest possible, direct span of text.
    4.  **Multiple Codes**: If a response contains multiple distinct ideas, create a separate JSON object for each.
    5.  **Ambiguity**: If the best label is not obvious, choose the closest match, set "ambiguous": true, and write a short note in "rationale".
    6.  **No Code (NC)**: If no label applies reasonably, return an empty list `[]`.
    7.  **Empty Answer (NA)**: If the response is empty or whitespace, return the string "NA".

    --- ALLOWED LABELS (CODEBOOK) ---
    {codebook_str}

    --- EXAMPLES OF CORRECT CODING ---
    {examples_str}

    --- NEW RESPONSE TO CODE ---
    Response: "{new_response}"

    --- YOUR JSON OUTPUT ---
    IMPORTANT: Your entire response must be ONLY the JSON list...
    """

def clean_json_output(raw_output):
    """Cleans the model's raw output to extract a valid JSON string."""
    match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output)
    if match: return match.group(1).strip()
    return raw_output.strip()


def main():
    
    config = load_config()

   
    model = genai.GenerativeModel(
        model_name=config["api_settings"]["model_name"],
        generation_config=config["generation_config"]
    )
    
    start_time = time.time()
    print("Loading data files...")
    
   
    codebook_df = load_data(config["input_files"]["codebook_file"])
    study1_df = load_data(config["input_files"]["example_file_1"])
    study2_df = load_data(config["input_files"]["example_file_2"])
    new_responses_df = load_data(config["input_files"]["input_data_file"])
    
    if any(df is None for df in [codebook_df, study1_df, study2_df, new_responses_df]): return

    print("Formatting codebook and examples...")
    codebook_str = format_codebook(codebook_df)
    examples_str = format_examples(study1_df, study2_df)
    
    raw_results, audit_log, error_log, model_output_log = [], [], [], []

    print(f"Starting to code {len(new_responses_df)} new responses...")
    for index, row in tqdm(new_responses_df.iterrows(), total=new_responses_df.shape[0]):
        response_id, response_text = row['response_id'], str(row['response_text']).strip()
        
        response_text = response_text.replace('"', '\\"')

        if not response_text:
            raw_results.append({'response_id': response_id, 'response_text': '', 'coded_output': 'NA'})
            continue
            
        prompt = create_prompt(codebook_str, examples_str, response_text)
        audit_log.append({'response_id': response_id, 'prompt_text': prompt})

        try:
            response = model.generate_content(prompt, safety_settings=config["safety_settings"])
            
            cleaned_output = clean_json_output(response.text)
            try:
                coded_data = json.loads(cleaned_output)
            except json.JSONDecodeError:
                error_log.append(f"ID: {response_id}, JSON Decode Error, Raw Output: {cleaned_output}")
                coded_data = [{"error": "JSON Decode Error"}]
            
            model_output_log.append({'response_id': response_id, 'coded_output': coded_data})
            raw_results.append({'response_id': response_id, 'response_text': response_text, 'coded_output': coded_data})

        except Exception as e:
            error_log.append(f"ID: {response_id}, API Exception: {str(e)}")
            raw_results.append({'response_id': response_id, 'response_text': response_text, 'coded_output': [{"error": str(e)}]})
        finally:
            time.sleep(config["api_settings"]["seconds_to_wait"])

    print("\nProcessing and flattening results...")
    flattened_results = []
    for record in raw_results:
        response_id, response_text, coded_output = record['response_id'], record['response_text'], record['coded_output']
        if pd.isna(response_text) or response_text.lower() == 'nan': response_text = ''
        if isinstance(coded_output, list) and coded_output:
            for item in coded_output:
                label = "ERROR" if isinstance(item, dict) and 'error' in item else item.get('label', 'NC') if isinstance(item, dict) else 'MALFORMED'
                flattened_results.append({'response_id': response_id, 'response_text': response_text, 'label': label})
        elif isinstance(coded_output, list):
             flattened_results.append({'response_id': response_id, 'response_text': response_text, 'label': 'NC'})
        else:
            label = 'NA' if coded_output == 'NA' else 'NC'
            flattened_results.append({'response_id': response_id, 'response_text': response_text, 'label': label})

    
    output_df = pd.DataFrame(flattened_results)
    output_df.insert(0, 'id', range(1, 1 + len(output_df)))
    output_df[['id', 'response_id', 'response_text', 'label']].to_csv(config["output_files"]["results_file"], index=False, sep=';', encoding='utf-8-sig')
    
    with open(config["output_files"]["audit_file"], 'w', encoding='utf-8') as f: json.dump(audit_log, f, indent=4, ensure_ascii=False)
    with open(config["output_files"]["model_log_file"], 'w', encoding='utf-8') as f: json.dump(model_output_log, f, indent=4, ensure_ascii=False)
    with open(config["output_files"]["error_file"], 'w', encoding='utf-8') as f: f.write("\n".join(error_log))

    print(f"\n--- Script Finished ---")
    print(f"✅ Results saved to: {config['output_files']['results_file']}")
    print(f"✅ Audit trail saved to: {config['output_files']['audit_file']}")
    print(f"✅ Model outputs saved to: {config['output_files']['model_log_file']}")
    if error_log: print(f"⚠️ Errors were logged. Please check: {config['output_files']['error_file']}")
    else: print("✅ No errors were logged.")
    
    duration = time.time() - start_time
    print(f"Total execution time: {duration:.2f} seconds.")

if __name__ == "__main__":
    main()