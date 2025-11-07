# 🧠 RAG-Coder: A Framework for Augmenting Qualitative Analysis

## 📘 Description

**RAG-Coder** is a Python-based framework for semi-automating the **qualitative analysis of open-ended survey data** using **Retrieval-Augmented Generation (RAG)** strategies with the **Google Gemini API**.

It applies a formal **codebook** to new textual data, assisting researchers in coding consistency, scalability, and reproducibility.  
The framework ensures **auditability** by generating detailed logs (audit trail, model outputs, and error reports) that support validation and transparency in empirical research.

This framework was developed as part of the paper:

> *“RAG-Coder: A Framework for Augmenting Qualitative Analysis in Empirical Software Engineering”*  
> by Lidiany Cerqueira and Renan Guerra, 2025.

---

## 🧩 Project Structure

```
rag_coder/
│
├── codebook.csv              # Portuguese version of the codebook
├── codebook_en.csv           # English version of the codebook
├── config.json               # Configuration file (input paths, model params)
├── rag_coder.py              # Main RAG-Coder framework script
├── study1.csv                # Example dataset 1
├── study2.csv                # Example dataset 2
│
evaluation/
│
├── rq1_gold_standard_eval.py # Script for evaluating gold standard agreement
├── rq2_ragcoder_agreement.py # Script for RQ2: human–AI agreement analysis
│
├── requirements.txt           # Python dependencies
```

---

## 🚀 Usage

1. **Configure environment**

   Make sure you have a valid `config.json` in the same directory as `rag_coder.py`.

2. **Prepare input files**

   Ensure your input CSV files (e.g., `codebook.csv`, `study1.csv`) follow the schema defined in `config.json`.

3. **Set your API key**

   ```bash
   (Windows)   $env:GOOGLE_API_KEY="your_key"
   (macOS/Linux) export GOOGLE_API_KEY="your_key"
   ```

4. **Run RAG-Coder**

   ```bash
   python rag_coder.py
   ```

5. **View logs and outputs**

   Logs and model outputs will be stored automatically for reproducibility and later auditing.

---

## 🧱 Requirements

Install dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- pandas
- google-generativeai
- tqdm

---

## 📊 Evaluation Scripts

The `evaluation` folder contains scripts to reproduce key analyses from the paper:
- `rq1_gold_standard_eval.py`: Quantitative metrics vs. human gold standard
- `rq2_ragcoder_agreement.py`: Agreement and effort reduction analysis

---

## 🧾 License

**MIT License**

```
Copyright (c) 2025 Lidiany Cerqueira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions.
```

---

## 🌐 Citation

If you use RAG-Coder in your research, please cite:

```
Cerqueira, L. (2025). RAG-Coder: A Framework for Augmenting Qualitative Analysis in Empirical Software Engineering.
```

---

## Contact
For questions or additional information, please contact:

Lidiany Cerqueira
lidianycs@gmail.com
