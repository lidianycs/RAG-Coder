# filename: rq2_ragcoder_agreement.py
# -----------------------------------------------------------
# **Author:** Lidiany Cerqueira
# **Date:** October 31, 2025
# **Version:** 1.0.0
# Calculate Cohen’s kappa and percentage of consensus
# between RAG-Coder and a human coder (R1)
# -----------------------------------------------------------
"""
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

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# === Load dataset ===
# The CSV must have columns: id;response_id;human_label;ragcoder_label;consensus
file_path = "dataset_varinha_consenso.csv"
df = pd.read_csv(file_path, sep=';')

# === Clean and prepare data ===
df['human_label'] = df['human_label'].astype(str).str.strip()
df['ragcoder_label'] = df['ragcoder_label'].astype(str).str.strip()

# === Compute Cohen’s kappa ===
kappa = cohen_kappa_score(df['human_label'], df['ragcoder_label'])

# === Compute % of responses not requiring adjudication ===
percent_consensus = df['consensus'].mean() * 100

# === Display results ===
print("=== RAG-Coder Evaluation Metrics ===")
print(f"Cohen’s κ (RAG vs. Human): {kappa:.3f}")
print(f"% of responses not requiring adjudication: {percent_consensus:.2f}%")

# Optional: interpret kappa level (Landis & Koch, 1977)
if kappa < 0.0:
    level = "Poor"
elif kappa < 0.20:
    level = "Slight"
elif kappa < 0.40:
    level = "Fair"
elif kappa < 0.60:
    level = "Moderate"
elif kappa < 0.80:
    level = "Substantial"
else:
    level = "Almost perfect"

print(f"Interpretation: {level} agreement")
