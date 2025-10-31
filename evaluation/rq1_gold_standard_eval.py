# filename: rq1_gold_standard_eval.py
# -----------------------------------------------------------
#* **Author:** Lidiany Cerqueira
#* **Date:** October 31, 2025
#* **Version:** 1.0.0
#
# RQ1: Agreement of RAG-Coder with the human Gold Standard
# Metrics: Cohen's kappa, macro Precision/Recall/F1, Accuracy
# Input: adjudication_varinha.csv
# Columns expected:
#   id, response_id, coderA, coderB, consensus, adjudication
# Semantics:
#   coderA = R1 (human), coderB = RAG
#   consensus = 1 means coderA == coderB; 0 means divergence
#   adjudication: 'A' -> coderA is correct (gold)
#                 'B' -> coderB is correct (gold)
#                 'C' or any *other non-empty string* -> a new code chosen by adjudicator
#                 (missing/NaN -> if consensus==1, use coderA; else fallback to coderA)
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

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

def build_gold_standard(df: pd.DataFrame) -> pd.Series:
    """Derive gold-standard label from adjudication outcomes."""
    # Normalize strings
    for col in ["coderA", "coderB", "adjudication"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    def decide(row):
        adj = row.get("adjudication", "")
        adj = "" if pd.isna(adj) else str(adj).strip()

        if adj == "A":
            return row["coderA"]
        elif adj == "B":
            return row["coderB"]
        elif adj == "" or adj.lower() == "nan":
            # No adjudication recorded:
            # if consensus==1 assume both equal → use coderA; else fallback to coderA
            cons = row.get("consensus", None)
            try:
                cons = int(cons)
            except Exception:
                cons = None
            if cons == 1:
                return row["coderA"]
            else:
                return row["coderA"]
        else:
            # Any other non-empty string denotes a new code chosen by the adjudicator.
            # (If your file literally uses 'C', this will become 'C' as the gold label.)
            return adj

    return df.apply(decide, axis=1)

def main():
    parser = argparse.ArgumentParser(description="RQ1 evaluation: RAG vs. Adjudicated Gold Standard")
    parser.add_argument("--input", "-i", default="adjudication_varinha.csv",
                        help="Path to adjudication CSV (default: adjudication_varinha.csv)")
    parser.add_argument("--outdir", "-o", default="results_rq1",
                        help="Directory to save results (default: results_rq1)")
    parser.add_argument("--average", "-a", default="macro",
                        choices=["macro", "micro", "weighted"],
                        help="Averaging method for precision/recall/F1 (default: macro)")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read CSV with automatic delimiter detection (handles ',' or ';') + BOM
    df = pd.read_csv(in_path, sep=None, engine="python")
    # Clean possible BOM in first column name
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required_cols = {"coderA", "coderB", "adjudication"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Build gold labels
    df["gold_standard"] = build_gold_standard(df)

    # Prepare y_true and y_pred
    y_true = df["gold_standard"].astype(str).str.strip()
    y_pred = df["coderB"].astype(str).str.strip()

    # Compute metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=args.average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=args.average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=args.average, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Labels for confusion matrix (sorted for stable view)
    labels = sorted(pd.unique(pd.concat([y_true, y_pred], ignore_index=True)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Save metrics JSON
    metrics = {
        "cohens_kappa": kappa,
        f"precision_{args.average}": precision,
        f"recall_{args.average}": recall,
        f"f1_{args.average}": f1,
        "accuracy": acc
    }
    (outdir / "metrics_rq1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=[f"gold::{l}" for l in labels],
                            columns=[f"rag::{l}" for l in labels])
    cm_df.to_csv(outdir / "confusion_matrix_rq1.csv", encoding="utf-8", index=True)

    # Save per-class metrics (helpful for strengths/weaknesses analysis)
    cls_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    pd.DataFrame(cls_report).to_csv(outdir / "classification_report_rq1.csv", encoding="utf-8")

    # Pretty print to console
    print("=== RQ1: RAG vs. Adjudicated Gold Standard ===")
    print(f"Cohen's kappa:         {kappa:.3f}")
    print(f"Precision ({args.average}): {precision:.3f}")
    print(f"Recall    ({args.average}): {recall:.3f}")
    print(f"F1-score  ({args.average}): {f1:.3f}")
    print(f"Accuracy:               {acc*100:.2f}%")
    print("\nConfusion matrix (saved to CSV). Top-left excerpt:")
    print(cm_df.head(min(10, len(cm_df.index))))

    print(f"\nSaved:")
    print(f" - {outdir/'metrics_rq1.json'}")
    print(f" - {outdir/'confusion_matrix_rq1.csv'}")
    print(f" - {outdir/'classification_report_rq1.csv'}")

if __name__ == "__main__":
    main()
