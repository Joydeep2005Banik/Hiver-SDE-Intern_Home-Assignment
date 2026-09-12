"""
Interactive LLM Judge Calibration Tool.

This script lets a human reviewer score a sample of the agent's drafted replies
on Tone, Helpfulness, and Groundedness (1-5 scale). It then compares the human
scores against the LLM Judge's scores to compute agreement metrics (Cohen's Kappa).

Usage:
    # First, generate predictions and let the LLM Judge score a sample:
    python3 -m src.evaluate --input data/golden_eval_handlabelled.csv --max 30 --output data/evaluation_results_sample.csv

    # Then run this tool to do the human scoring on the same sample:
    python3 -m src.llm_judge_calibration --input data/evaluation_results_sample.csv
"""

import pandas as pd
import json
import os
import argparse
from sklearn.metrics import cohen_kappa_score

def llm_judge_calibration(input_csv: str):
    print("=" * 60)
    print("  LLM JUDGE CALIBRATION & HUMAN AGREEMENT TOOL")
    print("=" * 60)
    
    df = pd.read_csv(input_csv)
    
    # Check if the LLM judge has already scored these. 
    # The evaluation_results.csv from evaluate.py doesn't currently save the LLM scores row-by-row,
    # it just prints the averages.
    
    print("\n[NOTE] We need to run the LLM judge and save row-by-row scores first, but for the sake of the exercise, let's collect the human scores.")
    
    if 'human_tone' not in df.columns:
        df['human_tone'] = pd.NA
        df['human_helpfulness'] = pd.NA
        df['human_groundedness'] = pd.NA

    total = min(len(df), 20) # Score 20 examples for a solid kappa metric
    
    # Find start idx
    start_idx = 0
    if df['human_tone'].notna().sum() > 0:
        start_idx = df['human_tone'].notna().sum()
        
    if start_idx >= total:
        print(f"\nYou have already scored {total} examples. Run the analysis notebook to see Kappa!")
        return

    print(f"\nYou will review {total - start_idx} drafted replies.")
    print("Score each on a scale of 1 to 5:")
    print("  Tone:         Polite, empathetic, sounds like a support agent?")
    print("  Helpfulness:  Addresses the query, offers a solution/next step?")
    print("  Groundedness: Avoids hallucinating policies not in the reference reply?")
    print("\n  Type 'q' to quit and save progress.")

    for i in range(start_idx, total):
        row = df.iloc[i]
        
        print(f"\n{'─' * 60}")
        print(f"  Example {i + 1}/{total}")
        print(f"{'─' * 60}")
        print(f"\n  Customer Query:")
        print(f"  \"{row['customer_query']}\"")
        print(f"\n  Reference Brand Reply:")
        print(f"  \"{str(row['brand_reply_reference'])}\"")
        print(f"\n  Agent's Draft Reply:")
        print(f"  \"{str(row['predicted_draft_reply'])}\"")
        
        try:
            tone = _get_score("Tone (1-5)")
            if tone is None: break
            helpfulness = _get_score("Helpfulness (1-5)")
            if helpfulness is None: break
            groundedness = _get_score("Groundedness (1-5)")
            if groundedness is None: break
            
            df.at[df.index[i], 'human_tone'] = tone
            df.at[df.index[i], 'human_helpfulness'] = helpfulness
            df.at[df.index[i], 'human_groundedness'] = groundedness
            
            # Save every time
            df.to_csv(input_csv, index=False)
            
        except KeyboardInterrupt:
            break
            
    print("\nSaved human scores to:", input_csv)
    
    # Calculate Cohen's Kappa
    df = pd.read_csv(input_csv) # reload
    if 'human_tone' in df.columns and 'llm_tone' in df.columns:
        if df['human_tone'].notna().sum() > 0 and df['llm_tone'].notna().sum() > 0:
            scored = df[df['human_tone'].notna() & df['llm_tone'].notna()].copy()
            if len(scored) > 0:
                # Convert to int for sklearn
                for col in ['human_tone', 'human_helpfulness', 'human_groundedness', 'llm_tone', 'llm_helpfulness', 'llm_groundedness']:
                    scored[col] = scored[col].astype(int)
                    
                kappa_tone = cohen_kappa_score(scored['human_tone'], scored['llm_tone'])
                kappa_help = cohen_kappa_score(scored['human_helpfulness'], scored['llm_helpfulness'])
                kappa_ground = cohen_kappa_score(scored['human_groundedness'], scored['llm_groundedness'])
                
                print(f"\n{'=' * 60}")
                print(f"  HUMAN-LLM AGREEMENT (COHEN'S KAPPA)")
                print(f"  (Evaluated on {len(scored)} examples)")
                print(f"{'=' * 60}")
                print(f"  Tone:         {kappa_tone:.3f}")
                print(f"  Helpfulness:  {kappa_help:.3f}")
                print(f"  Groundedness: {kappa_ground:.3f}")
                print(f"\n  Kappa Interpretation Guide:")
                print(f"  < 0: No agreement | 0.0-0.2: Slight | 0.2-0.4: Fair")
                print(f"  0.4-0.6: Moderate | 0.6-0.8: Substantial | 0.8-1.0: Perfect")
                print(f"{'=' * 60}")
            else:
                print("\n  No overlapping human + LLM scores found to compute Kappa.")
        else:
            if 'llm_tone' not in df.columns or df['llm_tone'].notna().sum() == 0:
                print("\n  LLM Judge scores not found. Run evaluate.py first:")
                print("  python3 -m src.evaluate --max 20 --output data/evaluation_results_sample.csv")
    elif 'human_tone' in df.columns:
        print(f"\n  Human scores saved ({df['human_tone'].notna().sum()} scored).")
        print("  LLM Judge scores not found. Run evaluate.py first to generate them:")
        print("  python3 -m src.evaluate --max 20 --output data/evaluation_results_sample.csv")
        print("  Then re-run this tool.")
    else:
        print("\n  No scores found in the file.")

def _get_score(prompt):
    while True:
        val = input(f"  {prompt}: ").strip().lower()
        if val == 'q': return None
        if val in ('1','2','3','4','5'):
            return int(val)
        print("  Please enter a number from 1 to 5, or 'q' to quit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/evaluation_results_sample.csv')
    args = parser.parse_args()
    
    llm_judge_calibration(args.input)
