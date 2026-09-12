import pandas as pd
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.agent import SupportAgent
from dotenv import load_dotenv
import json

load_dotenv()

def run_pipeline(input_csv: str, output_csv: str, max_samples: int = None):
    print(f"Loading {input_csv} for evaluation...")
    df = pd.read_csv(input_csv)
    
    if max_samples:
        df = df.head(max_samples)
        
    agent = SupportAgent()
    
    predictions = []
    
    print(f"Running agent on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = agent.process_query(row['customer_query'])
        predictions.append({
            'tweet_id': row.get('tweet_id', ''),
            'customer_query': row['customer_query'],
            'expected_intent': row.get('expected_intent', ''),
            'expected_auto_handle': row.get('expected_auto_handle', ''),
            'predicted_intent': result['intent'],
            'predicted_auto_handle': result['auto_handle'],
            'predicted_draft_reply': result['draft_reply'],
            'predicted_escalation_reason': result['escalation_reason'],
            'brand_reply_reference': row.get('brand_reply', '')
        })
        
    results_df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    return results_df

def compute_metrics(results_df: pd.DataFrame):
    print("\n--- Evaluation Metrics ---")
    
    # Intent Metrics
    if 'expected_intent' in results_df.columns and not results_df['expected_intent'].isna().all():
        # Clean labels to lowercase just in case
        y_true_intent = results_df['expected_intent'].str.lower()
        y_pred_intent = results_df['predicted_intent'].str.lower()
        
        acc = accuracy_score(y_true_intent, y_pred_intent)
        print(f"Intent Accuracy: {acc:.4f}")
        print("\nIntent Classification Report:")
        print(classification_report(y_true_intent, y_pred_intent, zero_division=0))
    
    # Auto Handle Metrics
    if 'expected_auto_handle' in results_df.columns and not results_df['expected_auto_handle'].isna().all():
        # Convert to boolean if needed
        y_true_auto = results_df['expected_auto_handle'].astype(bool)
        y_pred_auto = results_df['predicted_auto_handle'].astype(bool)
        
        acc_auto = accuracy_score(y_true_auto, y_pred_auto)
        f1_auto = f1_score(y_true_auto, y_pred_auto)
        print(f"Auto-Handle Accuracy: {acc_auto:.4f}")
        print(f"Auto-Handle F1 Score: {f1_auto:.4f}")

def llm_as_judge(results_df: pd.DataFrame, output_csv: str):
    """
    Evaluates the predicted_draft_reply using an LLM and saves row-by-row scores.
    Requires OPENAI_API_KEY or GROQ_API_KEY.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key and not groq_api_key:
        print("Skipping LLM-as-a-judge: Neither OPENAI_API_KEY nor GROQ_API_KEY set.")
        return results_df
        
    from openai import OpenAI
    
    if groq_api_key:
        client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        model_name = "openai/gpt-oss-20b"
    else:
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o-mini"
    
    print("\nRunning LLM-as-a-judge on top 20 examples...")
    
    # Ensure columns exist
    if 'llm_tone' not in results_df.columns:
        results_df['llm_tone'] = pd.NA
        results_df['llm_helpfulness'] = pd.NA
        results_df['llm_groundedness'] = pd.NA
        results_df['llm_reasoning'] = pd.NA
        
    total_tone = 0
    total_helpfulness = 0
    total_groundedness = 0
    evaluated_count = 0
    
    judge_prompt = """You are an expert customer support evaluator.
Given a Customer Query, the Agent's Draft Reply, and the Reference Brand Reply (how the brand actually answered), score the Draft Reply on three criteria from 1 to 5.

1. Tone (1-5): Does it sound like a polite, empathetic support agent?
2. Helpfulness (1-5): Does it actually address the customer's query?
3. Groundedness (1-5): Does it avoid hallucinating policies not found in the Reference Reply?

Output strictly in JSON:
{
  "tone": int,
  "helpfulness": int,
  "groundedness": int,
  "reasoning": "string"
}"""

    for i, row in tqdm(results_df.head(20).iterrows(), total=min(len(results_df), 20)):
        # Skip if already scored
        if pd.notna(row.get('llm_tone')):
            total_tone += row['llm_tone']
            total_helpfulness += row['llm_helpfulness']
            total_groundedness += row['llm_groundedness']
            evaluated_count += 1
            continue
            
        user_prompt = f"Customer Query: {row['customer_query']}\nReference Reply: {row['brand_reply_reference']}\nAgent Draft: {row['predicted_draft_reply']}"
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            scores = json.loads(response.choices[0].message.content)
            
            results_df.at[i, 'llm_tone'] = scores.get('tone')
            results_df.at[i, 'llm_helpfulness'] = scores.get('helpfulness')
            results_df.at[i, 'llm_groundedness'] = scores.get('groundedness')
            results_df.at[i, 'llm_reasoning'] = scores.get('reasoning')
            
            total_tone += scores.get('tone', 0)
            total_helpfulness += scores.get('helpfulness', 0)
            total_groundedness += scores.get('groundedness', 0)
            evaluated_count += 1
            
            # Save progress
            results_df.to_csv(output_csv, index=False)
            
        except Exception as e:
            print(f"Judge error: {e}")
            
    if evaluated_count > 0:
        print("\n--- LLM-as-a-judge Results (Averages over 20 samples) ---")
        print(f"Tone:         {total_tone/evaluated_count:.2f}/5")
        print(f"Helpfulness:  {total_helpfulness/evaluated_count:.2f}/5")
        print(f"Groundedness: {total_groundedness/evaluated_count:.2f}/5")
        print(f"\nSaved detailed LLM scores to {output_csv}")
        print("To compute Human Agreement (Cohen's Kappa), run `python3 -m src.llm_judge_calibration`")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/golden_eval_handlabelled.csv')
    parser.add_argument('--output', type=str, default='data/evaluation_results.csv')
    parser.add_argument('--max', type=int, default=None, help='Max rows to evaluate')
    args = parser.parse_args()
    
    results = run_pipeline(args.input, args.output, args.max)
    compute_metrics(results)
    results = llm_as_judge(results, args.output)
