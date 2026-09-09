import pandas as pd
import re
import os
import argparse

def heuristic_label(text):
    text = text.lower()
    
    # Intent
    if re.search(r'ios|update|app|crash|software|bug|glitch', text):
        intent = 'software_issue'
    elif re.search(r'battery|screen|crack|charge|hardware|button|speaker|macbook|phone', text):
        intent = 'hardware_issue'
    elif re.search(r'password|apple id|account|login|icloud', text):
        intent = 'account_issue'
    else:
        intent = 'general_inquiry'
        
    # Auto-handle vs Escalate
    # Escalate if angry or very complex
    if re.search(r'fuck|shit|angry|hate|ridiculous|worst|sucks|terrible', text):
        auto_handle = False
        escalation_reason = 'High negative sentiment / Angry customer'
    elif len(text.split()) > 40:
        auto_handle = False
        escalation_reason = 'Query is too long/complex'
    else:
        auto_handle = True
        escalation_reason = ''
        
    return intent, auto_handle, escalation_reason

def generate_golden_set(input_path: str, output_path: str, num_samples: int = 200):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Stratified/Random sampling (we use random state for reproducibility)
    sampled = df.sample(n=num_samples, random_state=42).copy()
    
    intents = []
    auto_handles = []
    escalation_reasons = []
    
    for _, row in sampled.iterrows():
        intent, auto, reason = heuristic_label(row['customer_query'])
        intents.append(intent)
        auto_handles.append(auto)
        escalation_reasons.append(reason)
        
    sampled['expected_intent'] = intents
    sampled['expected_auto_handle'] = auto_handles
    sampled['expected_escalation_reason'] = escalation_reasons
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sampled.to_csv(output_path, index=False)
    print(f"Golden set generated with {num_samples} samples at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed_conversations.csv')
    parser.add_argument('--output', type=str, default='data/golden_eval.csv')
    parser.add_argument('--n', type=int, default=200)
    args = parser.parse_args()
    
    generate_golden_set(args.input, args.output, args.n)
