import pandas as pd
import re
import os
import argparse
from src.intent_discovery import classify_by_heuristic


def heuristic_label(text):
    """Assign intent + auto-handle/escalate labels using data-derived rules.
    
    Intent classification uses the taxonomy discovered via keyword-frequency
    analysis of 106k+ AppleSupport conversations (see src/intent_discovery.py).
    
    Escalation heuristic flags highly negative sentiment or overly complex
    (long) queries.
    """
    intent = classify_by_heuristic(text)

    text_lower = text.lower()

    # Auto-handle vs Escalate
    if re.search(r'fuck|shit|angry|hate|ridiculous|worst|sucks|terrible', text_lower):
        auto_handle = False
        escalation_reason = 'High negative sentiment / Angry customer'
    elif len(text_lower.split()) > 40:
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
