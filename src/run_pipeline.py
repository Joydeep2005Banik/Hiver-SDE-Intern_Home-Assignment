import argparse
from src.evaluate import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Support Agent pipeline on a CSV dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV containing customer queries.")
    parser.add_argument('--output', type=str, default="data/predictions.csv", help="Path to output predictions CSV.")
    parser.add_argument('--max', type=int, default=None, help="Max rows to process (for quick testing).")
    
    args = parser.parse_args()
    
    print(f"Starting pipeline. Input: {args.input}, Output: {args.output}")
    run_pipeline(args.input, args.output, args.max)
    print("Pipeline finished successfully.")
