"""
Interactive Hand-Labelling Tool for the Golden Evaluation Set.

This tool walks through each example in the golden set, shows the customer
query and the heuristic-suggested labels, and lets the human reviewer
confirm or correct them.

Features:
  - Pre-fills with heuristic suggestions (press Enter to accept)
  - Saves progress after every entry (resume-safe)
  - Tracks confirmed vs. corrected counts for the methodology note
  - Produces a final CSV with a 'labelled_by' column = 'human'

Usage:
    python3 -m src.label_golden_set
    python3 -m src.label_golden_set --input data/golden_eval.csv --output data/golden_eval_handlabelled.csv
"""

import pandas as pd
import os
import argparse
from src.intent_discovery import INTENT_NAMES


def display_intents():
    """Print the numbered intent menu."""
    print("\n  Intent options:")
    for i, name in enumerate(INTENT_NAMES, 1):
        print(f"    {i}. {name}")
    print()


def label_golden_set(input_csv: str, output_csv: str):
    print("=" * 60)
    print("  GOLDEN SET HAND-LABELLING TOOL")
    print("=" * 60)
    print(f"\nInput:  {input_csv}")
    print(f"Output: {output_csv}")

    df = pd.read_csv(input_csv)
    total = len(df)

    # ----- Resume support -----
    # If the output file already exists, load it to find where we left off
    start_idx = 0
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        labelled_count = existing['labelled_by'].notna().sum() if 'labelled_by' in existing.columns else 0
        if labelled_count > 0 and labelled_count < total:
            print(f"\nFound {labelled_count}/{total} already labelled. Resuming from #{labelled_count + 1}.")
            df = existing.copy()
            start_idx = labelled_count
        elif labelled_count >= total:
            print(f"\nAll {total} examples already labelled!")
            _print_stats(df)
            return
    else:
        # Add tracking columns
        df['labelled_by'] = pd.NA
        df['human_corrected'] = pd.NA

    print(f"\nYou will review {total - start_idx} examples.")
    print("For each example:")
    print("  - Press ENTER to accept the suggested label")
    print("  - Type a number to pick a different intent")
    print("  - Type 'y' or 'n' for auto-handle (or ENTER to accept)")
    print("  - Type 'q' to save and quit (you can resume later)")
    print("  - Type 's' to skip an example")
    display_intents()

    confirmed = 0
    corrected = 0

    for i in range(start_idx, total):
        row = df.iloc[i]
        suggested_intent = row['expected_intent']
        suggested_auto = row['expected_auto_handle']

        print(f"\n{'─' * 60}")
        print(f"  Example {i + 1}/{total}")
        print(f"{'─' * 60}")
        print(f"\n  Customer Query:")
        print(f"  \"{row['customer_query'][:300]}\"")
        print(f"\n  Brand Reply (reference):")
        print(f"  \"{str(row['brand_reply'])[:300]}\"")
        print(f"\n  Suggested Intent:      {suggested_intent}")
        print(f"  Suggested Auto-Handle: {suggested_auto}")

        # ----- Intent -----
        intent_input = input(f"\n  Intent [{suggested_intent}] (1-{len(INTENT_NAMES)}/Enter/q/s): ").strip().lower()

        if intent_input == 'q':
            print("\nSaving progress...")
            _save(df, output_csv)
            print(f"Progress saved. {i - start_idx} examples labelled in this session.")
            _print_stats(df)
            return
        if intent_input == 's':
            print("  Skipped.")
            continue

        if intent_input == '':
            final_intent = suggested_intent
        elif intent_input.isdigit() and 1 <= int(intent_input) <= len(INTENT_NAMES):
            final_intent = INTENT_NAMES[int(intent_input) - 1]
        else:
            # Try to match by name
            if intent_input in INTENT_NAMES:
                final_intent = intent_input
            else:
                print(f"  Invalid input '{intent_input}', keeping suggestion.")
                final_intent = suggested_intent

        # ----- Auto-Handle -----
        auto_input = input(f"  Auto-Handle [{suggested_auto}] (y/n/Enter): ").strip().lower()

        if auto_input == 'q':
            print("\nSaving progress...")
            _save(df, output_csv)
            return
        if auto_input == '':
            final_auto = suggested_auto
        elif auto_input in ('y', 'yes', 'true'):
            final_auto = True
        elif auto_input in ('n', 'no', 'false'):
            final_auto = False
        else:
            final_auto = suggested_auto

        # ----- Escalation reason (only if auto_handle is False) -----
        final_reason = ''
        if not final_auto:
            existing_reason = str(row.get('expected_escalation_reason', ''))
            reason_input = input(f"  Escalation Reason [{existing_reason}]: ").strip()
            final_reason = reason_input if reason_input else existing_reason

        # ----- Record -----
        was_corrected = (final_intent != suggested_intent) or (str(final_auto) != str(suggested_auto))

        df.at[df.index[i], 'expected_intent'] = final_intent
        df.at[df.index[i], 'expected_auto_handle'] = final_auto
        df.at[df.index[i], 'expected_escalation_reason'] = final_reason
        df.at[df.index[i], 'labelled_by'] = 'human'
        df.at[df.index[i], 'human_corrected'] = was_corrected

        if was_corrected:
            corrected += 1
            print(f"  ✏️  Corrected → intent={final_intent}, auto_handle={final_auto}")
        else:
            confirmed += 1
            print(f"  ✓  Confirmed → intent={final_intent}, auto_handle={final_auto}")

        # Auto-save every 10 examples
        if (i + 1) % 10 == 0:
            _save(df, output_csv)
            print(f"\n  [Auto-saved progress at {i + 1}/{total}]")

    # Final save
    _save(df, output_csv)
    print(f"\n{'=' * 60}")
    print(f"  LABELLING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\n  Session: {confirmed} confirmed, {corrected} corrected")
    _print_stats(df)


def _save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _print_stats(df: pd.DataFrame):
    """Print overall labelling statistics."""
    if 'labelled_by' not in df.columns:
        return

    total = len(df)
    labelled = df['labelled_by'].notna().sum()
    corrected = df['human_corrected'].sum() if 'human_corrected' in df.columns else 0

    print(f"\n  Overall stats:")
    print(f"    Total examples:        {total}")
    print(f"    Human-labelled:        {labelled}")
    print(f"    Remaining:             {total - labelled}")
    print(f"    Heuristic confirmed:   {labelled - corrected}")
    print(f"    Human corrected:       {corrected}")
    if labelled > 0:
        print(f"    Agreement rate:        {(labelled - corrected) / labelled * 100:.1f}%")
    print(f"\n  Intent distribution:")
    print(df['expected_intent'].value_counts().to_string())
    print(f"\n  Saved to: {os.path.abspath(df.attrs.get('path', 'data/golden_eval_handlabelled.csv'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand-label the golden evaluation set.")
    parser.add_argument('--input', type=str, default='data/golden_eval.csv',
                        help='Path to the golden set CSV with heuristic pre-labels')
    parser.add_argument('--output', type=str, default='data/golden_eval_handlabelled.csv',
                        help='Path to save the hand-labelled output')
    args = parser.parse_args()

    label_golden_set(args.input, args.output)
