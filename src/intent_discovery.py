"""
Intent Discovery — data-driven taxonomy derivation for AppleSupport.

This script analyses the processed AppleSupport conversation corpus to discover
natural intent categories.  The approach is:

  1. Unigram + bigram frequency analysis (after stopword removal) to surface
     the most common topics customers talk about.
  2. Thematic keyword grouping — cluster the top keywords into coherent themes.
  3. Coverage analysis — measure what share of the corpus each theme covers and
     how large the residual "general_inquiry" bucket is.
  4. Output: a summary table + the final INTENT_TAXONOMY dict that downstream
     modules (agent.py, generate_golden_set.py) import.

Run standalone to reproduce the analysis:
    python3 -m src.intent_discovery --input data/processed_conversations.csv
"""

import pandas as pd
import re
import argparse
from collections import Counter

# ---------------------------------------------------------------------------
# Final intent taxonomy derived from the data analysis below.
# Ordered by priority (first match wins in the heuristic labeller).
# ---------------------------------------------------------------------------
INTENT_TAXONOMY = {
    "battery_charging": {
        "description": "Battery life, charging, overheating, and power issues",
        "pattern": r"batter|charg|drain|power.?off|dies|overheating|overheat",
        "corpus_share": "10.4%",
    },
    "account_access": {
        "description": "iCloud, Apple ID, login, password, and verification issues",
        "pattern": r"icloud|apple.?id|account|log.?in|sign.?in|password|locked.?out|verification|two.?factor|disabled",
        "corpus_share": "3.9%",
    },
    "connectivity": {
        "description": "WiFi, Bluetooth, cellular signal, and network problems",
        "pattern": r"wifi|wi-fi|bluetooth|cellular|signal|network|\blte\b|no.?service|connect",
        "corpus_share": "4.1%",
    },
    "keyboard_typing": {
        "description": "Keyboard, autocorrect, typing glitches, and emoji issues",
        "pattern": r"keyboard|typing|autocorrect|predictive|letter.{0,3}i|question.?mark|emoji",
        "corpus_share": "6.5%",
    },
    "audio_media": {
        "description": "Siri, Apple Music, speakers, AirPods, volume, and audio issues",
        "pattern": r"\bsiri\b|\bmusic\b|speaker|headphone|airpod|volume|sound|audio|itunes|apple.?music",
        "corpus_share": "5.6%",
    },
    "hardware_repair": {
        "description": "Physical device damage, warranty, repair, Genius Bar, and replacements",
        "pattern": r"macbook|imac|screen.?crack|broken|scratch|warranty|repair|genius|appointment|replac|refund|service.?cent",
        "corpus_share": "3.0%",
    },
    "app_bug_crash": {
        "description": "App crashes, freezes, glitches, sluggishness, and things not working",
        "pattern": r"crash|bug|glitch|freez|frozen|hang[s ]|stuck|\blag\b|\bslow\b|not.?working|stopped.?working",
        "corpus_share": "11.1%",
    },
    "software_update": {
        "description": "iOS / macOS updates, software upgrades, and update-related problems",
        "pattern": r"\bios\b|\bupdate[ds]?\b|\bsoftware\b|\bupgrad|\bupdating\b",
        "corpus_share": "15.0%",
    },
    "general_inquiry": {
        "description": "Catch-all: general questions, follow-ups, non-English queries, and vague complaints",
        "pattern": r".*",
        "corpus_share": "~40%",
    },
}

INTENT_NAMES = list(INTENT_TAXONOMY.keys())


def classify_by_heuristic(text: str) -> str:
    """Classify a customer query using the data-derived keyword taxonomy.
    
    Returns the first matching intent (priority-ordered) or 'general_inquiry'.
    """
    text = text.lower()
    for intent_name, meta in INTENT_TAXONOMY.items():
        if intent_name == "general_inquiry":
            return "general_inquiry"
        if re.search(meta["pattern"], text):
            return intent_name
    return "general_inquiry"


# ---------------------------------------------------------------------------
# Standalone analysis — run this to reproduce the intent discovery process
# ---------------------------------------------------------------------------

def _top_keywords(queries, n=60):
    """Return the top-n unigram keywords after stopword removal."""
    stopwords = {
        "the", "and", "for", "that", "this", "with", "you", "have", "not",
        "are", "but", "was", "can", "has", "had", "its", "from", "they",
        "been", "will", "one", "all", "would", "there", "what", "about",
        "when", "make", "like", "just", "over", "also", "did", "get", "how",
        "got", "your", "applesupport", "apple", "support", "please", "help",
        "need", "know", "why", "don", "use", "does", "isn", "won", "any",
        "still", "new", "even", "now", "try", "going", "keep", "say",
        "since", "every", "let", "see", "able", "want", "something", "take",
        "thing", "come", "put", "way", "back", "using", "much", "really",
        "should", "could", "here", "work", "day", "trying", "getting",
        "told", "doesn", "time", "after", "doing", "other", "some",
    }
    counter = Counter()
    for q in queries:
        words = re.findall(r"\b[a-z]{3,}\b", q)
        counter.update(words)
    for sw in stopwords:
        counter.pop(sw, None)
    return counter.most_common(n)


def _top_bigrams(queries, n=40):
    """Return the top-n bigrams."""
    counter = Counter()
    for q in queries:
        words = re.findall(r"\b[a-z]{3,}\b", q)
        for i in range(len(words) - 1):
            counter[f"{words[i]} {words[i + 1]}"] += 1
    return counter.most_common(n)


def run_analysis(input_csv: str):
    """Run the full intent-discovery analysis and print results."""
    df = pd.read_csv(input_csv)
    queries = df["customer_query"].str.lower().tolist()
    total = len(queries)

    print(f"Corpus size: {total:,} AppleSupport conversation pairs\n")

    # ---- Step 1: keyword frequencies ----
    print("=" * 60)
    print("STEP 1 — Top unigram keywords (after stopword removal)")
    print("=" * 60)
    for word, count in _top_keywords(queries, 40):
        print(f"  {word:20s} {count:6d}  ({count / total * 100:5.1f}%)")

    print()
    print("=" * 60)
    print("STEP 2 — Top bigrams")
    print("=" * 60)
    for bg, count in _top_bigrams(queries, 30):
        print(f"  {bg:30s} {count:6d}  ({count / total * 100:5.1f}%)")

    # ---- Step 3: apply taxonomy and show coverage ----
    print()
    print("=" * 60)
    print("STEP 3 — Intent taxonomy coverage (priority-ordered, first match wins)")
    print("=" * 60)
    labels = [classify_by_heuristic(q) for q in queries]
    dist = pd.Series(labels).value_counts()
    for intent_name in INTENT_NAMES:
        count = dist.get(intent_name, 0)
        desc = INTENT_TAXONOMY[intent_name]["description"]
        print(f"  {intent_name:25s} {count:6d}  ({count / total * 100:5.1f}%)  — {desc}")

    # ---- Step 4: observations ----
    print()
    print("=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)
    print("""
  • The 8 specific intents cover ~60% of the corpus, with the remaining
    ~40% falling into 'general_inquiry'. This is expected for a noisy
    Twitter dataset that includes follow-ups ("@AppleSupport yes"),
    non-English tweets, and vague single-word messages.

  • The top themes directly mirror the highest-frequency keywords:
    phone/iphone/ios/update → software_update
    crash/bug/glitch/slow  → app_bug_crash
    battery/charge/drain   → battery_charging

  • Themes are priority-ordered so a tweet mentioning both "battery" and
    "update" is classified as battery_charging (the more specific intent)
    rather than software_update.

  • 9 total intent categories keep the set manageable for evaluation while
    still capturing the major topic areas that appear in the data.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover intents from the AppleSupport corpus.")
    parser.add_argument("--input", type=str, default="data/processed_conversations.csv")
    args = parser.parse_args()
    run_analysis(args.input)
