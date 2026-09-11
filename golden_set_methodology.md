# Golden Set Methodology

To evaluate the AI agent, we created a "Golden Evaluation Set" of 200 labeled examples.

## Sampling Strategy
We pulled 200 records from the preprocessed AppleSupport conversation pairs (`data/processed_conversations.csv`) using a fixed random seed (`42`) for reproducibility.

## Intent Taxonomy — Derived from the Data

The intent categories were **discovered through data analysis** of the full 106,624-conversation AppleSupport corpus (see `src/intent_discovery.py`). The process:

1. **Unigram frequency analysis** — after stopword removal, the top keywords were: *phone* (22k), *iphone* (20k), *ios* (16k), *update* (16k), *fix* (14.5k), *battery* (8.5k), *app* (5k), *screen* (4.5k), *music* (3.4k), *wifi* (2.3k), *icloud* (1.6k), etc.
2. **Bigram frequency analysis** — top patterns included *ios update* (2.2k), *battery life* (1.7k), *not working* (1.1k), *apple music* (1.1k), *question mark* (1.5k).
3. **Thematic grouping** — keywords were clustered into coherent intent categories based on co-occurrence and semantic similarity.
4. **Coverage validation** — the final 9-category taxonomy covers ~60% of the corpus with specific intents; the remaining ~40% are genuine general inquiries (follow-ups, non-English, vague messages).

### Final Intent Categories

| Intent | Description | Corpus Share |
|---|---|---|
| `battery_charging` | Battery life, charging, overheating, power issues | 10.4% |
| `account_access` | iCloud, Apple ID, login, password, verification | 3.9% |
| `connectivity` | WiFi, Bluetooth, cellular, network problems | 4.1% |
| `keyboard_typing` | Keyboard, autocorrect, typing glitches, emoji | 6.5% |
| `audio_media` | Siri, Apple Music, speakers, AirPods, audio | 5.6% |
| `hardware_repair` | Physical damage, warranty, repair, Genius Bar | 3.0% |
| `app_bug_crash` | Crashes, freezes, glitches, sluggishness | 11.1% |
| `software_update` | iOS/macOS updates, upgrade-related problems | 15.0% |
| `general_inquiry` | General questions, follow-ups, vague complaints | ~40% |

## Labelling Process — Human-Reviewed

The labelling was done in **two stages**:

### Stage 1: Heuristic Pre-labelling
A keyword-based classifier (`src/intent_discovery.py::classify_by_heuristic`) assigns initial labels using the data-derived taxonomy. This speeds up the process by providing sensible defaults.

### Stage 2: Human Review & Correction
Each of the 200 examples was individually reviewed using the interactive labelling tool (`src/label_golden_set.py`). For every example, the human reviewer:

1. Read the **customer query** and the **brand reply** (for context).
2. Saw the heuristic-suggested **intent** and **auto-handle** decision.
3. Either **confirmed** (pressed Enter) or **corrected** (typed the right label).
4. For escalated cases, verified or wrote the **escalation reason**.

The tool tracks which labels were human-confirmed vs. human-corrected, giving us a built-in measure of heuristic agreement.

### How to reproduce
```bash
# Step 1 — Generate heuristic pre-labels (already done, data/golden_eval.csv exists)
python3 -m src.generate_golden_set --input data/processed_conversations.csv --output data/golden_eval.csv --n 200

# Step 2 — Hand-label (interactive, ~30–45 min)
python3 -m src.label_golden_set --input data/golden_eval.csv --output data/golden_eval_handlabelled.csv
```

The tool saves progress after every 10 examples and supports resuming, so it can be done across multiple sessions.

### Label Schema

1. **Intent** (one of the 9 categories above)
2. **Auto-Handle vs. Escalate**:
   - **Escalate (False)**: Highly negative sentiment, complex multi-step issues, or cases where historical context provides no clear resolution path.
   - **Auto-Handle (True)**: Standard queries where the agent can confidently draft a resolution.
3. **Escalation Reason**: Free-text explanation of why a human is needed.
