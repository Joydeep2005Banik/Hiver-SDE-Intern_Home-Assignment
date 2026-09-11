# Golden Set Methodology

To evaluate the AI agent, we created a "Golden Evaluation Set" of 200 labeled examples.

## Sampling Strategy
We used a stratified random sampling approach. We pulled 200 records from the preprocessed AppleSupport conversation pairs (`data/processed_conversations.csv`) using a fixed random seed (`42`) to ensure reproducibility.

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

## Labeling Process
In a true production environment, these 200 samples would be hand-labeled by subject matter experts. For the purpose of this demonstration pipeline, the labels were generated using the data-derived heuristic classifier (`src/intent_discovery.py::classify_by_heuristic`) which applies the keyword taxonomy discovered above.

1. **Intent**: Classified via the priority-ordered keyword rules derived from corpus analysis.

2. **Auto-Handle vs. Escalate**:
   - **Escalate (False)**: If the query contains highly negative sentiment (e.g., swear words, "angry", "worst", "terrible") OR if the query is excessively long (>40 words) which indicates a complex issue.
   - **Auto-Handle (True)**: For all other queries where the sentiment is neutral/positive and the length is manageable.

3. **Escalation Reason**:
   - Provides a short string explaining the heuristic trigger (e.g., "High negative sentiment / Angry customer" or "Query is too long/complex").
