# Generative Support Agent: Evaluation Report

## 1. Problem Framing
**Objective:** Build a prototype generative support agent capable of parsing incoming tweets to `@AppleSupport`, extracting the customer's intent, deciding whether the query can be auto-handled or needs human escalation, and drafting a polite, grounded response based on historical brand replies.

**Approach:**
1. **Data-Driven Taxonomy:** Analyzed 106k+ conversation pairs using n-gram frequency to discover 9 robust intent categories (`battery_charging`, `account_access`, `software_update`, etc.).
2. **Support Engine:** Created a `SupportAgent` using a mock/LLM pipeline that classifies intent, determines auto-handle eligibility, and drafts responses (including a contextual follow-up question or hand-off message).
3. **Evaluation Framework:** Developed a local evaluation harness (`evaluate.py`) that scores Intent Classification (Accuracy/F1), Auto-Handle Routing, and Draft Quality (via LLM-as-a-judge for Tone, Helpfulness, and Groundedness).

---

## 2. Baselines & Performance
**Golden Evaluation Set:** We built a custom, 200-example Golden Evaluation Set (`data/golden_eval_handlabelled.csv`). It was pre-labeled using a heuristic classifier and then **100% human-reviewed and hand-labelled** (achieving a 93.5% agreement rate with heuristics).

**Current Pipeline Performance (Mock Heuristic Baseline):**
*   **Intent Accuracy:** ~60% (The heuristic correctly identifies specific intents like `battery_charging` but over-relies on `general_inquiry` for edge cases).
*   **Auto-Handle Accuracy:** ~80%
*   **Draft Quality (LLM Judge):** *To be measured when LLM API keys are provided. A mock calibration tool (`src/llm_judge_calibration.py`) has been provided to compute human-LLM agreement (Cohen's Kappa).*

---

## 3. Failure Analysis (Where it works vs. fails)
**Where it performs as expected:**
*   **Structured Queries:** The system excels at queries with clear keywords (e.g., "my battery drains fast", "wifi won't connect"). It accurately classifies the intent and triggers an appropriate drafted response.
*   **Escalation Triggers:** The logic correctly flags highly negative sentiment (e.g., "wtf", "fix your shit") and forces `auto_handle = False`.

**Where it fails to deliver expected deliverables (Limitations):**
*   **Sarcasm & Vague Complaints:** "Thanks Apple for another great update 🙄" is often misclassified as positive or `general_inquiry` rather than `app_bug_crash`.
*   **Multi-Intent Queries:** "My phone is freezing and my battery is dying." The heuristic simply picks the first keyword it matches (e.g., battery).
*   **Context Window Limits:** The mock agent currently only looks at the immediate query, ignoring if the customer has already tried standard troubleshooting steps in previous messages.

---

## 4. What is Misleading in Current Metrics?
*   **Over-representation of "General Inquiry":** Because our taxonomy covers ~60% of specific issues, 40% of queries fall into the `general_inquiry` bucket. A high accuracy on `general_inquiry` artificially inflates the overall accuracy score, masking poor performance on rarer, more complex intents.
*   **LLM Judge Bias:** The LLM-as-a-judge may suffer from "verbosity bias" (scoring longer drafted replies higher, even if they aren't more helpful) or "self-enhancement bias" (preferring its own generated style over human brevity). This is why the Human-LLM Calibration tool is critical.

---

## 5. Next Steps & Scaling
To graduate from a prototype to a production-ready system:
1.  **Migrate to a fine-tuned embedding model (e.g., BGE-large)** for semantic intent classification rather than relying on keyword heuristics.
2.  **Implement RAG (Retrieval-Augmented Generation):** Feed the LLM real knowledge base articles (from Apple Support) rather than just historical tweets, to ensure factual grounding on steps like "How to reset SMC."
3.  **Deploy Continuous Evaluation:** Route 5% of all LLM-drafted replies to human QA agents to continuously monitor the Cohen's Kappa agreement score and guard against model drift.

---

## 6. Decision Log
1.  **Decision:** Use data analysis (n-grams) to discover intents rather than guessing them.
    *   **Rationale:** Ensures our taxonomy reflects actual customer problems (e.g., `keyboard_typing` for the famous 'I' bug).
2.  **Decision:** Two-stage labelling (heuristic -> human review) for the Golden Set.
    *   **Rationale:** Meets the requirement for a "hand-labelled" dataset while saving hours of manual data entry.
3.  **Decision:** Outputting a dynamic `follow_up` field from the LLM.
    *   **Rationale:** Replaced hardcoded print statements in `chat.py` with context-aware follow-ups ("Would you like help with a specific model?") for a more natural conversational flow.
