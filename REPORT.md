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

**Performance Comparison vs. Baselines:**
To prove our LLM pipeline provides value, we evaluate it against two baselines (calculated dynamically in `src/evaluate.py`):
1.  **Trivial Baseline (Majority Class):** Always predicts the most common intent (`general_inquiry`) and always predicts the most common auto-handle flag (`True`).
2.  **Simple Baseline (Heuristics):** Uses our regex keyword matcher (`classify_by_heuristic`) for intent, and sets `auto_handle = False` for `general_inquiry`, `True` otherwise.

| Metric | Trivial Baseline | Simple Baseline (Heuristics) | Pipeline (LLM API) |
|---|---|---|---|
| **Intent Accuracy** | 38.0% | 96.5%* | 60.0%** |
| **Auto-Handle Accuracy** | 84.0% | 59.0% | 80.0% |
| **Draft Quality (LLM Judge)** | N/A | N/A | Tone: 4.0/5, Help: 4.4/5, Ground: 4.1/5 |

*\*Note: The Simple Baseline's 96.5% intent accuracy is artificially inflated because the Golden Set was initially pre-labelled using the exact same heuristic before human review, strongly biasing the dataset towards queries that the heuristic is already good at.*
*\*\*Note: The LLM Pipeline was evaluated on a 20-sample subset to save API costs during development.*

**Human-LLM Agreement (Cohen's Kappa):**
Evaluated on 20 examples:
*   **Tone:** κ = 0.000 (Degenerate: human scored 5/5 uniformly; LLM judge averaged 4.0/5)
*   **Helpfulness:** κ = 0.000 (Degenerate: human scored 5/5 uniformly; LLM judge averaged 4.4/5)
*   **Groundedness:** κ = 0.003 (Slight: human found hallucinated policies the LLM judge missed)

---

## 3. Failure Analysis (Where it works vs. fails)
**Where it performs as expected:**
*   **Structured Queries:** The system excels at queries with clear keywords (e.g., "my battery drains fast", "wifi won't connect"). It accurately classifies the intent and triggers an appropriate drafted response.
*   **Escalation Triggers:** The logic correctly flags highly negative sentiment (e.g., "wtf", "fix your shit") and forces `auto_handle = False`.

**Top 5 Failure Modes (Limitations & Examples):**

1.  **Hidden Context (Missing Images/Media)**
    *   *Example:* "@AppleSupport hi my iphone 7 stays like this for hours. What I can do"
    *   *Hypothesis:* The customer attached an image or video showing their screen frozen, but the dataset only provides text. The text-only agent sees a generic inquiry and attempts to auto-handle it, whereas a human would escalate or ask device-specific questions based on the visual evidence.
2.  **Profanity & High Emotion (Escalation Misses)**
    *   *Example:* "lm using lowercase L’s as l’s until @AppleSupport fixes this question mark shit😑"
    *   *Hypothesis:* While explicit profanity triggers escalation, creative swearing or frustration ("shit", "trashed my brand new iphone") is sometimes misclassified as standard feedback, leading to an inappropriately cheerful automated reply to an angry customer.
3.  **Ambiguous Feature vs. Bug Classification**
    *   *Example:* "So you know that new SOS feature on the iPhone ? Well mine just popped off and I️ didn’t nothing to set it off . What’s going on @115858 ?"
    *   *Hypothesis:* The agent predicts `software_update` because of keywords like "new feature," but it's actually an `app_bug_crash` about the SOS feature behaving unexpectedly. Keyword heuristics fail on descriptive, conversational bug reports.
4.  **Misinterpreting Literal Strings vs. Grammar**
    *   *Example:* "@AppleSupport Ok, how do I fix “it” changing to I.T."
    *   *Hypothesis:* The agent classifies this as `general_inquiry` instead of the specific `keyboard_typing` autocorrect bug. Standard tokenization and n-gram analysis struggle to differentiate the literal string "it" from the pronoun "it".
5.  **Multi-Step Conversational Context Loss**
    *   *Example:* "@AppleSupport It’s hard to tell, nothing really reaches that far when I try. I have noticed that if I want to select something I have to start above it and drag my finger down to what I’m trying to press and then it’ll work"
    *   *Hypothesis:* This is clearly a customer replying in the middle of an ongoing thread about a touch screen calibration issue (`hardware_repair`). The stateless agent receives this single message, loses the prior context, and incorrectly auto-handles it as a `general_inquiry`.

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
4.  **Decision:** Choosing `@AppleSupport` as the brand over others.
    *   **Rationale:** AppleSupport has high volume and a wide variety of intents (hardware, software, account, billing), providing a rich dataset for intent classification compared to a single-service brand.
5.  **Decision:** Using `ChromaDB` for retrieval (RAG) instead of just prompting the LLM.
    *   **Rationale:** Ensures the drafted replies are grounded in *actual* historical brand policies, preventing the LLM from hallucinating generic troubleshooting advice.
6.  **Decision:** Choosing `SentenceTransformers` (`all-MiniLM-L6-v2`) for embeddings.
    *   **Rationale:** It's lightweight, runs locally without an API key, and is fast enough to process the corpus in under 15 minutes as required.
7.  **Decision:** Making the LLM generate a dynamic `escalation_reason`.
    *   **Rationale:** Returning just a boolean `False` for auto-handle isn't helpful to human agents. A generated reason (e.g., "Customer is highly frustrated") provides immediate context upon hand-off.
8.  **Decision:** Capping the pipeline evaluation at a subset during development.
    *   **Rationale:** LLM API calls are rate-limited and incur costs. The `evaluate.py` script supports a `--max` flag to allow fast iteration on the evaluation loop before running a full pass.
9.  **Decision:** Using Cohen's Kappa for human-LLM judge calibration.
    *   **Rationale:** Simple accuracy doesn't account for agreement by chance. Kappa is the statistical standard for measuring inter-rater reliability.
10. **Decision:** Defining a fallback mock LLM response system.
    *   **Rationale:** Ensures the grading pipeline is fully reproducible and runnable by evaluators in under 15 minutes, even if they don't supply a `.env` API key.
11. **Decision:** Creating an interactive CLI tool (`label_golden_set.py`) for human review.
    *   **Rationale:** Opening a CSV in Excel is error-prone. A CLI tool enforces valid enum inputs, displays context, and prevents accidental formatting corruption.
12. **Decision:** Computing baselines dynamically in `evaluate.py` rather than hardcoding.
    *   **Rationale:** If the taxonomy or dataset is updated in the future, hardcoded baseline numbers in the report would become stale. Dynamic calculation ensures metrics stay accurate.
