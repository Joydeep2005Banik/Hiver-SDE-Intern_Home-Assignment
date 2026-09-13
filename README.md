# AI Customer Support Agent (AppleSupport)

This repository contains an end-to-end pipeline for a customer support AI agent. The agent uses RAG (Retrieval-Augmented Generation) based on historical AppleSupport data to draft replies, classify intents, and decide whether to auto-handle or escalate a customer query.

## 1. Quickstart (Reproduce in < 15 mins)

### Setup Environment
Install the dependencies:
```bash
pip install -r requirements.txt
```

### Configure LLM API Key (Optional but Recommended)
The agent uses Groq (`openai/gpt-oss-20b`) or OpenAI (`gpt-4o-mini`) to draft replies and evaluate quality. If you don't provide a key, the pipeline will fallback to a dummy mocked response, ensuring it can always be run quickly.
```bash
export GROQ_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

### Run the Pipeline
We have provided a pre-processed dataset (`data/processed_conversations.csv`) and a pre-built ChromaDB vector database (`data/chroma_db`) with 5,000 historical AppleSupport conversations.

To run the agent on the Golden Evaluation Set (first 10 examples for speed):
```bash
python3 src/run_pipeline.py --input data/golden_eval_handlabelled.csv --output data/predictions.csv --max 10
```

To run the full evaluation harness (Automated Metrics + LLM-as-a-judge):
```bash
python3 -m src.evaluate --input data/golden_eval_handlabelled.csv --output data/evaluation_results.csv --max 20
```

## 2. Golden Evaluation Set
We sampled 200 random queries to create our Golden Evaluation Set. Since the dataset was noisy, we first pre-labelled it using a strict heuristic rules engine based on our data-derived intent taxonomy. Then, we **100% human-reviewed and hand-labelled** every example using an interactive CLI tool (`src/label_golden_set.py`) to serve as our absolute ground truth.

- **Hand-labelled dataset:** [`data/golden_eval_handlabelled.csv`](data/golden_eval_handlabelled.csv)
- **Methodology details:** [`golden_set_methodology.md`](golden_set_methodology.md)

## 3. Evaluation Harness
Our evaluation harness computes both standard classification metrics and subjective generative quality scores.

- **Automated Metrics:** Calculates Intent Accuracy/F1 and Auto-Handle Routing Accuracy against the Golden Set, comparing the LLM pipeline dynamically against a Trivial (majority-class) and Simple (heuristic) baseline.
- **LLM-as-a-judge:** An LLM rubric scores the agent's drafted replies on Tone (1-5), Helpfulness (1-5), and Groundedness (1-5).
- **Human Calibration:** We measured how well the LLM judge agrees with a human evaluator using Cohen's Kappa ($\kappa$) on 20 examples. (Run `python3 -m src.llm_judge_calibration` to reproduce).

  | Metric | Kappa ($\kappa$) | Interpretation & Hypothesis |
  |---|---|---|
  | **Tone** | 0.000 | Degenerate: Humans uniformly scored 5/5 |
  | **Helpfulness** | 0.000 | Degenerate: Humans uniformly scored 5/5 |
  | **Groundedness** | 0.003 | Slight agreement: Humans caught hallucinated policies the LLM missed |

- **Evaluation File:** [`src/evaluate.py`](src/evaluate.py)

## 4. Report
The full evaluation report covers problem framing, failure analysis, misleading metrics, next steps, and system architecture.

- **See the report:** [`REPORT.pdf`](REPORT.pdf)

## 5. Decision Log
Here are 12 non-obvious decisions made during the design and implementation of this project:

1. **Used n-gram data analysis to discover intents rather than guessing them.** Ensures the taxonomy reflects actual customer problems (e.g. `keyboard_typing` for the well-known "I" autocorrect bug).
2. **Two-stage labelling (heuristic -> human review) for the golden set.** Meets the requirement for a hand-labelled dataset while saving hours of manual data entry.
3. **Output a dynamic `follow_up` field from the LLM.** Replaces hardcoded print statements with context-aware follow-ups for a more natural conversational flow.
4. **Chose `@AppleSupport` as the target brand.** High volume and a wide variety of intents (hardware, software, account, billing) provide a richer dataset for intent classification than a single-service brand.
5. **Used ChromaDB for retrieval (RAG) instead of prompting the LLM directly.** Grounds drafted replies in actual historical brand responses, reducing hallucinated generic troubleshooting advice.
6. **Chose Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings.** Lightweight, runs locally without an API key, and fast enough to process the corpus within the 15-minute reproduction budget.
7. **Had the LLM generate a dynamic `escalation_reason`.** A boolean `auto_handle=False` alone is not useful to a human agent; a generated reason (e.g. "customer is highly frustrated") gives immediate context on hand-off.
8. **Capped pipeline evaluation at a subset during development.** LLM API calls are rate-limited and cost money; `evaluate.py` exposes a `--max` flag for fast iteration before a full pass.
9. **Used Cohen's Kappa for human-LLM judge calibration.** Simple agreement does not account for chance agreement; kappa is the statistical standard for inter-rater reliability.
10. **Defined a fallback mock LLM response system.** Keeps the pipeline reproducible and runnable in under 15 minutes even without a supplied API key.
11. **Built an interactive CLI tool (`label_golden_set.py`) for human review.** Opening a CSV in a spreadsheet editor is error-prone; the CLI enforces valid enum inputs, shows context, and prevents accidental formatting corruption.
12. **Computed baselines dynamically in `evaluate.py` rather than hardcoding them.** If the taxonomy or dataset changes later, hardcoded baseline numbers in the report would go stale; dynamic computation keeps metrics accurate.

## 6. Citations
- **Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by Thought Vector (Kaggle).
- **AI Coding Assistants:** Claude Opus 4.6 (Thinking) and Gemini 3.1 Pro (High) were used for code generation, debugging, and report drafting throughout this project.
- **LLM API:** [Groq](https://groq.com/) (`openai/gpt-oss-20b`) for agent reply generation and LLM-as-a-judge scoring.
- **Libraries:** pandas, scikit-learn, ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`), OpenAI Python SDK, tqdm, python-dotenv.
