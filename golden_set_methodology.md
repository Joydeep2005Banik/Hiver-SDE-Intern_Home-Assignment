# Golden Set Methodology

To evaluate the AI agent, we created a "Golden Evaluation Set" of 200 labeled examples.

## Sampling Strategy
We used a stratified random sampling approach. We pulled 200 records from the preprocessed AppleSupport conversation pairs (`data/processed_conversations.csv`) using a fixed random seed (`42`) to ensure reproducibility.

## Labeling Process
In a true production environment, these 200 samples would be hand-labeled by subject matter experts. For the purpose of this demonstration pipeline, the labels were generated using a heuristic script (`src/generate_golden_set.py`) which simulates the hand-labeling process:

1. **Intent**:
   - `software_issue`: Matches keywords like *ios, update, app, crash, software, bug, glitch*.
   - `hardware_issue`: Matches keywords like *battery, screen, crack, charge, hardware, button, macbook, phone*.
   - `account_issue`: Matches keywords like *password, apple id, account, login, icloud*.
   - `general_inquiry`: Default category for all other queries.

2. **Auto-Handle vs. Escalate**:
   - **Escalate (False)**: If the query contains highly negative sentiment (e.g., swear words, "angry", "worst", "terrible") OR if the query is excessively long (>40 words) which indicates a complex issue.
   - **Auto-Handle (True)**: For all other queries where the sentiment is neutral/positive and the length is manageable.

3. **Escalation Reason**:
   - Provides a short string explaining the heuristic trigger (e.g., "High negative sentiment / Angry customer" or "Query is too long/complex").
