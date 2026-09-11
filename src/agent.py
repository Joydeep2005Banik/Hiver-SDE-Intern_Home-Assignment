import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from src.intent_discovery import INTENT_NAMES, INTENT_TAXONOMY

class SupportAgent:
    def __init__(self, db_path: str = 'data/chroma_db'):
        # Initialize Vector DB
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_collection(name="support_kb", embedding_function=self.emb_fn)
        
        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key:
            self.llm_client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
            self.model_name = "openai/gpt-oss-20b"
        elif api_key:
            self.llm_client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
        else:
            print("WARNING: Neither OPENAI_API_KEY nor GROQ_API_KEY found in environment. The agent will mock LLM responses if run.")
            self.llm_client = None

        self.conversation_history = []

    def retrieve_context(self, query: str, top_k: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        context = ""
        for idx in range(len(results['documents'][0])):
            past_query = results['documents'][0][idx]
            past_reply = results['metadatas'][0][idx]['reply']
            context += f"Past Customer: {past_query}\nBrand Reply: {past_reply}\n\n"
        return context.strip()

    def _mock_llm_response(self, query: str, context: str):
        # A fallback if no API key is provided, just for testing the pipeline flow.
        # Uses the data-derived heuristic classifier so mock results are
        # at least directionally correct.
        from src.intent_discovery import classify_by_heuristic
        return {
            "intent": classify_by_heuristic(query),
            "draft_reply": "Thank you for reaching out! We are currently looking into this. DM us if you need more help.",
            "auto_handle": True,
            "escalation_reason": ""
        }

    def process_query(self, query: str) -> dict:
        context = self.retrieve_context(query)
        
        if not self.llm_client:
            return self._mock_llm_response(query, context)
            
        # Build intent list string from the data-derived taxonomy
        intent_descriptions = "\n".join(
            f"  - {name}: {meta['description']}"
            for name, meta in INTENT_TAXONOMY.items()
        )

        system_prompt = f"""You are an AI support agent for AppleSupport on Twitter.
Your job is to read an incoming customer query and past similar resolved cases.

1. Classify the intent into EXACTLY ONE of the following categories (derived from analysis of 106k+ historical AppleSupport conversations):
{intent_descriptions}

2. Draft a reply grounded in how similar cases were resolved. Keep it short, polite, and under 280 characters.
3. Decide if the query can be auto-handled (True) or if it requires human escalation (False). Escalate if the customer is very angry, the issue is highly complex, or past context doesn't provide a clear solution. Provide a reason if escalated.

Respond strictly in JSON format matching this schema:
{{
  "intent": "string (one of: {', '.join(INTENT_NAMES)})",
  "draft_reply": "string",
  "auto_handle": boolean,
  "escalation_reason": "string (empty if auto_handle is true)"
}}"""
        
        # Format the history
        history_str = ""
        for msg in self.conversation_history:
            history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
        user_prompt = f"### Past Similar Cases:\n{context}\n\n"
        if history_str:
            user_prompt += f"### Conversation History:\n{history_str}\n\n"
        user_prompt += f"### Current Customer Query:\n{query}"
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            
            result_str = response.choices[0].message.content
            result_dict = json.loads(result_str)
            
            # Update history
            self.conversation_history.append({"role": "customer", "content": query})
            self.conversation_history.append({"role": "agent", "content": result_dict.get("draft_reply", "")})
            
            return result_dict
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._mock_llm_response(query, context)
