import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from src.intent_discovery import INTENT_NAMES, INTENT_TAXONOMY

load_dotenv()

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
        from src.intent_discovery import classify_by_heuristic
        intent = classify_by_heuristic(query)
        
        # Diverse mock responses based on intent to make local testing more realistic
        mock_replies = {
            "battery_charging": "We'd like to look into this battery issue. Which iOS version are you currently running? Let us know.",
            "account_access": "We can help you regain access to your account. Please DM us your Apple ID and we'll take a look.",
            "connectivity": "Let's get you reconnected! Try toggling Airplane mode on and off, or resetting your network settings.",
            "keyboard_typing": "We're aware of the autocorrect issue and a fix is included in the latest update. Please update your device.",
            "audio_media": "We'd love to help with your audio issue. Does this happen on speaker, headphones, or both?",
            "hardware_repair": "We can help you set up a Genius Bar appointment for a repair. DM us your location to start.",
            "app_bug_crash": "Sorry to hear the app is crashing. Have you tried force closing the app and restarting your device?",
            "software_update": "We understand update issues can be frustrating. Please ensure you have enough storage space and try again.",
            "general_inquiry": "Thank you for reaching out! We are currently looking into this. DM us if you need more help."
        }
        
        # Varied follow-ups based on auto-handle
        # We will mock a few failures (auto_handle = False) for specific keywords
        is_angry = any(word in query.lower() for word in ['wtf', 'fuck', 'shit', 'sucks', 'worst'])
        auto_handle = not is_angry
        
        draft = mock_replies.get(intent, mock_replies["general_inquiry"])
        follow_up = "Is there anything else I can help you with?" if auto_handle else "A human specialist will take over to assist you."
        reason = "" if auto_handle else "High negative sentiment detected."
        
        return {
            "intent": intent,
            "draft_reply": draft,
            "auto_handle": auto_handle,
            "escalation_reason": reason,
            "follow_up": follow_up
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
4. Write a contextual follow-up: if auto_handle is true, ask a relevant follow-up question based on the customer's issue (e.g. "Would you like steps for a force restart?" or "Can you tell me which app is crashing?"). If auto_handle is false, write a brief handoff message explaining a human agent will take over.

Respond strictly in JSON format matching this schema:
{{
  "intent": "string (one of: {', '.join(INTENT_NAMES)})",
  "draft_reply": "string",
  "auto_handle": boolean,
  "escalation_reason": "string (empty if auto_handle is true)",
  "follow_up": "string (contextual follow-up question or escalation handoff message)"
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
            
            # Update history with both the reply and follow-up
            self.conversation_history.append({"role": "customer", "content": query})
            agent_msg = result_dict.get("draft_reply", "")
            follow_up = result_dict.get("follow_up", "")
            if follow_up:
                agent_msg += f" {follow_up}"
            self.conversation_history.append({"role": "agent", "content": agent_msg})
            
            return result_dict
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._mock_llm_response(query, context)
