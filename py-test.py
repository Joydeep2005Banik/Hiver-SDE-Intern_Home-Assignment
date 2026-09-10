





from src.agent import SupportAgent

# Initialize the agent (this loads the ChromaDB vector database)
agent = SupportAgent()

# Try a custom Apple Support query
query = "My iPhone 12 Pro Max keeps crashing when I open the camera app."

# See what the RAG database retrieved as historical context
context = agent.retrieve_context(query)
print("--- Historical Context Found ---")
print(context)

# Process the query through the LLM
result = agent.process_query(query)
print("\n--- Agent's Decision ---")
print(result)
