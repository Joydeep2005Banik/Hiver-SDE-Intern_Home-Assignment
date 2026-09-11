from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.agent import SupportAgent

def main():
    print("Initializing Support Agent...")
    agent = SupportAgent()
    print("Agent initialized! Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            query = input("\nYou: ")
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit']:
                print("Ending chat. Goodbye!")
                break
            
            if not query.strip():
                continue

            print("\nProcessing...")
            
            # Process the query
            result = agent.process_query(query)
            
            # Print the agent's response
            print("\nAgent's Decision:")
            print(f"Intent: {result.get('intent', 'Unknown')}")
            print(f"Auto-handle: {result.get('auto_handle', False)}")
            print(f"Draft Reply: {result.get('draft_reply', '')}")
            if result.get('escalation_reason'):
                print(f"Escalation Reason: {result.get('escalation_reason')}")
            
            # Print the LLM-generated contextual follow-up
            follow_up = result.get('follow_up', '')
            if follow_up:
                print(f"\n{follow_up}")
            
        except KeyboardInterrupt:
            print("\nEnding chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
