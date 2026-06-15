"""
Entry point for the rule-based chatbot.

EXECUTION FLOW:
1. Initialize: Print welcome banner and instructions.
2. REPL Loop: Read user input, process, and respond until exit condition.
3. Cleanup: Gracefully exit on 'exit'/'quit'/'bye' or Ctrl+C.

KEY DESIGN DECISIONS:
- The exit check (step 1b) PRECEDES the knowledge base lookup (step 2).
  This ensures "bye" functions as an exit command, not a greeting response.
- Empty input is silently skipped, preventing confusion from the fallback message.
- Ctrl+C is caught to prevent ugly stack traces and ensure clean shutdown.
"""

import sys
from chatbot.engine import sanitize, get_response

def main():
    # STEP 1: Print a welcome banner with the bot name and a short description on startup.
    # This provides immediate user feedback and instructions for interaction.
    print("=========================================")
    print(" Welcome to RuleBot!                     ")
    print(" A simple dictionary-powered chatbot     ")
    print("=========================================")
    print("Type 'help' to see what I can do.")
    print("Type 'exit', 'quit', or 'bye' to leave.\n")

    # STEP 2: REPL (Read-Eval-Print Loop) - Main interaction loop.
    # Wrapped in try-except to catch KeyboardInterrupt (Ctrl+C) for graceful exit.
    try:
        while True:
            # 2a. READ: Prompt user and read raw input from stdin.
            # input() blocks until user presses Enter, returning the string entered.
            raw_input = input("You: ")
            
            # 2b. PROCESS: Sanitize the raw input for consistent intent matching.
            # Calls sanitize() which applies .lower().strip() normalization.
            clean_input = sanitize(raw_input)
            
            # 2c. EXIT CHECK (CRITICAL ORDER): Check for exit conditions BEFORE knowledge base lookup.
            # This ensures "bye" is treated as an exit command, not as a greeting response.
            # Using tuple membership test for O(1) average-case lookup (Python optimizes).
            if clean_input in ("exit", "quit", "bye"):
                print("Bot: Goodbye! Have a fantastic day!")
                break  # Exit the while True loop cleanly.
                
            # 2d. EMPTY INPUT HANDLING: Skip empty strings to avoid unnecessary fallback response.
            # This prevents confusing the user with "I do not understand" for accidental Enter presses.
            if not clean_input:
                continue  # Jump back to 2a (Read), skipping response generation.
                
            # 2e. EVAL & RESPOND: Query the knowledge base for the appropriate response.
            # Calls get_response() which performs O(1) dictionary lookup.
            response = get_response(clean_input)
            print(f"Bot: {response}")  # PRINT: Display the response to the user.
            
    except KeyboardInterrupt:
        # CLEANUP: Catch Ctrl+C signal to provide graceful exit instead of stack trace.
        # KeyboardInterrupt is raised when the user presses Ctrl+C during input() or any operation.
        print("\nBot: Goodbye! Have a fantastic day!")  # (\n handles terminal formatting)
        sys.exit(0)  # Exit with status code 0 (success/clean exit).

# ENTRY POINT: Execute main() only when script is run directly, not when imported as a module.
# This allows the module to be imported for testing without automatically starting the REPL.
if __name__ == "__main__":
    main()
