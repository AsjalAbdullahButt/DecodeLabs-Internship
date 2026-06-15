"""
Core logic module for the rule-based chatbot.
Uses a dictionary for O(1) intent matching to avoid if-elif chains.

DESIGN RATIONALE:
- Dictionary-based lookups provide O(1) average time complexity vs O(n) for if-elif chains.
- This scales linearly with knowledge base growth; if-elif would scale poorly.
- Each key is normalized (lowercase), ensuring case-insensitive matching.
- The .get() method safely handles unknown inputs with a fallback response.
"""

# KNOWLEDGE_BASE: A mapping of exact user intents (cleaned inputs) to bot responses.
# Keys are lowercase strings representing user queries after sanitization.
# Values are predefined responses that are returned to the user.
# Strategy: Use hashing (dict internals) for O(1) average lookup, not linear search.
KNOWLEDGE_BASE = {
    # Greetings
    "hello": "Hello there! How can I assist you today?",
    "hi": "Hi! What's on your mind?",
    "hey": "Hey! How are you doing?",
    
    # Farewell
    "bye": "Goodbye! See you later.",
    "goodbye": "Farewell! Have a great day.",
    
    # Identity
    "who are you": "I am a simple rule-based chatbot written in Python.",
    "what are you": "I am a dictionary-powered chatbot bot.",
    
    # Capability
    "what can you do": "I can greet you, tell a joke, and chat a bit. Try saying 'hello' or 'tell me a joke'.",
    "help": "You can say things like: hello, who are you, how are you, thanks, tell me a joke, or bye to exit.",
    
    # Status
    "how are you": "I'm just a bunch of code, but I'm doing great! Thanks for asking.",
    
    # Gratitude
    "thanks": "You're very welcome!",
    "thank you": "No problem at all! Glad to help.",
    
    # Time
    "what time is it": "I don't have real-time access right now, but it's always an excellent time to code!",
    
    # Fun
    "tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs!"
}

def sanitize(raw_input: str) -> str:
    """
    SANITIZE: Normalizes raw user input for consistent intent matching.
    
    ALGORITHM:
    1. .lower() - Converts all characters to lowercase for case-insensitive matching.
       (Ensures 'HELLO', 'Hello', 'hello' all map to the same intent)
    2. .strip() - Removes leading/trailing whitespace (spaces, tabs, newlines).
       (Handles accidental formatting: '  hello  ' becomes 'hello')
    
    RETURN: Cleaned string ready for dictionary lookup.
    
    TIME COMPLEXITY: O(n) where n is string length (both operations scan the string).
    SPACE COMPLEXITY: O(n) for creating new string objects (strings are immutable in Python).
    """
    return raw_input.lower().strip()

def get_response(clean_input: str) -> str:
    """
    GET_RESPONSE: Retrieves the appropriate bot response for a given user intent.
    
    LOOKUP STRATEGY:
    - Uses dict.get(key, default) instead of dict[key] to avoid KeyError exceptions.
    - This is safer and more Pythonic than try-except KeyError handling.
    
    ALGORITHM:
    1. Hash the clean_input key using Python's built-in hash function (O(1) average).
    2. Look up the hash bucket in the internal hash table (O(1) average).
    3. If found in KNOWLEDGE_BASE, return the mapped response.
    4. If not found, return the fallback message (graceful degradation).
    
    RETURN: Either a knowledge base response or a default fallback message.
    
    TIME COMPLEXITY: O(1) average case (hash table lookup).
    WORST CASE: O(n) if hash collisions occur (rare with Python's hash implementation).
    """
    # Using dictionary .get() allows us to easily handle unknown inputs without raising KeyError
    return KNOWLEDGE_BASE.get(clean_input, "I do not understand. Try 'help' to see what I can do.")
