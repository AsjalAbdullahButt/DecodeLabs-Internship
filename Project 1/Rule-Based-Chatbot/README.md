# Rule-Based Chatbot

A pure Python rule-based chatbot implemented with O(1) dictionary lookups instead of long conditional chains.

## How to Run
Make sure you have Python 3.8+ installed. You can launch the bot using:
```bash
python main.py
```

## Supported Commands/Intents
Here are some of the intents the chatbot recognizes:
- **Greetings:** `hello`, `hi`, `hey`
- **Farewell / Exit:** `bye`, `goodbye`, `exit`, `quit`
- **Identity:** `who are you`, `what are you`
- **Capability:** `what can you do`, `help`
- **Status:** `how are you`
- **Gratitude:** `thanks`, `thank you`
- **Time:** `what time is it`
- **Fun:** `tell me a joke`

## Architecture Notes & Deep Analysis

### Why a Dictionary over If-Elif Chains?
This project **strictly avoids** `if-elif` chains for intent matching. Instead, it uses a Python dictionary with `.get()`.

**Complexity Comparison:**

| Approach | Average Case | Worst Case | Scalability | Code Clarity |
|----------|--------------|-----------|-------------|--------------|
| Dictionary Lookup | **O(1)** | O(n)* | Excellent | Declarative |
| If-Elif Chain | **O(n)** | O(n) | Poor | Verbose |

*Python's hash implementation rarely encounters collisions with modern hardware.

**Real-World Impact:**
- 8 intents (current): If-elif evaluates ~4 conditions avg. vs Dict: 1 lookup.
- 1,000 intents: If-elif evaluates ~500 conditions avg. vs Dict: 1 lookup.
- 10,000 intents: If-elif evaluates ~5,000 conditions avg. vs Dict: 1 lookup.

### Execution Flow Analysis

```
User Input → Sanitize() → Exit Check → Knowledge Base Lookup → Response
                ↓              ↓              ↓
          [O(n): scan]  [O(1): tuple]  [O(1): hash table]
```

**Critical Design Decision:** Exit checks occur BEFORE knowledge base lookup.
This ensures `"bye"` functions as an exit command, not a greeting response. Without this ordering, the user couldn't exit using a word from the knowledge base.

### Implementation Details

**Sanitize Function:**
- `.lower()` + `.strip()` normalize input for consistent matching
- Handles: `"  HELLO  "` → `"hello"`
- Time: O(n) where n = string length
- Prevents duplicate intents for different cases

**Get Response Function:**
- Uses `dict.get(key, default)` for safe lookups
- Avoids `KeyError` exceptions through default fallback message
- Time: O(1) average (hash table lookup)
- Gracefully degrades unknown inputs instead of crashing

**REPL Loop:**
1. Reads user input (blocks on `input()`)
2. Sanitizes for normalization
3. Checks exit conditions before response lookup
4. Skips empty inputs silently
5. Fetches response via O(1) dictionary lookup
6. Prints formatted bot response
7. Catches `KeyboardInterrupt` (Ctrl+C) for clean shutdown

### Why This Scales Better

| Knowledge Base Size | If-Elif Evaluations | Dict Lookups | Time Ratio |
|---|---|---|---|
| 10 intents | ~5 avg | 1 | 5x slower |
| 100 intents | ~50 avg | 1 | 50x slower |
| 1,000 intents | ~500 avg | 1 | 500x slower |

A dictionary-based approach remains constant regardless of knowledge base size, making it ideal for chatbot scaling.
