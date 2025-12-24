# Tokenization in Language Models

## WHAT A TOKENIZER IS (CONCEPTUALLY)
The core problem: Neural networks do not understand text. They only understand numbers
Question: How do we convert raw text into numbers in a way a model can learn from? 
Answer: That conversion process is called tokenization. 

## WHY TOKENIZATION IS NECESSARY
- Models operate on tensors (numbers)
- Text is symbolic (characters, words, meaning)
- We need a stable, reversible mapping between text <-> numbers
That's literally the tokenizer's job.

### Part 1: Vocabulary Construction
set(text) -> finds all unique symbols in the dataset
- These symbols are the atomic units the model will operate on
- This set is the vocabulary; the model can only generate things that exist in this vocab

### Part 2: Symbol <-> Integer Mapping
- This creates a bijective mapping: 
    - Each character -> exactly one integer
    - Each integer -> exactly one character
- Reversibility is critical - otherwise, generation is impossible

### Part 3: Encoding
- Convert "hello" to [7, 4, 11, 11, 14] -> this is a sequence modeling problem (why LLMs are just next-token predictors)

### Part 4: Decoding
- The model outputs numbers, not words
- Dcecoding converts integers to text, allowing generated outputs to be interpreted. 

### Part 5: The Most Important Concept
- The tokenizer defines the model's universe
    - The model cannot reason about characters outside the vocab, cannot generate unseen symbols, learns probabilites over tokens

## Advantages of Character-Level Tokenization:
- No external dependencies
- Fully interpretable
- Minimal assumptions

## Disadvantages of Character-Level Tokenization:
- Longer sequence lengths 
- Lower semantic compression
- Inefficient for large-scale models 