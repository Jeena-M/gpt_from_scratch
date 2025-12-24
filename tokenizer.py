class CharTokenizer:
    def __init__(self, text: str):
        # Get all unique characters in the dataset
        chars = sorted(list(set(text))) 
        self.vocab_size = len(chars)

        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        '''
        Convert a string into a list of integers
        '''
        return [self.stoi[c] for c in s]
    
    def decode(self, tokens):
        '''
        Convert a list of integers back into a string
        '''
        return "".join([self.itos[i] for i in tokens])
    
if __name__ == "__main__":
    text = "hello world"
    tokenizer = CharTokenizer(text)

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)