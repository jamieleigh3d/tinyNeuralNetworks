class UTF8Tokenizer:
    def __init__(self):
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.pad_token = "<PAD>"
        self.sta_token = "<STA>"
        self.eos_token = "<EOS>"
        self.unknown_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.unknown_token, self.eos_token, self.sta_token]
        self.vocab_set = set(self.special_tokens)
        
    def tokenize(self, texts):
        """Tokenize a list of strings by individual UTF-8 characters."""
        return [list(text) for text in texts]
    
    def add_special_token(self, token):
        """Add a new special token to the tokenizer."""
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            self.build_vocab(self.special_tokens) # Rebuild vocab to include new special token
    
    def special_token_to_index(self, token):
        """Get the index of a special token."""
        return self.vocab_idx.get(token, None)
    
    def build_vocab(self, texts):
        """Creates a vocabulary for utf-8 characters present in the training text."""
        self.vocab_set.update(self.special_tokens)  # Directly add all special tokens
        
        for text in texts:
            self.vocab_set.update(text)

        self.vocab_idx = {char: idx for idx, char in enumerate(sorted(self.vocab_set))}
        self.idx_vocab = {idx: char for char, idx in self.vocab_idx.items()}
        return self.vocab_idx
        
    def vocab_size(self):
        return len(self.vocab_idx)

    def text_to_indices(self, text):
        """Converts a list of strings into a list of lists of indices."""
        return [self.vocab_idx.get(char, self.vocab_idx[self.unknown_token]) for char in text]

    def texts_to_indices(self, texts):
        """Converts a list of strings into a list of lists of indices."""
        return [[self.vocab_idx.get(char, self.vocab_idx[self.unknown_token]) for char in text] for text in texts]

    def indices_to_texts(self, indices):
        """Converts a list of lists of indices back to their original strings."""
        return [''.join([self.idx_vocab[idx] for idx in text_indices]) for text_indices in indices]

    def indices_to_text(self, indices):
        """Converts a list of lists of indices back to their original strings."""
        return ''.join([self.idx_vocab.get(idx,self.unknown_token) for idx in indices])


if __name__ == "__main__":
    input_texts = [
        "hello world!",
        "NLTK is a leading platform for building Python programs.",
        "See spot run. Run spot run!",
        "My # is 123-456-7890. Got that?",
        "Hello!!!? Are you there??",
        "this is a test\nwith a newline\tand a tab"
    ]

    tokenizer = UTF8Tokenizer()
    tokenizer.build_vocab(input_texts)

    # Example usage of the added functions:
    tokenizer.add_special_token("<CUSTOM>")
    print("Index of custom token:", tokenizer.special_token_to_index("<CUSTOM>"))

    text_indices = tokenizer.texts_to_indices(input_texts)
    print("Indices:", text_indices)
    
    reconstructed_texts = tokenizer.indices_to_texts(text_indices)
    print("Reconstructed:", reconstructed_texts)
