
class UTF8Tokenizer:
    def __init__(self):
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.pad_token = "<PAD>"
        self.unknown_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.unknown_token]
    
    def tokenize(self, texts):
        """Tokenize a list of strings by individual UTF-8 characters."""
        return [list(text) for text in texts]
    
    def build_vocab(self, texts):
        """Creates a vocabulary for utf-8 characters present in the training text."""
        vocab_set = set(self.special_tokens)
        for text in texts:
            vocab_set.update(text)

        self.vocab_idx = {char: idx for idx, char in enumerate(sorted(vocab_set))}
        self.idx_vocab = {idx: char for char, idx in self.vocab_idx.items()}
        return self.vocab_idx

    def texts_to_indices(self, texts):
        """Converts a list of strings into a list of lists of indices."""
        return [[self.vocab_idx.get(char, self.vocab_idx[self.unknown_token]) for char in text] for text in texts]

    def indices_to_texts(self, indices):
        """Converts a list of lists of indices back to their original strings."""
        return [''.join([self.idx_vocab[idx] for idx in text_indices]) for text_indices in indices]

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

    text_indices = tokenizer.texts_to_indices(input_texts)
    print("Indices:", text_indices)
    
    reconstructed_texts = tokenizer.indices_to_texts(text_indices)
    print("Reconstructed:", reconstructed_texts)
    