import re
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    def __init__(self):
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.idx_vocab_no_pad = {}
        self.pad_token = "<P>"
        self.sta_token = "<S>"
        self.eos_token = "<E>"
        self.unknown_token = "<U>"
        self.special_tokens = [self.pad_token, self.unknown_token, self.eos_token, self.sta_token]
        self.vocab_set = set(self.special_tokens)
    
    @abstractmethod
    def build_vocab(self, texts):
        pass
        
    @abstractmethod
    def _tokenize(self, texts):
        pass
    
    def special_token_to_index(self, token):
        """Get the index of a special token."""
        idx = self.vocab_idx.get(token, None)
        assert idx is not None, "Error: build_vocab() must be called before special_token_to_index()"
        return idx

    def vocab_size(self):
        return len(self.vocab_idx)

    def text_to_indices(self, text):
        """Converts a list of strings into a list of lists of indices."""
        tokens = self._tokenize([text])[0]
        return [self.vocab_idx.get(t, self.vocab_idx[self.unknown_token]) for t in tokens]

    def texts_to_indices(self, texts):
        """Converts a list of strings into a list of lists of indices."""
        return [self.text_to_indices(text) for text in texts]

    def _mask_filter(self, str, mask):
        if mask:
            return str
        else:
            return f"[{str}]"

    def indices_to_texts(self, indices, masks=None, hide_pad=True):
        """Converts a list of lists of indices back to their original strings."""
        idx_v = self.idx_vocab_no_pad if hide_pad else self.idx_vocab

        if masks:
            assert len(indices) == len(masks), f"Mask should be same length as indices. Expected {len(indices)} found {len(masks)}"
            return [''.join([self._mask_filter(idx_v[idx], mask) for idx, mask in zip(seq, mask)]) for seq, mask in zip(indices, masks)]
        else:
            return [''.join([idx_v.get(idx, self.unknown_token) for idx in text_indices]) for text_indices in indices]

    def indices_to_text(self, indices, mask=None, hide_pad=True):
        """Converts a list of lists of indices back to their original strings."""
        idx_v = self.idx_vocab_no_pad if hide_pad else self.idx_vocab

        if mask:
            assert len(indices) == len(mask), f"Mask should be same length as indices. Expected {len(indices)} found {len(masks)}"
            return ''.join([self._mask_filter(idx_v[idx], m) for idx, m in zip(indices, mask)])
        else:
            return ''.join([idx_v.get(idx, self.unknown_token) for idx in indices])

    def wrap(self, token_list):
        start_token = self.special_token_to_index(self.sta_token)
        end_token = self.special_token_to_index(self.eos_token)
        for t in token_list:
            t.insert(0, start_token)
            t.append(end_token)


class WordTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def _tokenize(self, texts):
        """Tokenize a list of strings into words."""
        return [re.findall(r'\S+|\s', text) for text in texts]
    
    def build_vocab(self, texts):
        """Creates a vocabulary for words present in the training text and sorts them by frequency."""
        self.vocab_set.update(self.special_tokens)  # Directly add all special tokens

        token_freq = {}  # Dictionary to store token frequencies
        for text in texts:
            words = self._tokenize([text])[0]
            for word in words:
                token_freq[word] = token_freq.get(word, 0) + 1

        # Sorting tokens by frequency
        sorted_tokens = sorted(token_freq.keys(), key=lambda k: token_freq[k], reverse=True)

        # Updating vocab_set with the sorted tokens
        self.vocab_set.update(sorted_tokens)

        self.vocab_idx = {word: idx for idx, word in enumerate(self.special_tokens + sorted_tokens)}
        self.idx_vocab = {idx: word for word, idx in self.vocab_idx.items()}

        self.idx_vocab_no_pad = {idx: word for word, idx in self.vocab_idx.items()}

        # Set pad tokens to nothing
        token_idx = self.special_token_to_index(self.pad_token)
        self.idx_vocab_no_pad[token_idx] = ''


class UTF8Tokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def _tokenize(self, texts):
        """Tokenize a list of strings by individual UTF-8 characters."""
        return [list(text) for text in texts]

    def build_vocab(self, texts):
        """Creates a vocabulary for utf-8 characters present in the training text."""
        self.vocab_set.update(self.special_tokens)  # Directly add all special tokens

        for text in texts:
            self.vocab_set.update(text)

        self.vocab_idx = {char: idx for idx, char in enumerate(sorted(self.vocab_set))}
        self.idx_vocab = {idx: char for char, idx in self.vocab_idx.items()}
        self.idx_vocab_no_pad = {idx: char for char, idx in self.vocab_idx.items()}

        # Set pad tokens to nothing
        token_idx = self.special_token_to_index(self.pad_token)
        self.idx_vocab_no_pad[token_idx] = ''
    
class BPETokenizer(Tokenizer):
    def __init__(self, max_vocab_size=32000):
        super().__init__()
        self.max_vocab_size = max_vocab_size

    def _get_stats(self, vocab):
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            w_out = word.replace(bigram, replacement)
            new_vocab[w_out] = vocab[word]
        return new_vocab

    def build_vocab(self, texts):
        vocab = {}
        for word in texts:
            word = ' '.join(list(word)) + ' </w>'  # Convert word to characters and add end of word symbol
            vocab[word] = vocab.get(word, 0) + 1

        vocab_size = sum(len(word.split()) for word in vocab)
        while vocab_size < self.max_vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            vocab_size += 1

        # Build vocab_idx and idx_vocab
        tokens = list(set(' '.join(list(vocab.keys())).split()))
        self.vocab_idx = {token: i for i, token in enumerate(self.special_tokens + tokens)}
        self.idx_vocab = {i: token for token, i in self.vocab_idx.items()}
        self.idx_vocab_no_pad = {idx: char for char, idx in self.vocab_idx.items()}

        self.vocab_set = set(tokens)
        self.vocab_set.update(self.special_tokens)
        
        # Set pad tokens to nothing
        token_idx = self.vocab_idx.get(self.pad_token,None)
        assert token_idx is not None, "Internal error handling special tokens"
        self.idx_vocab_no_pad[token_idx] = ''
        

    def _tokenize(self, text):
        """Tokenize the input text using BPE."""
        print("1",text)
        text = ' '.join(list(text)) + ' </w>'
        print("2",text)
        for token in self.vocab_idx:
            text = text.replace(' '.join(list(token)), token)
        print("3",text.split())
        return text.split()


if __name__ == "__main__":
    input_texts = [
        "hello world!",
        "NLTK is a leading platform for building Python programs.",
        "See spot run. Run spot run!",
        "My # is 123-456-7890. Got that?",
        "Hello!!!? Are you there??",
        "this is a test\nwith a newline\tand a tab"
    ]
    
    print(input_texts)
    
    print('\n--- UTF8Tokenizer ---\n')
    
    # Character tokenizer usage:
    tokenizer = UTF8Tokenizer()

    tokenizer.build_vocab(input_texts)
    
    text_indices = tokenizer.texts_to_indices(input_texts)
    print("Indices:", text_indices)
    
    reconstructed_texts = tokenizer.indices_to_texts(text_indices)
    print("Reconstructed:", reconstructed_texts)
    
    print('\n--- WordTokenizer ---\n')
    
    # Word Tokenizer usage:
    tokenizer = WordTokenizer()

    tokenizer.build_vocab(input_texts)

    text_indices = tokenizer.texts_to_indices(input_texts)
    print("Indices:", text_indices)
    
    reconstructed_texts = tokenizer.indices_to_texts(text_indices)
    print("Reconstructed:", reconstructed_texts)
    
    print('\n--- BPETokenizer ---\n')
    
    # Word Tokenizer usage:
    tokenizer = BPETokenizer()

    tokenizer.build_vocab(input_texts)

    text_indices = tokenizer.texts_to_indices(input_texts)
    print("Indices:", text_indices)
    
    print(tokenizer.idx_vocab)
    
    reconstructed_texts = tokenizer.indices_to_texts(text_indices)
    print("Reconstructed:", reconstructed_texts)

