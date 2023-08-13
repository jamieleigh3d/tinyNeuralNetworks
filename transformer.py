import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        self.activations = []

        # Initializing weights and biases for each layer
        prev_size = input_size
        for size in hidden_sizes + [output_size]:
            self.weights.append(np.random.randn(prev_size, size) * 0.01)
            self.biases.append(np.zeros((1, size)))
            prev_size = size

    def forward(self, x):
        self.activations = []
        for weight, bias in zip(self.weights, self.biases[:-1]):
            x = self._relu(np.dot(x, weight) + bias)
            self.activations.append(x)
        # For the output layer, we'll just use a linear activation for simplicity
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        return output

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def backward(self, x, y, output, learning_rate=0.01):
        # Compute the derivative of the loss
        dloss = (output - y) / output.shape[0]
        
        # Backpropagate the error
        for i in reversed(range(len(self.weights))):
            dactivation = dloss if i == len(self.weights) - 1 else self._relu_derivative(self.activations[i])
            dloss = np.dot(dactivation, self.weights[i].T)
            self.weights[i] -= learning_rate * np.dot(self.activations[i-1].T if i != 0 else x.T, dactivation)
            self.biases[i] -= learning_rate * np.sum(dactivation, axis=0, keepdims=True)


def train(network, data, labels, epochs, learning_rate):
    for epoch in range(epochs):
        outputs = network.forward(data)
        loss = np.mean(0.5 * (outputs - labels) ** 2)
        network.backward(data, labels, outputs, learning_rate)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


def generate_dataset(word_list, n=2, num_pairs=1000):
    """
    Generates a dataset of word pairs and labels them as rhyming or not.
    
    Parameters:
    - word_list: list of words
    - n: number of characters to consider from the end of words
    - num_pairs: number of word pairs to generate

    Returns:
    - pairs: list of word pairs
    - labels: list of labels (1 if rhymes, 0 if not)
    """
    
    pairs = []
    labels = []

    for _ in range(num_pairs):
        # Randomly select two words
        word1, word2 = np.random.choice(word_list, 2, replace=False)
        
        # Check if they rhyme
        if word1[-n:] == word2[-n:]:
            pairs.append((word1, word2))
            labels.append(1)  # Rhymes
        else:
            pairs.append((word1, word2))
            labels.append(0)  # Doesn't rhyme

    return pairs, labels

def build_char_vocab():
    """
    Build a character vocabulary from a list of words.
    """
    words = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
        'y', 'z', ' ', '.', ',', "'", '!', '\n', '"', '?'
    ]

    unique_chars = set(''.join(words))
    return sorted(list(unique_chars))

def char_one_hot_encode(character, char_vocab):
    """
    One-hot encodes a character based on the character vocabulary.
    """
    vector = [0] * len(char_vocab)
    if character in char_vocab:
        index = char_vocab.index(character)
        vector[index] = 1
    return vector

def encode_word(word, char_vocab):
    """
    Encodes a word into a sequence of one-hot vectors for each character.
    """
    return [char_one_hot_encode(char, char_vocab) for char in word]

# Example usage:
char_vocab = build_char_vocab()
encoded_word = encode_word("cat", char_vocab)

print(encoded_word)

word_list = ["cat", "bat", "dog", "bog", "mouse", "house", "goo", "too", "shine", "pine"]
pairs, labels = generate_dataset(word_list, 2, 20)


print('hello world')
print(pairs)
print(labels)