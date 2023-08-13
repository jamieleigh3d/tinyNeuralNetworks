import numpy as np

class SimpleRNN:
    def __init__(self, input_size, embedding_size, hidden_size, output_size, learning_rate=0.001):
        # Initialize weights and biases
        self.embedding_weights = np.random.randn(input_size, embedding_size) * 0.01
        self.hidden_weights = np.random.randn(embedding_size, hidden_size) * 0.01
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.01
        
        self.hidden_biases = np.zeros(hidden_size)
        self.output_biases = np.zeros(output_size)
        
        # Hidden state initialization
        self.hidden_state = np.zeros(hidden_size)

        # Learning rate
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0) * 1.0
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward(self, input_vector):
        self.input_vector = input_vector
        # Embedding Layer
        self.embedded_vector = np.dot(input_vector, self.embedding_weights)
        
        # Hidden Layer
        self.hidden_state_before_activation = np.dot(self.embedded_vector, self.hidden_weights) + self.hidden_biases
        self.hidden_state = self.relu(self.hidden_state_before_activation)
        
        # Output Layer
        self.output_logits = np.dot(self.hidden_state, self.output_weights) + self.output_biases
        self.output_probs = self.softmax(self.output_logits)
        
        return self.output_probs

    def compute_loss(self, true_labels):
        # Cross entropy loss
        return -np.sum(true_labels * np.log(self.output_probs))

    def backpropagate(self, true_labels):
        # Gradients for output layer
        doutput = self.output_probs - true_labels
        
        doutput_weights = np.outer(self.hidden_state, doutput)
        doutput_biases = doutput
        
        dhidden_state = np.dot(self.output_weights, doutput)
        dhidden_state_before_activation = dhidden_state * self.relu_derivative(self.hidden_state_before_activation)
        
        dhidden_weights = np.outer(self.embedded_vector, dhidden_state_before_activation)
        dhidden_biases = dhidden_state_before_activation
        
        dembedded_vector = np.dot(self.hidden_weights, dhidden_state_before_activation)
        dembedding_weights = np.outer(self.input_vector, dembedded_vector)

        # Update weights and biases
        self.embedding_weights -= self.learning_rate * dembedding_weights
        self.hidden_weights -= self.learning_rate * dhidden_weights
        self.output_weights -= self.learning_rate * doutput_weights
        self.hidden_biases -= self.learning_rate * dhidden_biases
        self.output_biases -= self.learning_rate * doutput_biases

def scaled_dot_product_attention(query, key, value):
    matmul_qk = np.dot(query, key.T)
    d_k = query.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # Apply softmax to get the weights
    attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
    
    output = np.dot(attention_weights, value)
    return output

def encode(character):
    char_to_index = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
        'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15,
        'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
        'y': 24, 'z': 25, ' ': 26, '.': 27, ',': 28, "'": 29, '!': 30, '\n': 31, '"': 32, '?': 33
    }
    
    # Create a zero vector of size 32
    one_hot_vector = np.zeros(34)
    
    # Set the appropriate position to 1
    one_hot_vector[char_to_index[character]] = 1

    return one_hot_vector

def decode(output_probs):
    index_to_char = {
        0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
        8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p',
        16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
        24: 'y', 25: 'z', 26: ' ', 27: '.', 28: ',', 29: "'", 30: '!', 31: '\n', 32: '"', 33: '?'
    }

    # Get the index of the maximum probability
    max_index = np.argmax(output_probs)

    return index_to_char[max_index]

def moving_window(s, n):
    if n <= 0:
        return []
    return [s[i:i+n] if i+n <= len(s) else s[i:] + s[:(i+n)%len(s)] for i in range(len(s))]


# Testing the function
print(decode(encode('a')))  # Expected: [1, 0, 0, ...]

# # Sample text

text = """in the town of spreeville, down by the blue hill,
lived a jolly young cat named jack, who sat still.
he wore a large hat, tilted just so,
and played a green flute, that let melodies flow.

on sunny day mornings, with the sky oh so clear,
jack's tunes would dance, bringing everyone near.
birds, bees, and the trees, even fishes would hear,
and all would come hopping, or flying, or hopping from rear.

"why do you play, dear jack with your flute?"
asked a curious squirrel, in a bright orange suit.
jack winked and he twirled, his tail up in loop,
"for the joy and the fun, and the kiwi fruit!"

because in spreeville, when jack's tune was spun,
the kiwi trees danced, and dropped fruits by the ton.
so, the town rejoiced, in laughter and fun,
thanks to jack's melodies, under the sun.

so here's to the tales, of places unseen,
where cats play flutes, and squirrels are keen.
in the world of seuss, where wonders convene,
every story's a journey, every moment a dream.
"""
#text = """roses are red,
#violets are blue,
#in my head,
#i love you!
#"""

#text="hello world!"
# Initialize the dataset
dataset = []

window_size = 20 # Taking 2 characters as input and predicting the next one, thus window size is 3

windows = moving_window(text, window_size)

for window in windows:
    input_chars = window[:-1]  # All characters except the last
    true_char = window[-1]  # The last character
    
    # Construct the input vector by encoding the characters and concatenating
    input_vector = np.concatenate([encode(c) for c in input_chars])
    dataset.append((input_vector, encode(true_char)))

# Let's see the first pair (for clarity)
print(dataset[0])

# Testing
input_size = 34*(window_size-1)
embedding_size = 20
hidden_size = 25
output_size = 34

rnn = SimpleRNN(input_size, embedding_size, hidden_size, output_size, learning_rate=0.01)

# An example training loop
for epoch in range(1000):
    total_loss = 0
    # This loop assumes you have a dataset with input-output pairs
    for input_vector, true_label in dataset:  
        output_probs = rnn.forward(input_vector)
        loss = rnn.compute_loss(true_label)
        total_loss += loss
        rnn.backpropagate(true_label)
    if epoch % 10 == 0:
        newtext = 'in the town of night,'

        # Create a list to store the initial sequence of characters based on the window_size
        initial_chars = [newtext[i] for i in range(window_size - 1)]
        
    # Print the initial sequence
        for ch in initial_chars:
            print(ch, end="")

    # Generate new characters
        for i in range(len(text)):
        # Create the input vector by encoding and concatenating the initial characters
            input_vector = np.concatenate([encode(ch) for ch in initial_chars])

        # Get the predicted character
            new_char = decode(rnn.forward(input_vector))
            print(new_char, end="")

        # Shift characters for next iteration
            initial_chars = initial_chars[1:] + [new_char]
        print()
        print(f"Epoch: {epoch}, Loss: {total_loss}")
        