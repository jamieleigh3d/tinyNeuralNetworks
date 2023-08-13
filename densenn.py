import numpy as np

class DenseNN:
    def __init__(self, input_size, embedding_size, hidden1_size, hidden2_size):
        self.w_emb = np.random.randn(input_size, embedding_size) * 0.01
        self.w1 = np.random.randn(embedding_size, hidden1_size) * 0.01
        self.w2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.w_out = np.random.randn(hidden2_size, 1) * 0.01
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def forward(self, x):
        self.x = x.reshape(1, -1)  # Ensure 2D shape
        self.h_emb = np.dot(self.x, self.w_emb)
        self.h1 = self.relu(np.dot(self.h_emb, self.w1))
        self.h2 = self.relu(np.dot(self.h1, self.w2))
        self.out = self.sigmoid(np.dot(self.h2, self.w_out))
        return self.out
    
    def backward(self, y, lr=0.05):
        d_out = self.out - y
        d_w_out = np.dot(self.h2.T, d_out)
        
        d_h2 = np.dot(d_out, self.w_out.T) * (self.h2 > 0)
        d_w2 = np.dot(self.h1.T, d_h2)
        
        d_h1 = np.dot(d_h2, self.w2.T) * (self.h1 > 0)
        d_w1 = np.dot(self.h_emb.T, d_h1)
        
        d_h_emb = np.dot(d_h1, self.w1.T)
        d_w_emb = np.dot(self.x.T, d_h_emb)
        
        self.w_emb -= lr * d_w_emb
        self.w1 -= lr * d_w1
        self.w2 -= lr * d_w2
        self.w_out -= lr * d_w_out

def create_dataset(words, input_size):
    chars = 'abcdefghijklmnopqrstuvwxyz '
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    def encode_word(word):
        one_hot = np.zeros((input_size, 27))
        for i, char in enumerate(word):
            one_hot[i, char_to_int[char]] = 1
        return one_hot.flatten()
    
    data = []
    for word in words:
        encoded_word = encode_word(word.ljust(input_size))
        data.append(encoded_word)
    return np.array(data)

def naive_rhyme_check(word1, word2):
    if len(word1) == 3 or len(word2) == 3:
        return word1[-2:] == word2[-2:]
    else:
        return word1[-3:] == word2[-3:]

def train(words, val_words, nn, epochs=100):
    data = create_dataset(words, 10)
    val_data = create_dataset(val_words, 10)
    n = len(data)
    val_n = len(val_data)
    for epoch in range(epochs):
        loss = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    x = np.hstack([data[i], data[j]])
                    y = np.array([[1 if naive_rhyme_check(words[i], words[j]) else 0]])
                    pred = nn.forward(x)
                    nn.backward(y, lr=0.01)
                    loss += np.mean(np.square(pred - y))
        if epoch % 10 == 0:
            
            val_loss = 0
            for i in range(val_n):
                for j in range(val_n):
                    
                    y = 1 if naive_rhyme_check(val_words[i], val_words[j]) else 0
                    
                    if i != j:
                        x = np.hstack([val_data[i], val_data[j]])
                        pred = nn.forward(x)
                        val_loss += np.mean(np.square(pred - y))
                        if y == 1:
                            print(val_words[i] +' '+val_words[j] +': ' + "{:.6f}".format(pred[0,0]) +' ('+str(y)+')')

            print(f"Epoch {epoch}, Loss: {loss}, Val_loss: {val_loss}")


words = [
    "cat", "bat", 
    "dog", "bog", 
    "rat", "fog", 
    "log", "sat", 
    "mat", "pat", 
    "dot", "hot",
    "moon", "spoon",
    "sun", "fun",
    "day", "play",
    "night", "light",
    "star", "car",
    "blue", "shoe",
    "tree", "bee",
    "rain", "train",
    "snow", "glow",
    "road", "toad",
    "seat", "beat",
    "man", "tan",
    "fly", "sky",
    "wave", "cave",
    "hill", "grill",
    "fan", "van",
    "jam", "clam",
    "bear", "hair",
    "food", "mood",
    "boat", "coat",
    "fish", "dish",
    "bed", "red",
    "cow", "plow",
    "tie", "pie",
    "ring", "sing",
    "horse", "course",
    "whale", "pale",
    "duck", "truck",
    "far", "bar",
    "mice", "rice",
    "pen", "hen",
    "phone", "cone",
    "seat", "feet",
    "wall", "ball",
    "bat", "hat",
    "cake", "lake",
    "time", "lime",
    "plan", "tan",
    "mug", "bug",
    "door", "floor",
    "tail", "pail",
    "mail", "sail",
    "nose", "rose",
    "pool", "cool",
    "read", "lead",
    "rock", "sock",
    "dream", "cream",
    "card", "yard",
    "leaf", "beef",
    "burn", "turn",
    "steer", "peer",
    "stream", "beam",
    "float", "coat",
    "prize", "eyes",
    "street", "fleet",
    "loud", "cloud",
    "brace", "face",
    "bliss", "kiss",
    "pray", "stay",
    "flame", "game",
    "glare", "stare",
    "snore", "bore",
    "plight", "kite",
    "steak", "cake",
    "fleece", "peace",
    "clue", "blue",
    "stone", "phone"
]

batch_size = 16
training_words = words[:batch_size]#[:int(0.9*len(words))]
validation_words = words[:batch_size]#[int(0.9*len(words)):]

nn = DenseNN(2 * 10 * 27, 40, 20, 10)
train(training_words, validation_words, nn, 10000)

print("Hello, World!")
