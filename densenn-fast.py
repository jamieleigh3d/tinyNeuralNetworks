import numpy as np

class DenseNN:
    def __init__(self, input_size, embedding_size, hidden1_size, hidden2_size):
        # implement He initiation
        # * np.sqrt(2.0 / input_size) instead of * 0.01
        
        # weights for each layer, initialized randomly with He normalization
        self.w_emb = np.random.randn(input_size, embedding_size) * .01
        self.w1 = np.random.randn(embedding_size, hidden1_size) * .01
        self.w2 = np.random.randn(hidden1_size, hidden2_size) * .01
        self.w_out = np.random.randn(hidden2_size, 1) * 0.01
        
        # biases for each layer
        self.b_emb = np.zeros((1, embedding_size))
        self.b1 = np.zeros((1, hidden1_size))
        self.b2 = np.zeros((1, hidden2_size))
        self.b_out = np.zeros((1, 1))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def forward(self, x):
        # Assuming x is of shape (batch_size, 2 * 10 * 27)
        
        #self.x = x.reshape(batch_size, -1)  # Ensure 2D shape
        
        # for each layer:
        # calculate the values of dot(x, layer_weights) + layer_bias
        # with appropriate activation function
        
        self.h_emb = np.dot(x, self.w_emb)# + self.b_emb
        self.h1 = self.relu(np.dot(self.h_emb, self.w1))# + self.b1)
        self.h2 = self.relu(np.dot(self.h1, self.w2))# + self.b2)
        self.out = self.sigmoid(np.dot(self.h2, self.w_out))# + self.b_out)
        return self.out
        
    def backward(self, x, y, lr):
        # Assuming y is of shape (batch_size, 1)
        
        # Output layer gradient
        d_out = self.out - y
        d_w_out = np.dot(self.h2.T, d_out) / y.shape[0]
        d_b_out = np.sum(d_out, axis=0) / y.shape[0]
        
        # Hidden layer 2 gradient
        d_h2 = np.dot(d_out, self.w_out.T) * (self.h2 > 0)
        d_w2 = np.dot(self.h1.T, d_h2) / y.shape[0]
        d_b2 = np.sum(d_h2, axis=0) / y.shape[0]
        
        # Hidden layer 1 gradient
        d_h1 = np.dot(d_h2, self.w2.T) * (self.h1 > 0)
        d_w1 = np.dot(self.h_emb.T, d_h1) / y.shape[0]
        d_b1 = np.sum(d_h1, axis=0) / y.shape[0]
        
        # Embedding layer gradient
        d_h_emb = np.dot(d_h1, self.w1.T)
        d_w_emb = np.dot(x.T, d_h_emb) / y.shape[0]
        d_b_emb = np.sum(d_h_emb, axis=0) / y.shape[0]
        
        # Update weights and biases
        self.w_emb -= lr * d_w_emb
        self.b_emb -= lr * d_b_emb

        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1

        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2

        self.w_out -= lr * d_w_out        
        self.b_out -= lr * d_b_out

    def train_batch(self, x, y, lr=0.05):
        # Assuming x is of shape (batch_size, 2 * 10 * 27)
        # and y is of shape (batch_size, 1)
        
        pred = self.forward(x)
        self.backward(x, y, lr)
        loss = np.mean(np.square(pred - y))
        
        return loss

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
    
def train(words, val_words, nn, batch_size=32, epochs=100):
    data = create_dataset(words, 10)
    n = len(data)
    
    val_data = create_dataset(val_words, 10)
    val_n = len(val_data)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = n // batch_size
        
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            
            x_batch = []
            #x_words = f"[{batch_size}]"
            y_batch = []
            for i in range(start, end):
                for j in range(start, end):
                    x_batch.append(np.hstack([data[i], data[j]]))
                    rhyme = 1 if naive_rhyme_check(words[i], words[j]) else 0
                    #x_words += f"{words[i]}:{words[j]}=({rhyme}) "
                    y_batch.append(rhyme)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch).reshape(-1,1)
            #print(x_words)
            
            total_loss += nn.train_batch(x_batch, y_batch)
        
        average_loss = total_loss / num_batches
        
        if epoch % 10 == 0:
            
            x_batch = []
            y_batch = []
            #val_loss = 0
            
            for i in range(val_n):
                for j in range(val_n):
                    x_batch.append(np.hstack([val_data[i], val_data[j]]))
                    rhyme = 1 if naive_rhyme_check(val_words[i], val_words[j]) else 0
                    y_batch.append(rhyme)
                    
                    #val_loss += np.mean(np.square(pred - rhyme))
                    if epoch == epochs-10:
                        x = np.hstack([val_data[i], val_data[j]])
                        pred = nn.forward(x)
                        print(f"{val_words[i]} {val_words[j]}: {pred[0]:.2f} ({rhyme})")
            
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch).reshape(-1,1)
            
            #average_val_loss = val_loss / (val_n*val_n)
            
            pred = nn.forward(x_batch)
            batch_val_loss = np.mean(np.square(pred - y_batch))
            
            print(f"Epoch {epoch}, Loss: {average_loss}, Val_loss: {batch_val_loss}")


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

batch_size = 4
training_words = words[:batch_size]#[:int(0.9*len(words))]
validation_words = words[:batch_size]#[int(0.9*len(words)):]

nn = DenseNN(2 * 10 * 27, 10, 10, 10)
train(training_words, validation_words, nn, batch_size, 1000)

print("Done!")
