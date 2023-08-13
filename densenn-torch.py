import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Convert numpy arrays to torch tensors
def np2torch(arr):
    return torch.tensor(arr, dtype=torch.float32)

# Neural network model
class DenseNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden1_size, hidden2_size):
        super(DenseNN, self).__init__()

        # Linear layers
        self.fc_emb = nn.Linear(input_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc_out = nn.Linear(hidden2_size, 1)

        # Initialization with He normalization
        nn.init.kaiming_normal_(self.fc_emb.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc_out.weight)

    def forward(self, x):
        x = self.fc_emb(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))
        return x

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

# Training function
def train(words, val_words, model, batch_size=32, epochs=100):
    # [all the pre-processing functions and dataset creation remain the same]
    data = create_dataset(words, 10)
    n = len(data)
    
    val_data = create_dataset(val_words, 10)
    val_n = len(val_data)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # ... [Data loading remains the same]
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
        
            model.train()
            optimizer.zero_grad()

            x_batch_t = np2torch(x_batch)
            y_batch_t = np2torch(y_batch)
            
            outputs = model(x_batch_t)
            loss = criterion(outputs, y_batch_t)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                x_val_batch_t = np2torch(x_batch)
                y_val_batch_t = np2torch(y_batch)
                val_outputs = model(x_val_batch_t)
                val_loss = criterion(val_outputs, y_val_batch_t)
                
                print(f"Epoch {epoch}, Loss: {loss.item()}, Val_loss: {val_loss.item()}")


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

batch_size = 32
training_words = words[:batch_size]#[:int(0.9*len(words))]
validation_words = words[:batch_size]#[int(0.9*len(words)):]

# Using GPU if available
device_string = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_string}")
device = torch.device(device_string)

model = DenseNN(2 * 10 * 27, 50, 20, 10).to(device)
train(training_words, validation_words, model, batch_size, 1000)
