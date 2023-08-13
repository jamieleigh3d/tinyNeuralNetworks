import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RhymeNN(nn.Module):
    def __init__(self, input_dim):
        super(RhymeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


rawwords = [
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

word_count = int(0.1*len(rawwords))
words = rawwords[:word_count]
val_words = rawwords[word_count:]

def naive_rhyme_check(word1, word2):
    if len(word1) == 3 or len(word2) == 3:
        return word1[-2:] == word2[-2:]
    else:
        return word1[-3:] == word2[-3:]

# Find unique characters and add a space for padding
vocab = list(set('abcdefghijklmnopqrstuvwxyz '))
char_to_index = {char: i for i, char in enumerate(vocab)}
index_to_char = {i: char for char, i in char_to_index.items()}

# Define a function to one-hot encode a word
def one_hot_encode(word, char_to_index, max_length):
    encoding = np.zeros((max_length, len(char_to_index)))
    for i, char in enumerate(word):
        encoding[i, char_to_index[char]] = 1
    return encoding

# Find the maximum length of word in our dataset
max_length = max([len(word) for word in rawwords])

# Create pairs and labels
num_words = len(words)
pairs = []
for i in range(num_words):
    for j in range(num_words):
        pairs.append((words[i], words[j]))
labels = [1 if naive_rhyme_check(pair[0], pair[1]) else 0 for pair in pairs]

print(f"max_length: {max_length} training num_words: {num_words} training pairs: {len(pairs)}")
print(f"validation num_words: {len(val_words)} validation pairs: {len(val_words)*len(val_words)}")

X_data = [(
    one_hot_encode(pair[0], char_to_index, max_length),
    one_hot_encode(pair[1], char_to_index, max_length)
) for pair in pairs]

# Using GPU if available
device_string = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_string}")
device = torch.device(device_string)

input_dim = max_length * len(vocab)

resume = False

model = RhymeNN(input_dim).to(device)

if resume:
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert our dataset to PyTorch tensors
X_data_tensor = [(torch.FloatTensor(x[0]).to(device).view(-1), torch.FloatTensor(x[1]).to(device).view(-1)) for x in X_data]
y_data_tensor = torch.FloatTensor(labels).to(device).view(-1, 1)

epochs = 30
log_interval = 1 #5

for epoch in range(epochs):
    total_loss = 0
    for i, (x1, x2) in enumerate(X_data_tensor):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = criterion(outputs, y_data_tensor[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % log_interval == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(X_data_tensor):.4f}')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoint.pth')

torch.save(model.state_dict(), 'rhyme_nn_model_params.pth')

# You can then use the model to predict if two new words rhyme
# by passing their one-hot encodings to the model.

def predict_rhyme(word1, word2, model):
    # One-hot encode the words
    x1_encoded = one_hot_encode(word1, char_to_index, max_length)
    x2_encoded = one_hot_encode(word2, char_to_index, max_length)

    # Convert to PyTorch tensors
    x1_tensor = torch.FloatTensor(x1_encoded).to(device).view(-1)
    x2_tensor = torch.FloatTensor(x2_encoded).to(device).view(-1)

    # Get the prediction
    with torch.no_grad():  # Disable gradient computation
        output = model(x1_tensor, x2_tensor)
        # Convert the output to a probability score
        predicted_prob = output.item()

    # If the probability is more than 0.5, the words are predicted to rhyme
    if predicted_prob > 0.5:
        return True
    else:
        return False

# Test the function with some example words
errors = 0
print(f"Checking training set... ({num_words*num_words} pairs)")
for i in range(num_words):
    for j in range(num_words):
        word1 = words[i]
        word2 = words[j]
        result = predict_rhyme(word1, word2, model)
        rhyme = naive_rhyme_check(word1, word2)
        if result != rhyme:
            errors += 1
            #print(f"{word1} {word2}: {'1' if result else '0'} ({rhyme})")

val_errors = 0
num_val_words = len(val_words)
print(f"Checking validation set... ({num_val_words*num_val_words} pairs)")
for i in range(num_val_words):
    for j in range(num_val_words):
        word1 = val_words[i]
        word2 = val_words[j]
        result = predict_rhyme(word1, word2, model)
        rhyme = naive_rhyme_check(word1, word2)
        if result != rhyme:
            val_errors += 1
            #print(f"{word1} {word2}: {result} ({rhyme})")

total_pairs = num_words * num_words
accuracy = 100.0 * (total_pairs - errors) / total_pairs
print(f"Training Errors: {errors} / {total_pairs} (Accuracy {accuracy:.2f}%)")

total_val_pairs = num_val_words * num_val_words
val_accuracy = 100.0 * (total_val_pairs - val_errors) / total_val_pairs
print(f"Validation Errors: {val_errors} / {total_val_pairs} (Accuracy {val_accuracy:.2f}%)")
