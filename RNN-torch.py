import torch
import torch.nn as nn
import torch.optim as optim

# 1. SimpleRNN Class:
class SimpleRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, learning_rate=0.001):
        super(SimpleRNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Hidden Layer
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Set initial hidden state
        self.hidden_state = torch.zeros(1, 1, hidden_size)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# 3. Encoding and Decoding Mechanisms:
def encode(character, char_to_index):
    return torch.tensor([[char_to_index[character]]])

def decode(output_probs, index_to_char):
    max_index = torch.argmax(output_probs, dim=1).item()
    return index_to_char[max_index]

# 4. Sequence Extraction:
def moving_window(s, n, char_to_index):
    sequences = [s[i:i+n] for i in range(len(s)-n+1)]
    dataset = []
    for seq in sequences:
        input_seq = [char_to_index[char] for char in seq[:-1]]
        target_char = char_to_index[seq[-1]]
        dataset.append((input_seq, target_char))
    return dataset

def train_rnn(dataset, rnn, epochs=1000):
    for epoch in range(epochs):
        rnn.train()
        total_loss = 0
        for input_seq, target_char in dataset:
            input_tensor = torch.tensor([input_seq])
            target_tensor = torch.tensor([target_char])
            
            output, _ = rnn(input_tensor, rnn.hidden_state)
            loss = rnn.criterion(output, target_tensor)
            
            rnn.optimizer.zero_grad()
            loss.backward()
            rnn.optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss}")

# Main
    
# Define character mapping
characters = "abcdefghijklmnopqrstuvwxyz .,!'?\n"
char_to_index = {char: i for i, char in enumerate(characters)}
index_to_char = {i: char for i, char in enumerate(characters)}

# Sample text
text = """roses are red,
violets are blue,
in my head,
i love you!"""

window_size = 10
dataset = moving_window(text, window_size, char_to_index)

# Initialize the model
input_size = len(characters)
embedding_size = 30
hidden_size = 25
output_size = len(characters)

rnn = SimpleRNN(input_size, embedding_size, hidden_size, output_size)

# Train the RNN
train_rnn(dataset, rnn)

# Text generation example (can be expanded upon)
initial_text = "roses are "
generated = initial_text
for _ in range(50):
    input_seq = [char_to_index[char] for char in initial_text]
    input_tensor = torch.tensor([input_seq])
    output, _ = rnn(input_tensor, rnn.hidden_state)
    next_char = decode(output, index_to_char)
    generated += next_char
    initial_text = initial_text[1:] + next_char
print("Generated:", generated)
