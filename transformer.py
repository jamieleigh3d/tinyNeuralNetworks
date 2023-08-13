from collections import Counter
import torch
import torch.nn as nn
import string

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer(embedded, embedded)
        return self.fc(output)

class WarmupThenDecaySchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupThenDecaySchedule, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_scale = float(step) / float(max(1, self.warmup_steps))
        else:
            decay_steps = max(1, step - self.warmup_steps)
            lr_scale = (self.total_steps - decay_steps) / self.total_steps
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]

def filter_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation and keep only alphabets and spaces
    filtered_text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    
    return filtered_text

rawdata = "Cat sat on mat. Dog jumped over cat. Bird flew under bridge."
#rawdata = "the quick brown fox jumped over the lazy dog"
#rawdata = "Artificial neural networks (ANNs, also shortened to neural networks (NNs) or neural nets) are a branch of machine learning models that are built using principles of neuronal organization discovered by connectionism in the biological neural networks constituting animal brains."

padding_token = "PAD"
end_of_sequence = "QUIT"

# Using GPU if available
device_string = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_string}")
device = torch.device(device_string)

data = filter_text(rawdata) + ' ' + end_of_sequence
print(data)

# Tokenize the data
tokens = data.split()
print(f"Token count: {len(tokens)}")

# Build vocabulary
vocab = Counter(tokens)
# Give it a negative count to ensure it's added last
vocab[padding_token] = -1
word2idx = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# Convert tokens to tensor indices
token_indices = [word2idx[word] for word in tokens]

# Create sequences of tokens
min_sequence_length = 8
# "The cat sat on" -> "the"
sequence_length = 8 #max(len(tokens),min_sequence_length)

input_sequences = []
for s in range(1,sequence_length+1):
    for i in range(len(token_indices) - s):
        sequence = token_indices[i:i+s+1]
        
        # Pad the sequence
        while len(sequence) < sequence_length + 1:  # +1 because we're predicting the next token
            sequence.insert(0, word2idx[padding_token])
        
        input_sequences.append(sequence)

print(input_sequences)

print(f"input seq len: {len(input_sequences)}")
input_sequences = torch.tensor(input_sequences).to(device)

# Model hyperparameters
d_model = 32
nhead = 1
num_encoder_layers = 2
num_decoder_layers = 2
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

# Hyperparameters
epochs = 1000
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

warmup_steps = 10
total_steps = epochs
scheduler = WarmupThenDecaySchedule(optimizer, warmup_steps, total_steps)


def predict_next_word(model, tokens, word2idx, idx2word, sequence_length):
    model.eval()

    # Convert tokens to indices
    token_indices = [word2idx[token] for token in tokens if token in word2idx]

    # Pad the token_indices list with the index of the padding_token until it's of length sequence_length
    while len(token_indices) < sequence_length:
        token_indices.insert(0, word2idx[padding_token])

    # Convert list of indices to tensor, send to device, and add a batch dimension
    input_tensor = torch.tensor(token_indices).to(device).unsqueeze(0)

    # Get model output
    output = model(input_tensor)

    # Determine index of most probable next word
    predicted_index = torch.argmax(output[0, -1]).item()

    return idx2word[predicted_index]

logging_interval = 1
test_length = 2

for epoch in range(epochs):
    model.train()
    for sequence in input_sequences:
        optimizer.zero_grad()

        inputs = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        loss.backward()
        optimizer.step()
    
    scheduler.step()
    if (epoch+1) % logging_interval == 0:
        model.eval()
        phrase = ' '.join(tokens[:test_length])
        print(f"{phrase} => ",end="")
        for i in range(len(tokens)):
            phrase_tokens = phrase.split()#[-sequence_length:]

            next_word = predict_next_word(model, phrase_tokens, word2idx, idx2word, sequence_length)
            phrase = f"{phrase} {next_word}"
            print(f"{next_word} ",end="")
            if next_word == end_of_sequence:
                break;
        if phrase == data:
            break
        print(f"\nEpoch: {epoch+1}/{epochs}, Loss: {loss.item()}, Learning Rate: {scheduler.get_lr()[0]}")

