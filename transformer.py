from collections import Counter
import torch
import torch.nn as nn
import string
import nltk
from nltk.tokenize import WordPunctTokenizer
import re

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
        
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


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

def tokenize_text(text):
    # This pattern recognizes words, punctuation, and spaces
    #pattern = r"[\w]+|[.,!?;]|[\s]"
    #return re.findall(pattern, text)
    
    
    tokenizer = WordPunctTokenizer()
    return tokenizer.tokenize(text)


rawdata = "Cat sat on mat. Dog jumped over cat. Bird flew under bridge."
rawdata = "The quick brown fox jumped over the lazy dog."
#rawdata = "Artificial neural networks are a branch of machine learning models that are built using principles of neuronal organization discovered by connectionism in the biological neural networks constituting animal brains."
#rawdata = "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!"

padding_token = "<|PAD|>"
end_of_sequence_token = "<|ENDOFTEXT|>"

# Using GPU if available
device_string = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_string}")
device = torch.device(device_string)

data = rawdata

print(data)

# Tokenize the input string
tokens = tokenize_text(data)
tokens.append(end_of_sequence_token)

# Tokenize the data
#tokens = data.split()
print(f"Token count: {len(tokens)}")
print(tokens)
#quit()

# Build vocabulary
vocab = Counter(tokens)
# Give it a negative count to ensure it's added last
vocab[padding_token] = -1
word2idx = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# Convert tokens to tensor indices
token_indices = [word2idx[word] for word in tokens]
print(f"word2idx {word2idx}")
# Create sequences of tokens
min_sequence_length = 8
# "The cat sat on" -> "the"
sequence_length = max(len(tokens),min_sequence_length)

input_sequences = []
for s in range(0,sequence_length):
    for i in range(len(token_indices) - s):
        sequence = token_indices[i:i+s+1]
        
        # Pad the sequence
        while len(sequence) < sequence_length:
            sequence.insert(0, word2idx[padding_token])
        
        input_sequences.append(sequence)

for seq in input_sequences:
    str = ""
    for idx in seq:
        str += idx2word[idx] + ' '
#    print (f"{str} ({seq})")


print(f"input seq len: {len(input_sequences)}")
input_sequences = torch.tensor(input_sequences).to(device)

# Model hyperparameters
d_model = 16
nhead = 1
num_encoder_layers = 1
num_decoder_layers = 1
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)

# Hyperparameters
epochs = 100
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

warmup_steps = 10
total_steps = epochs
scheduler = WarmupThenDecaySchedule(optimizer, warmup_steps, total_steps)

total, trainable = model.count_parameters()
print(f"Total Parameters: {total:,}")
print(f"Trainable Parameters: {trainable:,}")

def predict_next_word(model, tokens, word2idx, idx2word, sequence_length, temperature=0.0):
    model.eval()

    # Convert tokens to indices
    token_indices = [word2idx[token] for token in tokens if token in word2idx]

    # Pad the token_indices list with the index of the padding_token until it's of length sequence_length
    while len(token_indices) < sequence_length:
        token_indices.insert(0, word2idx[padding_token])

    # Convert list of indices to tensor, send to device, and add a batch dimension
    input_tensor = torch.tensor(token_indices).to(device).unsqueeze(0)

    # Get model output
    logits = model(input_tensor)

    predicted_index = -1
    
    if temperature == 0.0:
        # Determine index of most probable next word
        predicted_index = torch.argmax(logits[0, -1]).item()
    else:
        # Apply temperature to logits
        probs = nn.Softmax(dim=0)(logits[0, -1] / temperature)

        # Sample from the probabilities
        predicted_index = torch.multinomial(probs, 1).item()

    return idx2word[predicted_index]


logging_interval = 1
test_start_length = 1

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
        phrase_tokens = tokens[:test_start_length]
        phrase = " ".join(phrase_tokens)
        print(f"[{phrase}]",end="")
        for i in range(len(tokens)+4):
            next_token = predict_next_word(model, phrase_tokens, word2idx, idx2word, sequence_length)
            phrase_tokens.append(next_token)
            phrase = " ".join(phrase_tokens)
            print(f"{next_token} ",end="")
            if next_token == end_of_sequence_token:
                break;
        if phrase_tokens == tokens:
            break
        print(f"\nEpoch: {epoch+1}/{epochs}, Loss: {loss.item()}, Learning Rate: {scheduler.get_lr()[0]}")

