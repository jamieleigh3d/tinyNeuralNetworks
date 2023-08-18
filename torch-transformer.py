import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tokenization as T
import lrschedulers
from datasets import TextDataset

# 1. Data Processing
# (For simplicity, we'll use random data here; replace this with your own dataset)

def generate_random_data(num_samples=1000, seq_length=10, vocab_size=20):
    data = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples,))
    return data, labels


# 2. Transformer Model
class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_encoder_layers=2):
        super(TextTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer.encoder(x)
        x = self.fc(x[:, -1, :]) # taking only the last token for prediction
        return x

# 3. Training Loop
def train(model, data, labels, epochs=50, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":

    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)
    
    # Dummy dataset
    input_texts = [
        "hello world!",
        "NLTK is a leading platform for building Python programs.",
        "See spot run. Run spot run!",
        "My # is 123-456-7890. Got that?",
        "Hello!!!? Are you there??",
        "this is a test\nwith a newline\tand a tab",
        "Got milk?",
    ]

    # input_texts = [
        # "See spot run. Run spot run!"
    # ]

    input_texts = [ "Got milk?" ]

    #data, labels = generate_random_data()
    tokenizer = T.UTF8Tokenizer()
    dataset = TextDataset(tokenizer)
    
    MAX_SEQ_LEN = 8
    
    input_sequences, input_masks, target_sequences2 = dataset.load(input_texts, seq_len=MAX_SEQ_LEN)
    target_sequences = []
    for t in target_sequences2:
        target_sequences.append(t[:1][0])
    
    print(len(input_sequences))
    
    BATCH_SIZE = 16
    print(target_sequences)
    # Prepare data for DataLoader
    X = torch.tensor(input_sequences).to(device)
    Xm = torch.tensor(input_masks, dtype=bool).to(device)
    Y = torch.tensor(target_sequences).to(device)
    tensor_dataset = TensorDataset(X, Xm, Y)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    

    # Instantiate and train the model
    model = TextTransformer(vocab_size=20).to(device)
    print(X.shape)
    print(Y.shape)

    train(model, X, Y)
    
    print("Training finished!")
    
    end_seq_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
    model.eval()
    with torch.no_grad():
        while True:

            #prompt = "Got milk"
            print()
            # -1 due to start token
            prompt = input(">")[:MAX_SEQ_LEN]
            tokens = tokenizer.text_to_indices(prompt)
            #tokens.insert(0, start_seq_idx)
            while len(tokens) < MAX_SEQ_LEN:
                tokens.insert(0, pad_idx)
            
            recon = tokenizer.indices_to_text(tokens)
            print(f"[{recon}]",end='')
            
            x = torch.tensor(tokens).unsqueeze(0).to(device)

            for i in range(1):
                mask = None#generate_mask(x, pad_idx)
                outputs = model(x).argmax(dim=-1).squeeze()
                outputs_list = [outputs.cpu().tolist()]
                recon = tokenizer.indices_to_text(outputs_list)
                print(recon)#, end='')
                
                # Check for <EOS>
                if end_seq_idx in outputs_list:
                    print("\nEOS")
                    break
                # Cycle for next round
                x = outputs.unsqueeze(0)
