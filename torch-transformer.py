import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tokenization as T
import lrschedulers
from datasets import TextDataset, escape

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

def train_text(model, dataloader, NUM_TOKENS, epochs=50, lr=0.0001):
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x, xm, y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(x)
            
            loss = F.cross_entropy(outputs.view(-1, NUM_TOKENS), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss}")

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

    #input_texts = [
    #    "See spot run. Run spot run!"
    #]

    input_texts = [ "Got milk?" ]

    #data, labels = generate_random_data()
    tokenizer = T.UTF8Tokenizer()
    dataset = TextDataset(tokenizer)
    
    MAX_SEQ_LEN = 8
    
    input_sequences, input_masks, target_sequences = dataset.load(input_texts, seq_len=MAX_SEQ_LEN)
    
    print(input_sequences)
    print(target_sequences)
    str_list = tokenizer.indices_to_texts(input_sequences, input_masks)
    target = tokenizer.indices_to_text(target_sequences)
    for idx, str in enumerate(str_list):
        targ = target[idx]
        print(f"'{escape(str)}' => '{escape(targ)}'")
    
    print(f"Num inputs: {len(input_sequences)}")
    
    BATCH_SIZE = 16
    NUM_TOKENS = tokenizer.vocab_size()
    
    #exit()
    # Prepare data for DataLoader
    X = torch.tensor(input_sequences).to(device)
    Xm = torch.tensor(input_masks, dtype=bool).to(device)
    Y = torch.tensor(target_sequences).to(device)
    tensor_dataset = TensorDataset(X, Xm, Y)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    

    # Instantiate and train the model
    model = TextTransformer(vocab_size=NUM_TOKENS).to(device)
    print(X.shape)
    print(Y.shape)

    train_text(model, dataloader, NUM_TOKENS)
    
    print("Training finished!")
    
    end_seq_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
    model.eval()
    with torch.no_grad():
        while True:

            print()
            prompt = input(">")[:MAX_SEQ_LEN]
            tokens = tokenizer.text_to_indices(prompt)
            
            while len(tokens) < MAX_SEQ_LEN:
                tokens.insert(0, pad_idx)
            
            recon = tokenizer.indices_to_text(tokens)
            x = torch.tensor(tokens).unsqueeze(0).to(device)
            print(f"[{recon}]",end='')
            
            for i in range(20):
                mask = None#generate_mask(x, pad_idx)
                outputs = model(x).argmax(dim=-1).squeeze()
                outputs_list = [outputs.cpu().tolist()]
                recon = tokenizer.indices_to_text(outputs_list)
                print(recon, end='')
                
                # Check for <EOS>
                if end_seq_idx in outputs_list:
                    print("\nEOS")
                    break
                # Cycle for next round
                #print(tokens)
                tokens = tokens[1:MAX_SEQ_LEN]
                tokens.append(outputs_list[0])
                x = torch.tensor(tokens).unsqueeze(0).to(device)
