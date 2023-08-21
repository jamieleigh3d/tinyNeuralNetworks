import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import tokenization as T
import lrschedulers
from datasets import TextDataset, escape

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=1024, embed_dim=64, num_heads=4, num_decoder_layers=2):
        super(TextTransformer, self).__init__()
        
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, 
            nhead=num_heads, 
            num_decoder_layers=num_decoder_layers
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
        # TODO: Tie embedding weights with fc weights
        
    def forward(self, x):
        #TODO: Add mask parameter
        emb = self.embedding(x)
        
        t = x.shape[1]
        positions = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.pos_embedding(positions)
        
        emb = emb + pos_emb
        
        #TODO: Add dropout layer here
        
        x = self.transformer.encoder(emb)
        x = self.fc(x[:, [-1], :]) # taking only the last token for prediction, [-1] preserving the dimensionality
        return x

    # Modified from NanoGPT https://github.com/karpathy/nanoGPT/ Under MIT License
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=1, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            #idx = torch.cat((idx, idx_next), dim=1)

        return idx_next
        
def train_text(model, dataloader, NUM_TOKENS, epochs=50, lr=0.001):
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x, xm, y) in enumerate(dataloader):
            optimizer.zero_grad()
            #TODO: support mask xm
            outputs = model(x)
            
            loss = F.cross_entropy(outputs.view(-1, NUM_TOKENS), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss}")


if __name__ == "__main__":
    import sys
    
    sys.stdout.reconfigure(encoding='utf-8')

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

    #input_texts = [ "Got the milk?" ]
    #input_texts = [ "Got milk?" ]
    input_texts = [ "Go buy milk?" ]
    
    print(len(input_texts))
    
    #data, labels = generate_random_data()
    tokenizer = T.UTF8Tokenizer()
    dataset = TextDataset(tokenizer)
    
    MAX_SEQ_LEN = 8
    
    input_sequences, input_masks, target_sequences = dataset.load(input_texts, seq_len=MAX_SEQ_LEN)
    
    for idx, (tokens, next_token) in enumerate(zip(input_sequences, target_sequences)):
        str = tokenizer.indices_to_text(tokens)
        target = tokenizer.indices_to_text(next_token)
        
        print(f"'{escape(str)}' => '{escape(target)}'")
    
    print(f"Num inputs: {len(input_sequences)}")
    #exit()
    
    BATCH_SIZE = 128
    NUM_TOKENS = tokenizer.vocab_size()
    
    
    # Prepare data for DataLoader
    X = torch.tensor(input_sequences).to(device)
    Xm = torch.tensor(input_masks, dtype=bool).to(device)
    Y = torch.tensor(target_sequences).to(device)
    tensor_dataset = TensorDataset(X, Xm, Y)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    

    # Instantiate and train the model
    model = TextTransformer(vocab_size=NUM_TOKENS, block_size=MAX_SEQ_LEN).to(device)
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
            x = torch.tensor(tokens).to(device)
            print(f"[{recon}]",end='')
            
            for i in range(20):
                mask = None#generate_mask(x, pad_idx)
                # outputs = model(x.unsqueeze(0)).argmax(dim=-1).squeeze()
                # outputs_list = [outputs.cpu().tolist()]
                
                #outputs = beam_search(model, x)
                #outputs = beam_search_recursive(model, x)
                #outputs_list = [outputs[0][-1].cpu().tolist()]
                
                
                max_new_tokens = 1
                temperature = 0.01
                top_k = 3
                outputs = model.generate(x.unsqueeze(0), max_new_tokens, temperature=temperature, top_k=top_k)
                outputs_list = outputs[0].tolist()
                
                recon = tokenizer.indices_to_text(outputs_list)
                print(recon, end='')
                
                # Check for <EOS>
                if end_seq_idx in outputs_list:
                    #print("\nEOS")
                    break
                # Cycle for next round
                #print(tokens)
                tokens = tokens[1:MAX_SEQ_LEN]
                tokens.append(outputs_list[0])
                x = torch.tensor(tokens).to(device)
