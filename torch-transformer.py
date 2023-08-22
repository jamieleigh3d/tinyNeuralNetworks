import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

import tokenization as T
import lrschedulers
from datasets import TextDataset, escape
import abo as abo

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=1024, embed_dim=64, num_heads=4, num_decoder_layers=2, dropout=0.1):
        super(TextTransformer, self).__init__()
        
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.pos_enc = self.positional_encoding(block_size, embed_dim)
        
        # self.transformer = nn.Transformer(
            # d_model=embed_dim, 
            # nhead=num_heads, 
            # num_encoder_layers=0,
            # num_decoder_layers=num_decoder_layers,
        # )
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        # Tie embedding weights with fc weights
        self.fc.weight = self.embedding.weight
        
        
    def forward(self, x, mask=None):
        emb = self.embedding(x)
        
        t = x.shape[1]
        positions = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        #pos_emb = self.pos_embedding(positions)
        pos_emb = self.pos_enc[:, :emb.size(1)].to(x.device)
        
        emb = self.dropout(emb + pos_emb)
        
        #TODO: Support masking
        x = self.transformer_decoder(emb, emb)
        # taking only the last token for prediction, [-1] preserving the dimensionality
        x = self.fc(x[:, [-1], :])
        return x
        
    def positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pos_enc = torch.empty(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        pos_enc = pos_enc.unsqueeze(0)
        return pos_enc

    # Modified from NanoGPT https://github.com/karpathy/nanoGPT/ Under MIT License
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=1, temperature=1.0, top_k=None, eos_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        #generated_tokens = []
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
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_token and idx_next.item() == eos_token:
                break

        return idx
        
    @staticmethod
    def create_masks(src, src_pad):
        """Create padding masks for the src sequence."""
        src_mask = (src != src_pad)
        
        #print(f"mask shape: {src_mask.shape}")
        return src_mask

def train_text(model, dataloader, NUM_TOKENS, pad_token, epochs=50, lr=0.001):
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            mask = model.create_masks(x, pad_token)
            outputs = model(x, mask)
            
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
    #input_texts = [ "Go buy milk?" ]
    
    obj_data = abo.load_objects()[:100]
    
    input_texts = [abo.get_itemname_for_object(obj) for obj in obj_data]
    
    #[print(t) for t in input_texts]
    
    print(len(input_texts))
    
    tokenizer = T.UTF8Tokenizer()
    dataset = TextDataset(tokenizer)
    
    MAX_SEQ_LEN = 32
    
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
    Y = torch.tensor(target_sequences).to(device)
    tensor_dataset = TensorDataset(X, Y)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    

    # Instantiate and train the model
    model = TextTransformer(
        vocab_size=NUM_TOKENS, 
        block_size=MAX_SEQ_LEN,
        embed_dim=64, 
        num_heads=4, 
        num_decoder_layers=4, 
        dropout=0.1
    ).to(device)
    print(X.shape)
    print(Y.shape)

    end_seq_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
    train_text(model, dataloader, NUM_TOKENS, pad_idx, epochs=50)
    
    print("Training finished!")
    
    model.eval()
    with torch.no_grad():
        while True:

            print()
            prompt = input(">")[:MAX_SEQ_LEN]
            tokens = tokenizer.text_to_indices(prompt)
            
            while len(tokens) < MAX_SEQ_LEN:
                tokens.insert(0, pad_idx)
                #tokens.append(pad_idx)
            
            recon = tokenizer.indices_to_text(tokens)
            x = torch.tensor(tokens).to(device)
            print(f"[{recon}]",end='')
            
            for i in range(10):
                #mask = model.create_masks(x.unsqueeze(0), pad_idx)
                #outputs = model(x.unsqueeze(0)).argmax(dim=-1).squeeze()
                #outputs_list = [outputs.cpu().tolist()]
                
                max_new_tokens = 20
                temperature = 0.01
                top_k = 3
                outputs = model.generate(
                    x.unsqueeze(0), 
                    max_new_tokens, 
                    temperature=temperature, 
                    top_k=top_k,
                    eos_token=end_seq_idx)
                
                outputs_list = outputs[0].tolist()[len(tokens):]
                
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
