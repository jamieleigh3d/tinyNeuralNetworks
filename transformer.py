import torch
import torch.nn.functional as F
from x_transformers import TransformerWrapper, Decoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import tokenization as T
import lrschedulers

def escape(input_string):
    # Replace tab characters with '\\t'
    replaced_tabs = input_string.replace('\t', '\\t')
    
    # Replace newline characters with '\\n'
    replaced_newlines = replaced_tabs.replace('\n', '\\n')
    
    return replaced_newlines

def generate_mask(batch, pad_token):
    mask = [[1 if token != pad_token else 0 for token in seq] for seq in batch]
    tensor = torch.tensor(mask, device=batch.device, dtype=bool)
    return tensor

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
    
    #input_texts = [ "Got milk?" ]

    # Initialize tokenizer and add special tokens
    tokenizer = T.UTF8Tokenizer()
    tokenizer.build_vocab(input_texts)

    MAX_SEQ_LEN = 8
    
    start_seq_idx = tokenizer.special_token_to_index(tokenizer.sta_token)
    end_seq_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
    training_tokens = tokenizer.texts_to_indices(input_texts)
    
    for t in training_tokens:
        t.insert(0, start_seq_idx)
        t.append(end_seq_idx)
    str_list = tokenizer.indices_to_texts(training_tokens)
    for str in str_list:
        print(escape(str))
    print()
    
    input_sequences = []
    target_sequences = []
    for t in training_tokens:
        #for s in range(1,MAX_SEQ_LEN+1):
        s = MAX_SEQ_LEN
        for i in range(len(t) - s):
            sequence = t[i:i+s]
            next = t[i+s:i+s+MAX_SEQ_LEN]
            
            # Pad the sequence
            while len(sequence) < MAX_SEQ_LEN:
                sequence.insert(0, pad_idx)
            while len(next) < MAX_SEQ_LEN:
                next.append(pad_idx)
            
            input_sequences.append(sequence)
            target_sequences.append(next)

    str_list = tokenizer.indices_to_texts(input_sequences)
    #print(target_sequences)
    target = tokenizer.indices_to_texts(target_sequences)
    for idx, str in enumerate(str_list):
        targ = target[idx]
        print(f"'{escape(str)}' => '{escape(targ)}'")
        
    
    print(f"Num inputs: {len(input_sequences)}")
    #exit()
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 1000
    NUM_TOKENS = tokenizer.vocab_size()
    dim = 32
    depth = 4
    heads = 2
    
    # Prepare data for DataLoader
    X = torch.tensor(input_sequences)
    Y = torch.tensor(target_sequences)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = TransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        attn_layers=Decoder(
            dim = dim,
            depth = depth,
            heads = heads
        )
    ).to(device)
    

    LR = 1 # large because scheduler will update it
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = lrschedulers.NoamLR(optimizer, d_model=dim, warmup_steps=1000)

    
    # Initialize matplotlib
    plt.ion()   # Turn on interactive mode
    fig, ax = plt.subplots()
    losses = []
    learning_rates = []

    # Training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            mask = generate_mask(x, pad_idx)
            outputs = model(x, mask=mask)
            #print(f"x: {x.shape} outputs: {outputs.shape} y: {y.shape}")
            loss = F.cross_entropy(outputs.view(-1, NUM_TOKENS), y.view(-1))
            #loss = loss * mask
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        
        learning_rate = optimizer.param_groups[0]["lr"]
        learning_rates.append(learning_rate*1000)
        
        # Sample print reconstructed text
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                mask = generate_mask(x, pad_idx)
                sample_output = model(x[:1], mask=mask).argmax(dim=-1)
                prompt = tokenizer.indices_to_text(x[0].cpu().tolist())
                reconstructed_text = tokenizer.indices_to_text(sample_output[0].cpu().tolist())
                print(f"Epoch {epoch + 1}, Loss {avg_epoch_loss:.4f}, LR {learning_rate:.6f}: '{escape(prompt)}' => '{escape(reconstructed_text)}'")
            # Update the plot for each epoch
            ax.clear()
            ax.plot(losses, label='Training loss')
            #ax.plot(learning_rates, label='Learning rate')
            ax.legend()
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            plt.draw()
            plt.pause(0.1)  # Pause to update the plot

    print("Training completed!")
    
    
    model.eval()
    with torch.no_grad():
        while True:

            #prompt = "Got milk"
            prompt = input(">")[:MAX_SEQ_LEN]
            tokens = tokenizer.text_to_indices(prompt)
        #    tokens.insert(0, start_seq_idx)
            while len(tokens) < MAX_SEQ_LEN:
                tokens.insert(0, pad_idx)
            
            recon = tokenizer.indices_to_text(tokens)
            print(f"[{recon}]",end='')
            
            x = torch.tensor(tokens).unsqueeze(0).to(device)

            for i in range(40):
                mask = generate_mask(x, pad_idx)
                outputs = model(x, mask=mask).argmax(dim=-1).squeeze()
                outputs_list = outputs.cpu().tolist()
                recon = tokenizer.indices_to_text(outputs_list)
                print(recon, end='')
                
                # Check for <EOS>
                if end_seq_idx in outputs_list:
                    break
                # Cycle for next round
                x = outputs.unsqueeze(0)
        
    plt.ioff()
    plt.show()