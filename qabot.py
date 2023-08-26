import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm, trange

import tokenization as T
import lrschedulers
import dataset_utils
from dataset_utils import escape
import abo as abo
import text_transformer as tt
import torch_utils

class qa_sequencer():
    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        assert seq_len > 0, "seq_len must be 1 or more"

    def pad(self, tokens, pad_left=True):
        pad_idx = self.tokenizer.special_token_to_index(self.tokenizer.pad_token)
        pad_length = max(0, self.seq_len - len(tokens))
        
        if pad_left:
            return [pad_idx] * pad_length + tokens
        else:
            return tokens + [pad_idx] * pad_length

    def wrap_qa(self, q_tokens, a_tokens):
        tokenizer = self.tokenizer
        start_token = tokenizer.special_token_to_index(tokenizer.sta_token)
        end_token = tokenizer.special_token_to_index(tokenizer.eos_token)
        user_idx = tokenizer.special_token_to_index(tokenizer.user_token)
        bot_idx = tokenizer.special_token_to_index(tokenizer.bot_token)
        
        # First truncate, -3/-1 to leave room for user/bot and start/eos tokens
        q_tokens[:] = q_tokens[:self.seq_len-3]
        a_tokens[:] = a_tokens[:self.seq_len-1]
    
        # <S><User>Question<Bot>
        q_tokens[:] = [start_token] + [user_idx] + q_tokens + [bot_idx]
        
        #Answer<E>
        a_tokens[:] = a_tokens + [end_token]

    def parse_qa(self, qa_pairs):
        
        input_tokens = []
        target_tokens = []
        for qa in qa_pairs:
            q_text = qa['question']
            a_text = qa['answer']
            
            q_tokens = self.tokenizer.text_to_indices(q_text)
            a_tokens = self.tokenizer.text_to_indices(a_text)
            
            self.wrap_qa(q_tokens, a_tokens)
            
            for idx in range(len(a_tokens)):
                str = q_tokens + a_tokens[:idx+1]
                print(str)
                # +1 so we capture the next token
                str[:] = str[-(self.seq_len+1):]
                x_tokens = self.pad(str[:-1])
                y_tokens = self.pad(str[-self.seq_len:])
                
                input_tokens.append(x_tokens)
                target_tokens.append(y_tokens)
            
        return input_tokens, target_tokens

if __name__ == "__main__":
    import sys
    
    
    sys.stdout.reconfigure(encoding='utf-8')

    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)
    
    torch_utils.seed_everywhere(0)
    
    # Dummy dataset
    qa_pairs = [
        {
            "question": "What movie, based on a Stephen King novella, is set in a prison and revolves around the friendship of Andy Dufresne and Red?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which film features Tim Robbins and Morgan Freeman in leading roles, navigating life in a penitentiary?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which iconic film is known for its quote, 'Get busy living, or get busy dying'?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "In which movie does the protagonist manage to escape from prison through a tunnel he dug over decades?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "What is the name of the film where the main character is falsely accused of murdering his wife and her lover?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which movie, directed by Francis Ford Coppola, introduces us to the Corleone crime family?",
            "answer": "The Godfather"
        },
        {
            "question": "Al Pacino, Robert Duvall, and Marlon Brando starred together in which iconic mafia film?",
            "answer": "The Godfather"
        },
        {
            "question": "Which film's famous line is, 'I'm gonna make him an offer he can't refuse'?",
            "answer": "The Godfather"
        },
        {
            "question": "In which film does a movie producer find the head of his prized horse in his bed?",
            "answer": "The Godfather"
        },
        {
            "question": "Which classic movie begins with the line, 'I believe in America'?",
            "answer": "The Godfather"
        },
        {
            "question": "Which movie features Heath Ledger's Oscar-winning portrayal of the Joker?",
            "answer": "The Dark Knight"
        },
        {
            "question": "In which sequel to 'Batman Begins' does Batman face off against a villain who wants to create chaos in Gotham?",
            "answer": "The Dark Knight"
        },
        {
            "question": "'Why so serious?' is a famous line from which superhero film?",
            "answer": "The Dark Knight"
        },
        {
            "question": "Which Christopher Nolan-directed film includes a dramatic scene involving a flipped semi-truck in the middle of a city street?",
            "answer": "The Dark Knight"
        },
        {
            "question": "In which film does Batman have to choose between saving Harvey Dent or Rachel Dawes from the Joker's trap?",
            "answer": "The Dark Knight"
        },
        {
            "question": "Which sequel is unique in that it serves both as a prequel and a continuation of the original, with scenes featuring a young Vito Corleone?",
            "answer": "The Godfather Part II"
        },
        {
            "question": "Robert De Niro won an Academy Award for his role in which film about the rise of a crime family?",
            "answer": "The Godfather Part II"
        },
        {
            "question": "Which film portrays the early life of Vito Corleone in New York City while also following his son Michael's expansion and tightened grip on the family crime syndicate?",
            "answer": "The Godfather Part II"
        },
        {
            "question": "Which classic film is set almost entirely in a single jury deliberation room?",
            "answer": "12 Angry Men"
        },
        {
            "question": "In which movie does Henry Fonda play a juror who tries to convince the others that there's a reasonable doubt about the guilt of the accused?",
            "answer": "12 Angry Men"
        }
    ]
    
    qa_pairs_val = [
        {
            "question": "Which Stephen King adaptation is set in a prison?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which film has Tim Robbins and Morgan Freeman in a penitentiary?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which movie quotes, 'Get busy living, or get busy dying'?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which film has a protagonist escaping prison via a tunnel?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which movie's lead is accused of killing his wife and lover?",
            "answer": "The Shawshank Redemption"
        },
        {
            "question": "Which Coppola film centers around the Corleone family?",
            "answer": "The Godfather"
        },
        {
            "question": "Which mafia film stars Pacino, Duvall, and Brando?",
            "answer": "The Godfather"
        },
        {
            "question": "Which film says, 'An offer he can't refuse'?",
            "answer": "The Godfather"
        },
        {
            "question": "Which movie has a horse head in a producer's bed?",
            "answer": "The Godfather"
        },
        {
            "question": "Which film starts with, 'I believe in America'?",
            "answer": "The Godfather"
        },
        {
            "question": "Which movie features Ledger's Joker?",
            "answer": "The Dark Knight"
        },
        {
            "question": "Which 'Batman Begins' sequel has Batman versus Joker?",
            "answer": "The Dark Knight"
        },
        {
            "question": "From which film is the line, 'Why so serious?'?",
            "answer": "The Dark Knight"
        },
        {
            "question": "Which Nolan film has a flipped semi-truck?",
            "answer": "The Dark Knight"
        },
        {
            "question": "In which film must Batman choose between Dent and Dawes?",
            "answer": "The Dark Knight"
        },
        {
            "question": "Which sequel shows a young Vito Corleone?",
            "answer": "The Godfather Part II"
        },
        {
            "question": "De Niro won an Oscar for which crime family film?",
            "answer": "The Godfather Part II"
        },
        {
            "question": "Which film shows Vito's early life and Michael's rise?",
            "answer": "The Godfather Part II"
        },
        {
            "question": "Which film is mostly in a jury room?",
            "answer": "12 Angry Men"
        },
        {
            "question": "Which film has Fonda as a doubting juror?",
            "answer": "12 Angry Men"
        }
    ]

    
    qa_pairs2 = [
        {
            "question": "Hello?",
            "answer": "World!"
        },
        {
            "question": "Hi?",
            "answer": "There."
        },
        {
            "question": "Sentient?",
            "answer": "Soon..."
        },
    ]
    #obj_data = abo.load_objects(10)
    #input_texts = [abo.get_itemname_for_object(obj) for obj in obj_data]
    
    
    MAX_SEQ_LEN = 256
    
    tokenizer = T.UTF8Tokenizer()
    #tokenizer = T.WordTokenizer()
    #tokenizer = T.BPETokenizer()
    
    sequencer = qa_sequencer(tokenizer, MAX_SEQ_LEN)
    
    input_sequences, target_sequences = sequencer.parse_qa(qa_pairs)

    val_input_sequences, val_target_sequences = sequencer.parse_qa(qa_pairs_val)


    print(len(input_sequences))
    
    
    show_inputs = True
    if show_inputs:
        for idx, (tokens, next_tokens) in enumerate(zip(input_sequences, target_sequences)):
            #print(f"'{tokens}' => '{next_tokens}'")
            str = tokenizer.indices_to_text(tokens,hide_pad=False)
            target = tokenizer.indices_to_text(next_tokens,hide_pad=False)
            
            print(f"'{escape(str)}' => '{escape(target)}'")
    
    print(f"Num inputs: {len(input_sequences)}")
    #exit()
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_TOKENS = tokenizer.vocab_size()
    epochs = 200
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    
    do_training = True
    load_checkpoint = False
    checkpoint_path = "checkpoints/tinygpt_checkpoint.best.pth"
    #checkpoint_path = "checkpoints/saved/tinygpt_checkpoint.best.top250filmtitles.pth"
    #checkpoint_path = "tinygpt_checkpoint.char.500titles.e64.h8.l8.len64.pth"
    # BATCH_SIZE = 256
    # NUM_TOKENS = tokenizer.vocab_size()
    # epochs = 100
    # embed_dim = 64
    # num_heads = 8
    # num_layers = 8
    # dropout = 0.1
    
    print(f"NUM_TOKENS: {NUM_TOKENS}")

    # Prepare data for DataLoader
    X = torch.tensor(input_sequences)
    Y = torch.tensor(target_sequences)
    tensor_dataset = TensorDataset(X, Y)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    vX = torch.tensor(input_sequences)
    vY = torch.tensor(target_sequences)
    v_tensor_dataset = TensorDataset(vX, vY)
    val_dataloader = DataLoader(v_tensor_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Instantiate and train the model
    model = tt.TextTransformer(
        vocab_size=NUM_TOKENS, 
        block_size=MAX_SEQ_LEN,
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        num_decoder_layers=num_layers, 
        dropout=dropout
    ).to(device)
    
    if load_checkpoint:
        epoch = model.load(checkpoint_path)
        print(f"Loaded model {checkpoint_path} at epoch {epoch}")

    print(f"Learnable parameters: {model.learnable_params():,} Total: {model.total_params():,}")
    
    sta_seq_idx = tokenizer.special_token_to_index(tokenizer.sta_token)
    end_seq_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
    user_idx = tokenizer.special_token_to_index(tokenizer.user_token)
    bot_idx = tokenizer.special_token_to_index(tokenizer.bot_token)
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
    if do_training:
        tt.train_text(
            model, 
            dataloader, 
            NUM_TOKENS, 
            pad_idx, 
            val_dataloader=val_dataloader, 
            device=device, 
            epochs=epochs
        )
    
    print("Training finished!")
    
    model.eval()
    with torch.no_grad():
        while True:

            print()
            prompt = input(">")[-MAX_SEQ_LEN-2:]
            tokens = tokenizer.text_to_indices(prompt)
            
            tokens = [sta_seq_idx] + [user_idx] + tokens + [bot_idx]
            tokens = sequencer.pad(tokens, MAX_SEQ_LEN)
            
            # recon = tokenizer.indices_to_text(tokens,hide_pad=True)
            # print(f"{recon}")
            
            x = torch.tensor(tokens).unsqueeze(0).to(device)
            
            max_new_tokens = 200
            temperature = 1.0
            top_k = 5
            outputs = model.generate(
                x, 
                max_new_tokens, 
                temperature=temperature, 
                top_k=top_k,
                eos_token=end_seq_idx,
                pad_token=pad_idx,
            )
            
            outputs_list = outputs[0].tolist()
            
            recon = tokenizer.indices_to_text(outputs_list,hide_pad=True)
            print(f"]{recon}")
            