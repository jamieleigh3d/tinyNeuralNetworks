
import tokenization as T

def escape(input_string):
    # Replace tab characters with '\\t'
    replaced_tabs = input_string.replace('\t', '\\t')
    
    # Replace newline characters with '\\n'
    replaced_newlines = replaced_tabs.replace('\n', '\\n')
    
    return replaced_newlines

class TextDataset():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def load(self, input_texts, seq_len=8):
        tokenizer = self.tokenizer
        
        tokenizer.build_vocab(input_texts)
        training_tokens = tokenizer.texts_to_indices(input_texts)
        
        start_seq_idx = tokenizer.special_token_to_index(tokenizer.sta_token)
        end_seq_idx = tokenizer.special_token_to_index(tokenizer.eos_token)
        pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
        for t in training_tokens:
            #t.insert(0, start_seq_idx)
            t.append(end_seq_idx)
        str_list = tokenizer.indices_to_texts(training_tokens)
        for str in str_list:
            print(escape(str))
        print()
        
        input_sequences = []
        input_masks = []
        target_sequences = []
        for t in training_tokens:
            #for s in range(3,seq_len+1):
            s = seq_len
            for i in range(len(t) - s):
                sequence = t[i:i+s]
                next = t[i+s]
                
                # Pad the sequence
                while len(sequence) < seq_len:
                    sequence.insert(0, pad_idx)
                #while len(next) < seq_len:
                #    next.append(pad_idx)
                
                mask = self.generate_mask(sequence, pad_idx)
                
                input_sequences.append(sequence)
                input_masks.append(mask)
                target_sequences.append(next)

        return input_sequences, input_masks, target_sequences

    def generate_mask(self, seq, pad_token):
        mask = [True if token != pad_token else False for token in seq]
        #tensor = torch.tensor(mask, device=batch.device, dtype=bool)
        return mask