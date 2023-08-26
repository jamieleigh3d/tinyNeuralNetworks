import torch
from torch.utils.data import Sampler, Dataset
import tokenization as T
from collections import OrderedDict
from random import shuffle

def escape(input_string):
    # Replace tab characters with '\\t'
    replaced_tabs = input_string.replace('\t', '\\t')
    
    # Replace newline characters with '\\n'
    replaced_newlines = replaced_tabs.replace('\n', '\\n')
    
    return replaced_newlines

class TextDatasetSequencer():
    def __init__(self, tokenizer, pad_left=True):
        self.tokenizer = tokenizer
        self.pad_left = pad_left
    
    def pad(self, tokens, seq_len):
        pad_idx = self.tokenizer.special_token_to_index(self.tokenizer.pad_token)
        pad_length = max(0, seq_len - len(tokens))
        
        if self.pad_left:
            return [pad_idx] * pad_length + tokens
        else:
            return tokens + [pad_idx] * pad_length

    
    def load2(self, input_texts, seq_len):
        tokenizer = self.tokenizer
        
        tokenizer.build_vocab(input_texts)
        training_tokens = tokenizer.texts_to_indices(input_texts)
        
        pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
        tokenizer.wrap(training_tokens)
        
        input_sequences = []
        target_sequences = []
        for t in training_tokens:
            
            for s in range(1,min(seq_len,len(t))):
                #Start in the first character
                i=0
                sequence = t[i:i+s]
                next = t[i:i+s+1]
                
                sequence = self.pad(sequence, seq_len)
                next = self.pad(next, seq_len)
                
                input_sequences.append(sequence)
                target_sequences.append(next)
        

        return input_sequences, target_sequences
    
    def load(self, input_texts, seq_len, use_padding=True):
        tokenizer = self.tokenizer
        
        tokenizer.build_vocab(input_texts)
        training_tokens = tokenizer.texts_to_indices(input_texts)
        
        pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
        tokenizer.wrap(training_tokens)
        
        input_sequences = []
        target_sequences = []
        for t in training_tokens:
            
            # Initial variable length padding until seq_len
            if use_padding:
                for s in range(1,min(seq_len,len(t))):
                    #Start in the first character
                    i=0
                    sequence = t[i:i+s]
                    next = t[i:i+s+1]
                    
                    # Pad the sequence
                    while len(sequence) < seq_len:
                        sequence.insert(0, pad_idx)
                        #sequence.append(pad_idx)
                    
                    while len(next) < seq_len:
                        next.insert(0, pad_idx)
                        #next.append(pad_idx)

                    input_sequences.append(sequence)
                    target_sequences.append(next)
            
            if len(t) > seq_len:
                # Sliding a window through t if t > seq_len
                # Fixed seq_len
                s = seq_len
                # Start on the ith character
                for i in range(len(t) - s):
                    sequence = t[i : i+s]
                    next = t[i+1 : i+s+1]
                    input_sequences.append(sequence)
                    target_sequences.append(next)
                
            if use_padding:
                s = seq_len
                # Start on the ith character
                for i in range(max(0,len(t) - s),len(t)):
                    sequence = t[i : i+s]
                    next = t[i+1 : i+s+1]
                    
                    
                    # Pad the sequence after the EOS
                    while len(sequence) < seq_len:
                        sequence.append(pad_idx)
                    
                    while len(next) < seq_len:
                        next.append(pad_idx)
                        
                    input_sequences.append(sequence)
                    target_sequences.append(next)

        return input_sequences, target_sequences

    def generate_mask(self, seq, pad_token):
        mask = [True if token != pad_token else False for token in seq]
        return mask


# jagged_dataset = dataset_utils.JaggedDataset(input_sequences, target_sequences, device)
# bucket_batch_sampler = dataset_utils.BucketBatchSampler(input_sequences, BATCH_SIZE)
# dataloader = DataLoader(
    # jagged_dataset, 
    # batch_size=1, 
    # batch_sampler=bucket_batch_sampler, 
    # shuffle=False, #the sampler shuffles for us
# )


class JaggedDataset(Dataset):
    def __init__(self, inputs, targets=None, device='cpu'):
        assert targets is None or len(inputs) == len(targets), "inputs and targets must be the same length"
        self.inputs = inputs
        self.targets = targets
        self.device = device
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, index):
        if self.targets is None:
            return torch.tensor(self.inputs[index]).to(self.device)
        else:
            return torch.tensor(self.inputs[index]).to(self.device), torch.tensor(self.targets[index]).to(self.device)

# From https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13
class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, len(p)))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., 
        # batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i

