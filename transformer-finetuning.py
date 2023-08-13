from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from torch.utils.data import DataLoader, Dataset

# Load pretrained model and tokenizer
model_name = "gpt2-medium"  # You can change this to "gpt2-small", "gpt2-medium", "gpt2-large", or "gpt2-xl" based on your needs
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# Training loop
texts = ["""I bought this for my husband who plays the piano.  
He is having a wonderful time playing these old hymns.  
The music  is at times hard to read because we think the 
book was published for singing from more than playing from.  
Great purchase though!
""",
"Cat sat on mat. Dog jumped over cat. Bird flew under bridge.",
"the quick brown fox jumped over the lazy dog",
"Artificial neural networks are a branch of machine learning models that are built using principles of neuronal organization discovered by connectionism in the biological neural networks constituting animal brains."

]
dataset = CustomDataset(tokenizer, texts, max_length=100)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

epochs = 30
model.train()
for epoch in range(epochs):  # You can change the number of epochs
    for input_ids, attention_masks in dataloader:
        outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")


model.eval()

# Function to generate text
def generate_text(prompt, max_length=50, num_return_sequences=1, temperature=1.0):
    """Generate text using a given prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True  # Enable sampling
    )
    
    generated_text = [tokenizer.decode(token_id) for token_id in output]
    return generated_text

# Test the function
prompt = "I bought this for"
generated_sentences = generate_text(prompt, 
                                    max_length=100, 
                                    num_return_sequences=1)

for i, sentence in enumerate(generated_sentences, 1):
    print(f"Generated {i}: {sentence}")

prompt = "cat sat"
generated_sentences = generate_text(prompt, 
                                    max_length=100, 
                                    num_return_sequences=1)

for i, sentence in enumerate(generated_sentences, 1):
    print(f"Generated {i}: {sentence}")

prompt = "the quick brown"
generated_sentences = generate_text(prompt, 
                                    max_length=100, 
                                    num_return_sequences=1)

for i, sentence in enumerate(generated_sentences, 1):
    print(f"Generated {i}: {sentence}")

prompt = "Artificial neural networks"
generated_sentences = generate_text(prompt, 
                                    max_length=100, 
                                    num_return_sequences=1)

for i, sentence in enumerate(generated_sentences, 1):
    print(f"Generated {i}: {sentence}")

#model.save_pretrained("/path/to/save/directory")
#tokenizer.save_pretrained("/path/to/save/directory")
