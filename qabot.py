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

if __name__ == "__main__":
    import sys
    import torch_utils
    
    sys.stdout.reconfigure(encoding='utf-8')

    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device_string}")
    device = torch.device(device_string)
    
    torch_utils.seed_everywhere(0)
    
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

    film_titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight", "The Godfather Part II", "12 Angry Men", "Schindler's List", "The Lord of the Rings: The Return of the King", "Pulp Fiction", "The Lord of the Rings: The Fellowship of the Ring", "The Good, the Bad and the Ugly", "Forrest Gump", "Fight Club", "The Lord of the Rings: The Two Towers", "Inception", "Star Wars: Episode V - The Empire Strikes Back", "The Matrix", "Goodfellas", "Spider-Man: Across the Spider-Verse", "One Flew Over the Cuckoo's Nest", "Se7en", "It's a Wonderful Life", "Seven Samurai", "The Silence of the Lambs", "Interstellar", "Saving Private Ryan", "City of God", "Life Is Beautiful", "The Green Mile", "Star Wars: Episode IV - A New Hope", "Terminator 2: Judgment Day", "Back to the Future", "Spirited Away", "The Pianist", "Psycho", "Parasite", "Oppenheimer", "Gladiator", "The Lion King", "Léon: The Professional", "American History X", "The Departed", "Whiplash", "The Prestige", "The Usual Suspects", "Grave of the Fireflies", "Casablanca", "Harakiri", "The Intouchables", "Modern Times", "Cinema Paradiso", "Once Upon a Time in the West", "Rear Window", "Alien", "City Lights", "Apocalypse Now", "Memento", "Django Unchained", "Indiana Jones and the Raiders of the Lost Ark", "WALL·E", "The Lives of Others", "Sunset Blvd.", "Paths of Glory", "Avengers: Infinity War", "The Shining", "The Great Dictator", "Witness for the Prosecution", "Spider-Man: Into the Spider-Verse", "Aliens", "American Beauty", "The Dark Knight Rises", "Inglourious Basterds", "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb", "Oldboy", "Coco", "Amadeus", "Toy Story", "Braveheart", "Das Boot", "Joker", "Avengers: Endgame", "Princess Mononoke", "Good Will Hunting", "Once Upon a Time in America", "Your Name.", "3 Idiots", "High and Low", "Singin' in the Rain", "Requiem for a Dream", "Capernaum", "Toy Story 3", "Come and See", "Star Wars: Episode VI - Return of the Jedi", "Eternal Sunshine of the Spotless Mind", "2001: A Space Odyssey", "The Hunt", "Reservoir Dogs", "Ikiru", "Lawrence of Arabia", "Citizen Kane", "M", "North by Northwest", "The Apartment", "Vertigo", "Double Indemnity", "Amélie", "Scarface", "Full Metal Jacket", "A Clockwork Orange", "Incendies", "Hamilton", "Heat", "Up", "To Kill a Mockingbird", "The Sting", "A Separation", "Indiana Jones and the Last Crusade", "Metropolis", "Die Hard", "L.A. Confidential", "Bicycle Thieves", "Snatch", "Like Stars on Earth", "Taxi Driver", "1917", "Downfall", "Dangal", "Top Gun: Maverick", "For a Few Dollars More", "Batman Begins", "Some Like It Hot", "The Kid", "The Wolf of Wall Street", "The Father", "Green Book", "All About Eve", "Judgment at Nuremberg", "The Truman Show", "Ran", "There Will Be Blood", "Casino", "Shutter Island", "Pan's Labyrinth", "Unforgiven", "The Sixth Sense", "Jurassic Park", "A Beautiful Mind", "The Treasure of the Sierra Madre", "Yojimbo", "No Country for Old Men", "Monty Python and the Holy Grail", "Kill Bill: Vol. 1", "The Great Escape", "The Thing", "Rashomon", "Finding Nemo", "The Elephant Man", "Chinatown", "Spider-Man: No Way Home", "V for Vendetta", "Gone with the Wind", "Raging Bull", "Dial M for Murder", "Howl's Moving Castle", "Lock, Stock and Two Smoking Barrels", "The Secret in Their Eyes", "Inside Out", "Prisoners", "Three Billboards Outside Ebbing, Missouri", "The Bridge on the River Kwai", "Trainspotting", "Fargo", "Warrior", "Gran Torino", "Catch Me If You Can", "My Neighbor Totoro", "Million Dollar Baby", "Klaus", "Children of Heaven", "Harry Potter and the Deathly Hallows: Part 2", "Blade Runner", "12 Years a Slave", "Before Sunrise", "The Gold Rush", "The Grand Budapest Hotel", "Ben-Hur", "Gone Girl", "On the Waterfront", "Barry Lyndon", "Hacksaw Ridge", "In the Name of the Father", "The General", "The Deer Hunter", "Wild Strawberries", "Memories of Murder", "The Third Man", "The Wages of Fear", "Wild Tales", "Sherlock Jr.", "Mad Max: Fury Road", "Dead Poets Society", "Mr. Smith Goes to Washington", "Monsters, Inc.", "How to Train Your Dragon", "Mary and Max", "Jaws", "The Seventh Seal", "Room", "The Big Lebowski", "Ford v Ferrari", "Tokyo Story", "Ratatouille", "Hotel Rwanda", "The Passion of Joan of Arc", "Rocky", "Logan", "Platoon", "Spotlight", "The Terminator", "Jai Bhim", "Before Sunset", "Rush", "Network", "Stand by Me", "The Best Years of Our Lives", "The Wizard of Oz", "Into the Wild", "La haine", "The Exorcist", "Pirates of the Caribbean: The Curse of the Black Pearl", "The Incredibles", "To Be or Not to Be", "My Father and My Son", "Groundhog Day", "The Grapes of Wrath", "Hachi: A Dog's Tale", "The Battle of Algiers", "The Handmaiden", "Amores Perros", "Rebecca", "Cool Hand Luke", "Pather Panchali", "The Sound of Music", "It Happened One Night", "The Iron Giant", "The 400 Blows", "The Help", "Persona", "Life of Brian", "Aladdin", "Drishyam",
    ]

    #input_texts = [
    #    "See spot run. Run spot run!"
    #]

    #input_texts = [ "Got the milk?" ]
    input_texts = [ "Got milk?" ]
    #input_texts = [ "Go buy milk?" ]
    #input_texts = [ "Dog" ]
    
    #obj_data = abo.load_objects(10)
    #input_texts = [abo.get_itemname_for_object(obj) for obj in obj_data]
    
    #input_texts = film_titles
    
    #[print(t) for t in input_texts]
    
    print(len(input_texts))
    
    #tokenizer = T.UTF8Tokenizer()
    #tokenizer = T.WordTokenizer()
    tokenizer = T.BPETokenizer()
    sequencer = dataset_utils.TextDatasetSequencer(tokenizer)
    
    MAX_SEQ_LEN = 16
    
    input_sequences, target_sequences = sequencer.load2(
        input_texts, 
        seq_len=MAX_SEQ_LEN,
    )
    
    show_inputs = True
    if show_inputs:
        for idx, (tokens, next_token) in enumerate(zip(input_sequences, target_sequences)):
            str = tokenizer.indices_to_text(tokens,hide_pad=False)
            target = tokenizer.indices_to_text(next_token,hide_pad=False)
            
            print(f"'{escape(str)}' => '{escape(target)}'")
    
    print(f"Num inputs: {len(input_sequences)}")
    #exit()
    
    # Hyperparameters
    BATCH_SIZE = 128
    NUM_TOKENS = tokenizer.vocab_size()
    epochs = 100
    embed_dim = 128
    num_heads = 2
    num_layers = 2
    dropout = 0.1
    
    do_training = True
    load_checkpoint = False
    checkpoint_path = "checkpoints/saved/tinygpt_checkpoint.best.top250filmtitles.pth"
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
    X = torch.tensor(input_sequences).to(device)
    Y = torch.tensor(target_sequences).to(device)
    tensor_dataset = TensorDataset(X, Y)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
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
    pad_idx = tokenizer.special_token_to_index(tokenizer.pad_token)
    
    if do_training:
        tt.train_text(model, dataloader, NUM_TOKENS, pad_idx, epochs=epochs)
    
    print("Training finished!")
    
    model.eval()
    with torch.no_grad():
        while True:

            print()
            prompt = input(">")[-MAX_SEQ_LEN-1:]
            tokens = tokenizer.text_to_indices(prompt)
            
            tokens.insert(0, sta_seq_idx)
            tokens = sequencer.pad(tokens, MAX_SEQ_LEN)
            
            # recon = tokenizer.indices_to_text(tokens,hide_pad=True)
            # print(f"{recon}")
            
            x = torch.tensor(tokens).unsqueeze(0).to(device)
            
            max_new_tokens = 20
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
            