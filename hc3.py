# https://huggingface.co/datasets/Hello-SimpleAI/HC3
# https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl -> data/hc3-all.jsonl

import json
import os
import requests
from tqdm import tqdm

class JsonParser:
    def __init__(self, filepath='data/hc3-all.jsonl'):
        # Create the data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Check if the file exists. If not, download it.
        if not os.path.exists(filepath):
            print("Downloading HC3...")
            url = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8K
            
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print("Error, something went wrong with the download.")
            else:
                print("File downloaded successfully.")
        
        self.filepath = filepath
        self.data = []
        self._load_data()
        
    def _load_data(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line in file:
                self.data.append(json.loads(line))

    def len(self):
        return len(self.data)
                
    def get_question(self, index):
        return self.data[index]['question']
    
    def get_human_answers(self, index):
        return self.data[index]['human_answers']
    
    def get_chatgpt_answers(self, index):
        return self.data[index]['chatgpt_answers']
    
    def get_source(self, index):
        return self.data[index]['source']

# Usage example
if __name__ == "__main__":

    import sys
    
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = JsonParser('data/hc3-all.jsonl')
    question = parser.get_question(0)  # get the question from the first line
    human_answers = parser.get_human_answers(0)  # get human answers from the first line
    chatgpt_answers = parser.get_chatgpt_answers(0)  # get chatgpt answers from the first line
    source = parser.get_source(0)  # get the source from the first line

    print("Question:", question)
    print("Human Answers:", human_answers)
    print("ChatGPT Answers:", chatgpt_answers)
    print("Source:", source)
