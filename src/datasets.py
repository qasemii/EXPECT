import os
import json
import pandas as pd

from torch.utils.data import Dataset

class Multi30k(Dataset):
    
    def __init__(self, split="train", languages=None):
        """
        Args:
            split: ["train", "val", "test_2016_flickr", "test_2017_flickr", "test_2017_mscoco"]
            languages: ["en", "de", "fr", "cs"]. If None, loads all available.
        """
        self.data_path = "data/raw/multi30k"
        self.compile_file = f"data/raw/multi30k/compiled_data_{split}.json"
        self.split = split
        self.data_types = ["text", "image"]
        
        if languages is None:
            self.languages = ["en", "de", "fr", "cs"]
        else:
            self.languages = languages
        
        if os.path.exists(self.compile_file):
            print(f"Loading compiled data for split: {split}")
            with open(self.compile_file, "r") as f:
                self.data = json.load(f)
        else:
            print(f"Compiling data for split: {split}")
            self.data = self.compile_data(save_compiled=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {
            "img_path": self.data[idx]["img_path"],
            "image_id": self.data[idx]["image_id"],
        }
        
        # Add text for each language
        for lang in self.languages:
            if f"text_{lang}" in self.data[idx]:
                item[f"text_{lang}"] = self.data[idx][f"text_{lang}"]
        
        # Add default "text" field (English if available, otherwise first available language)
        if "text_en" in self.data[idx]:
            item["text"] = self.data[idx]["text_en"]
        else:
            # Use first available language
            for lang in self.languages:
                if f"text_{lang}" in self.data[idx]:
                    item["text"] = self.data[idx][f"text_{lang}"]
                    break
        
        return item
    
    def compile_data(self, save_compiled=False):
        """Compiles multilingual captions with image paths"""
        compiled_data = []
        
        # Load captions for each language
        captions_by_lang = {}
        for lang in self.languages:
            caption_file = f"{self.data_path}/{self.split}.{lang}"
            if os.path.exists(caption_file):
                with open(caption_file, "r", encoding="utf-8") as f:
                    captions_by_lang[lang] = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(captions_by_lang[lang])} captions for {lang}")
            else:
                print(f"Warning: Caption file not found: {caption_file}")
        
        # Check that all languages have the same number of captions
        if captions_by_lang:
            num_samples = len(next(iter(captions_by_lang.values())))
            for lang, captions in captions_by_lang.items():
                if len(captions) != num_samples:
                    print(f"Warning: Language {lang} has {len(captions)} captions, expected {num_samples}")
        else:
            print("Error: No caption files found!")
            return []
        
        # Create data entries
        # Multi30k typically has 5 captions per image, so we group them
        # Image IDs are typically sequential or can be inferred from line numbers
        for idx in range(num_samples):
            # Calculate image_id (Multi30k uses Flickr30k images)
            # Typically, each image has 5 captions, so image_id = idx // 5
            # But this depends on the dataset format - adjust as needed
            image_id = idx
            
            # Construct image path
            # Note: You may need to adjust this based on actual image naming convention
            img_path = f"{self.data_path}/images/flickr30k_images/{image_id}.jpg"
            
            entry = {
                "image_id": image_id,
                "img_path": img_path,
                "split": self.split,
                "caption_idx": idx
            }
            
            # Add captions for each language
            for lang, captions in captions_by_lang.items():
                entry[f"text_{lang}"] = captions[idx]
            
            compiled_data.append(entry)
        
        print(f"Compiled {len(compiled_data)} samples for split {self.split}")
        
        # Save compiled data
        if save_compiled:
            os.makedirs(os.path.dirname(self.compile_file), exist_ok=True)
            with open(self.compile_file, "w", encoding="utf-8") as f:
                json.dump(compiled_data, f, ensure_ascii=False, indent=2)
            print(f"Saved compiled data to {self.compile_file}")
        
        return compiled_data
    
    def get_all_data(self):
        return self.data
    
    def get_attribute(self, attribute_name):
        return [x.get(attribute_name) for x in self.data]
    
    def get_language_captions(self, language):
        """Get all captions for a specific language"""
        return [x.get(f"text_{language}") for x in self.data]