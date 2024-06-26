import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torchvision.transforms.functional as F

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from PIL import Image

import torchaudio

from torch import nn

from datasets import load_dataset

from snac import SNAC

class AudioTokenizer():
    def __init__(self):
        self.tokenizer = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval()

    def flatten_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        res = torch.cat(codes, dim=-1)
        return res
    
    def unflatten_codes(self, flat: torch.Tensor, n_lists: int = 4) -> torch.Tensor:
        total = flat.shape[-1]
        divisor = (2 ** n_lists - 1)

        if total % divisor != 0:
            raise ValueError("n_lists not compatible with total size.")
        
        k: int = int (total / divisor)
        splits = [k * (2**i) for i in range(n_lists)]
        codes = torch.split(flat, split_size_or_sections=splits, dim=-1)
        return list(codes)
    
    def encode(self, audio):
        codes = []
        with torch.inference_mode():
            codes = self.tokenizer.encode(audio)
        return self.flatten_codes(codes)
    
    def decode(self, tokens):
        sequence = self.unflatten_codes(tokens)
        audio_hat = self.tokenizer.decode(sequence)

        return audio_hat
    
class ToTensor(nn.Module):
    def __init__(self) -> None:
        return

    def __call__(self, pic):
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ImageAudioTextDataset(Dataset):
    def __init__(self, audio_path: str, max_samples: int = None):
        self.audio_path = Path(audio_path)
        self.image_data = load_dataset("reach-vb/pokemon-blip-captions")
        self.image_transform = torch.nn.Sequential(
            Resize((216, 216)),
            ToTensor()
        )
        self.audio_tokenizer = AudioTokenizer()
        self.triples = self.create_triples(max_samples)

    def create_triples(self, max_samples: int) -> List[Dict[str, str]]:

        image_dataset = self.image_data["train"]

        audio_files = list(self.audio_path.glob('*.wav'))

        triples = []

        for index, pair in enumerate(image_dataset):

            if index < len(audio_files):

                init_audio_tensor = torchaudio.load(str(audio_files[index]))[0]
                init_audio_tensor = init_audio_tensor[None, :, :]
                audio_tensor = self.audio_tokenizer.encode(init_audio_tensor) 

                triples.append({
                    'image': self.image_transform(pair["image"]),
                    'caption': pair["text"],
                    'audio': audio_tensor
                })
        if index % 5 == 0 or index == 0:
            print(index)

        return triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.triples[idx]
        
        # Load and process image
        image_tensor = item["image"]

        # Load and process audio
        audio_tensor = item["audio"]

        return image_tensor, audio_tensor, item['text']

# Usage
dataset = ImageAudioTextDataset('/Users/samherring/Desktop/Projects/kandinsky/audio', max_samples=100)
image, audio, text = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Audio shape: {audio.shape}")
print(f"Caption: {text}")