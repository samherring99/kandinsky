import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torchvision.transforms.functional as F

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
import torch.optim as optim
from PIL import Image

import torchaudio

import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import DataLoader

from snac import SNAC
from transformers import GPT2Model, GPT2Config, AutoTokenizer

from tqdm import tqdm

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
    def __init__(self, audio_path: str):
        self.audio_path = Path(audio_path)
        self.image_data = load_dataset("reach-vb/pokemon-blip-captions")
        self.image_transform = torch.nn.Sequential(
            Resize((128, 128)),
            ToTensor()
        )
        self.audio_tokenizer = AudioTokenizer()
        self.triples = self.create_triples()

    def create_triples(self) -> List[Dict[str, str]]:

        image_dataset = self.image_data["train"]

        audio_files = list(self.audio_path.glob('*.wav'))

        triples = []

        for index, pair in enumerate(image_dataset):

            if index < len(audio_files):

                if index % 10 == 0 or index == 0:
                    print("Preparing batch #" + str(index))

                init_audio_tensor = torchaudio.load(str(audio_files[index]))[0]
                init_audio_tensor = init_audio_tensor[None, :, :]

                batch_size, feature_dim,  seq_length, = init_audio_tensor.shape
                # 100548 is max audio tensor length
                padding = torch.full((init_audio_tensor.shape[0], init_audio_tensor.shape[1], 100548 - seq_length), 
                             0, 
                             dtype=init_audio_tensor.dtype, 
                             device=init_audio_tensor.device)
                
                padded_tensor = torch.cat([init_audio_tensor, padding], dim=2)

                audio_tensor = self.audio_tokenizer.encode(padded_tensor)

                triples.append({
                    'image': self.image_transform(pair["image"]),
                    'text': pair["text"],
                    'audio': audio_tensor
                })

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
    
class ImageEmbedding(nn.Module):
    def __init__(self, input_channels=3, image_size=128, embedding_dim=192): # 768
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
class ImageEmbeddingLoss(nn.MSELoss): # Module 
    def __init__(self, n_ctx=1024, embedding_dim=128):
        super().__init__()
        self.n_ctx = n_ctx
        self.embedding_dim = embedding_dim
        
        # Projection layer to match dimensions
        self.projection = nn.Linear(3 * 128 * 128, n_ctx * embedding_dim)
        
    # Needs backward function!
    def forward(self, image_outputs, images):
        # Flatten the images
        batch_size = images.size(0)
        flattened_images = images.view(batch_size, -1)
        
        # Project flattened images to match image_outputs dimensions
        projected_images = self.projection(flattened_images)
        projected_images = projected_images.view(batch_size, self.n_ctx, self.embedding_dim)

        # Compute mean squared error between embeddings
        loss = nn.MSELoss()(image_outputs, projected_images)
        return loss
    
class AudioEmbeddingLoss(nn.MSELoss): # Module
    def __init__(self, context_size=1024, audio_size=540, embedding_dim=64):
        super().__init__()
        self.audio_size = audio_size
        self.embedding_dim = embedding_dim
        self.context_size=context_size
        
        # Projection layer to match dimensions - FIX
        self.projection = nn.Linear(audio_size, embedding_dim)
        
    def forward(self, audio_outputs, audios):

        #print(audios.shape)
        #print(audio_outputs.shape)
        #print(audios.dtype)
        #print(audio_outputs.dtype)

        if audios.dim() == 3 and audios.size(1) == 1:
            audios = audios.squeeze(1)
        
        # Pad audios to match the context size
        batch_size = audios.size(0)
        padded_audios = torch.zeros(batch_size, self.context_size, self.audio_size, device=audios.device)
        padded_audios[:, :self.audio_size, :] = audios.unsqueeze(1).expand(-1, self.audio_size, -1)

        #batch_size = audios.size(0)
        #flattened_audios = audios.view(batch_size, -1)
        
        # Project flattened images to match image_outputs dimensions
        projected_audios = self.projection(padded_audios)
        projected_audios = padded_audios.view(batch_size, self.context_size, self.audio_size)

        #print(projected_audios.shape)
        #print(projected_audios.dtype)
        #print(audio_outputs.shape)
        #print(audio_outputs.dtype)

        # Compute mean squared error between embeddings
        loss = nn.MSELoss()(audio_outputs, projected_audios)
        return loss

class ModalityEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)  # 3 modalities: text, image, audio

    def forward(self, x):
        return self.embedding(x)

class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(config)

        # Embeddings
        self.text_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.image_embedding = ImageEmbedding(
            input_channels=3,
            image_size=128,
            embedding_dim=config.hidden_size
        )
        self.audio_embeddings = nn.Linear(config.max_audio_size, config.hidden_size)
        self.modality_embeddings = ModalityEmbedding(config.hidden_size)

        # Transformer
        self.transformer = GPT2Model(config)

        # Output heads
        self.text_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.image_head = nn.Linear(config.hidden_size, config.image_embedding_size) #768 -> 128 - might wanna change to recreate image lol
        self.audio_head = nn.Linear(config.hidden_size, config.max_audio_size) # same here rn is 768 -> 540 - good

    def forward(self, input_ids=None, image_features=None, audio_features=None, attention_mask=None):
        batch_size = input_ids.shape[0] if input_ids is not None else image_features.shape[0]
        seq_length = self.config.max_position_embeddings

        #print("Forward pass")

        # Prepare embeddings
        embeddings = torch.zeros(batch_size, seq_length, self.config.hidden_size).to(self.text_embeddings.weight.device)
        modality_ids = torch.zeros(batch_size, seq_length, dtype=torch.long).to(self.text_embeddings.weight.device)

        position = 0
        if input_ids is not None:
            #print(input_ids.shape)
            text_length = input_ids.shape[1]
            embeddings[:, position:position+text_length] = self.text_embeddings(input_ids)
            modality_ids[:, position:position+text_length] = 0  # 0 for text
            position += text_length

        if image_features is not None:
            #print(image_features.shape)
            #image_length = image_features.shape[1] # flatten image into sequence?
            #embeddings[:, position:position+image_length] = self.image_embeddings(image_features)
            image_embeddings = self.image_embedding(image_features)
            image_length = 1  # Since we're reducing the image to a single embedding
            embeddings[:, position:position+image_length] = image_embeddings.unsqueeze(1)
            modality_ids[:, position:position+image_length] = 1  # 1 for image
            position += image_length

        if audio_features is not None:
            #print(audio_features.shape)
            audio_length = audio_features.shape[1]

            #print(audio_features.float())

            embeddings[:, position:position+audio_length] = self.audio_embeddings(audio_features.float())
            modality_ids[:, position:position+audio_length] = 2  # 2 for audio

        # Add modality embeddings
        embeddings += self.modality_embeddings(modality_ids)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length).to(self.text_embeddings.weight.device)

        # Pass through transformer
        outputs = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Generate outputs for each modality
        text_output = self.text_head(hidden_states)
        image_output = self.image_head(hidden_states)
        audio_output = self.audio_head(hidden_states)

        return text_output, image_output, audio_output

# Example usage
config = GPT2Config.from_pretrained('gpt2')
config.image_embedding_size = 128  # Example size, adjust as needed
config.audio_embedding_size = 64
config.hidden_size = 192

config.max_audio_size = 540

model = MultimodalTransformer(config)

# Example forward pass
batch_size = 4
seq_length = 512

# Usage
dataset = ImageAudioTextDataset('/Users/samherring/Desktop/Projects/kandinsky/audio')
image, audio, text = dataset[0]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

#input_ids = torch.randint(0, config.vocab_size, (batch_size, 50))  # 50 text tokens
#image_features = torch.randn(batch_size, 10, config.image_embedding_size)  # 10 image tokens
#audio_features = torch.randn(batch_size, 20, config.audio_embedding_size)  # 20 audio tokens

#text_output, image_output, audio_output = model(input_ids, image_features, audio_features)

#print(f"Text output shape: {text_output.shape}")
#print(f"Image output shape: {image_output.shape}")
#print(f"Audio output shape: {audio_output.shape}")

## Training Loop

text_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # -100 is typically used for padding
#image_loss_fn = nn.MSELoss()
audio_loss_fn = AudioEmbeddingLoss()

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.1)
model.train()
total_loss = 0

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

image_loss_fn = ImageEmbeddingLoss()

for epoch in range(100):  # Changed from 'i' to 'epoch' for clarity
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/100"):
        images, audios, texts = batch
        
        # Tokenize the entire batch of texts
        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
        
        optimizer.zero_grad()

        # Forward pass
        text_outputs, image_outputs, audio_outputs = model(input_ids, images, audios)

        #print(input_ids.view(-1))
        #print(text_outputs.view(-1))
        #print(input_ids.shape)
        #print(text_outputs.shape)
        seq_length = input_ids.size(1)
        text_outputs = text_outputs[:, :seq_length, :]

        # Reshape for loss calculation
        text_outputs_flat = text_outputs.reshape(-1, text_outputs.size(-1))
        input_ids_flat = input_ids.reshape(-1)

        # Print flattened shapes for debugging
        #print(f"Flattened text_outputs shape: {text_outputs_flat.shape}")
        #print(f"Flattened input_ids shape: {input_ids_flat.shape}")

        # Calculate loss
        #print("Computing text loss...")
        text_loss = text_loss_fn(text_outputs_flat, input_ids_flat)

        images = images.float()
        #print("Computing image loss...")
        image_loss = image_loss_fn(image_outputs, images)

        audios = audios.float()
        #print("Computing audio loss...")
        audio_loss = audio_loss_fn(audio_outputs, audios)

        # Combine losses
        loss = text_loss + image_loss + audio_loss

        # Backward pass and optimize
        #print("Backward pass")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.4f}")

        #print(f"Text output shape: {text_outputs.shape}")
        #print(f"Image output shape: {image_outputs.shape}")
        #print(f"Audio output shape: {audio_outputs.shape}")
    
print("Done")