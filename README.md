# Kandinsky Multimodal Model

This project is meant to be a demonstration of all the topics covered in the [NightwingCurriculum](https://github.com/samherring99/NightwingCurriculum)

Essentially I want to make an anything to anything transformer. I found audio tokenization code shared by @Maharshi-Pandya ([github](https://github.com/Maharshi-Pandya)) for nanoGPT and adapted it to use with text and image data.

The text-image dataset is `pokemon-blip-captions`, a toy dataset with 833 Pokemon image with text captions describing their shape and color - [link](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions)

The audio files for the first 151 (Gen 1) Pokemon - which is the training set as of now - are found on [sounds-resource.com](https://www.sounds-resource.com/game_boy_gbc/pokemonredblueyellow/sound/20294/)

Combining these into `text-image-audio` triples, we can now train our model.

## Usage

You need an `audio` directory with the contents of the sound resources zip file above, I won't provide it here.

```
python3 model.py
```

More work needed here, very much a POC for now, will add details ASAP.