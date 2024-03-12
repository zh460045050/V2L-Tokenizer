import numpy as np
import torch
import clip

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-L/14", device=DEVICE)
llama_texts = np.load("Subword_Bigram_Trigram_Vocabulary.npy", allow_pickle=True)
local_codebook = []
global_codebook = []

####Generate Subword Embeddings
print("Generate Subword Embeddings....")
for token_cell in llama_texts:
    
    cur_token = token_cell["1"]

    texts = [template.format(cur_token) for template in imagenet_templates] 
    
    text_features = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = torch.mean(text_features, dim=0)
    text_features = text_features.unsqueeze(0)
    global_codebook.append(text_features)
    local_codebook.append(text_features)

####Generate Bigrams Embeddings
print("Generate Bigrams Embeddings....")
for token_cell in llama_texts:
    
    cur_token = token_cell["1"] + token_cell["2"]
    texts = [template.format(cur_token) for template in imagenet_templates] 
    
    text_features = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = torch.mean(text_features, dim=0)
    text_features = text_features.unsqueeze(0)
    global_codebook.append(text_features)

####Generate Trigrams Embeddings
print("Generate Trigrams Embeddings....")
for token_cell in llama_texts:
    
    cur_token = token_cell["1"] + token_cell["2"] + token_cell["3"]
    texts = [template.format(cur_token) for template in imagenet_templates] 
    text_features = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = torch.mean(text_features, dim=0)
    text_features = text_features.unsqueeze(0)
    global_codebook.append(text_features)


##Global Codebook: Both Subword Bigram Trigram
global_codebook = torch.concat(global_codebook, dim=0)
torch.save(global_codebook, "Subword_Bigram_Trigram_Embedding.pth")

##Local Codebook: Only for Subwords
local_codebook = torch.concat(local_codebook, dim=0)
torch.save(local_codebook, "local_codebook_embedding.pth")