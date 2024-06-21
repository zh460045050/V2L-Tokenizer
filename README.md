# V2L-Tokenizer

## Code of V2L Tokenizer

Official Code of the paper "Beyond Text: Frozen Large Language Models in Visual Signal Comprehension"

### Training
* * * 


The proposed V2L Tokenizer can be trained with the following steps (Run.sh):

1. Downloading the few-shot splits and imagenet split on [Google Drive](https://drive.google.com/drive/folders/1Z8GxE-WMEijJV-JZmqL7AGzsB0gHk4ow?usp=share_link)

2. Confirming "$imagenet_path" is set as the folder of ImageNet1K dataset that has been arranged with following layout:
     <br/>|--ImageNet1K
     <br/>    |--train
     <br/>    |   |---n01440764
     <br/>    |   |---01443537
     <br/>    |   |---...
     <br/>    |--val
     <br/>    |    |--ILSVRC2012_val_00000001.JPEG
     <br/>    |    |--ILSVRC2012_val_00000002.JPEG
     <br/>    |    |--....

3. Confirming "$llama_path" is set as the folder of LLaMA-2 model, containing its original model weight and tokenizer.

4. Run "step1_epanding_vocabulary_set.py" to expand the vocabulary set of LLaMA-2 with the proposed codebook extension strategy. 

5. Run "step2_generate_codebook_embedding.py" to generate the vision-language codebook embeddings for the vocabulary sets.

6. Run "step3_global_codebook_filtering.py" to filter the vocabulries that has less visual semantics.

7. Run "step4_training_v2l_tokenizer.py" to train the V2L Tokenizer based on the codebook produced by the above 3 steps.

We also provided our codebooks and checkpoints at: https://drive.google.com/drive/folders/1Z8GxE-WMEijJV-JZmqL7AGzsB0gHk4ow?usp=sharing


### Validation
* * * 

The proposed V2L Tokenizer can be used for visual signal reconstruction, comprehension and denoising generation with LLaMA-V2:

1. Run "eval_reconstruction.py" to evalute reconstruction performance on ImageNet1K validation set.

2. Run "eval_understanding.py" to evalute comprehension performance on nway-kshot classficiation performance on mini-ImageNet.

3. Run "eval_denoising_generation.py" to evaluate the denoising generation performance on a subset of ImageNet1K validation set. 

