# V2L-Tokenizer

## Code of V2L Tokenizer

Official Code of the paper "Beyond Text: Frozen Large Language Models in Visual Signal Comprehension"

### Training
* * * 

The proposed V2L Tokenizer can be trained with the following steps (Run.sh):

1. Confirming "$imagenet_path" is set as the folder of ImageNet1K dataset that has been arranged with following layout:
|-ImageNet1K
|   |--train
|   |   |---n01440764
|   |   |---n01443537
|   |   |---...
|   |--val
|   |   |---n01440764
|   |   |---n01443537
|   |   |---...

2. Confirming "$llama_path" is set as the folder of LLaMA-2 model, containing its original model weight and tokenizer.

3. Run "step1_epanding_vocabulary_set.py" to expand the vocabulary set of LLaMA-2 with the proposed codebook extension strategy. 

4. Run "step2_generate_codebook_embedding.py" to generate the vision-language codebook embeddings for the vocabulary sets.

5. Run "step3_global_codebook_filtering.py" to filter the vocabulries that has less visual semantics.

6. Run "step4_training_v2l_tokenizer.py" to train the V2L Tokenizer based on the codebook produced by the above 3 steps.


### Validation
* * * 

The proposed V2L Tokenizer can be used for visual signal reconstruction, comprehension and denoising generation with LLaMA-V2:

1. Run "eval_reconstruction.py" to evalute reconstruction performance on ImageNet1K validation set.

2. Run "eval_understanding.py" to evalute comprehension performance on nway-kshot classficiation performance on mini-ImageNet.

3. Run "eval_denoising_generation.py" to evaluate the denoising generation performance on a subset of ImageNet1K validation set. 

