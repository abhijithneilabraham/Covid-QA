# Covid-QA
Fine tuned models for question answering on Covid-19 data.

## Dataset

Covid 19 QA data obtained from transformers: [covid QA deepset](https://huggingface.co/datasets/covid_qa_deepset).  
Data Generation script: `generate_data.py`

## Inference

Longformer model fine tuned on the data: [download](https://drive.google.com/drive/folders/1g-bZ2eiLZv2vI1g-oQ9eTm2E7uhatYff?usp=sharing).  
Make sure the model checkpoints are in `covid_qa_longformer/`.  
Scripts on `longformer_inference.py`
Example:
 ```
 from longformer_inference import qa
 print(qa(question,context))
 ```

## Fine Tuning

Fine tuning script on `longformer_finetune.py`

## Evaluation

Run `evaluator.py` to evaluate F1/EM scores. Current Scores : F1:49.86 | EM:23.43



