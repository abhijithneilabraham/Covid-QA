# Covid-QA
Fine tuned models for question answering on Covid-19 data.   

## Dependencies
`pip install -r requirements.txt`

## Getting Started
```
cd longformer_qa
```

## Dataset

Covid 19 QA data obtained from transformers: [covid QA deepset](https://huggingface.co/datasets/covid_qa_deepset).  
Data Generation script: `generate_data.py`

## Inference

Longformer model fine tuned on the data: [download](https://drive.google.com/file/d/11jO8zSvJFeINRvIIJPL34D0nsWGXdNkD/view?usp=sharing).  
download and unzip inside `longformer_qa/`.   
Scripts on `longformer_inference.py`
Example:
 ```
 from longformer_inference import qa
 print(qa(question,context))
 ```

## Fine Tuning

Fine tuning script on `longformer_finetune.py`

## Evaluation

Run `evaluator.py` to evaluate F1/EM scores. Current Scores : F1:62.3 | EM:37.83




