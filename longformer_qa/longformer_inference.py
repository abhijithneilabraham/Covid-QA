from transformers import LongformerTokenizerFast
import torch
from transformers import LongformerForQuestionAnswering
tokenizer = LongformerTokenizerFast.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')
model = LongformerForQuestionAnswering.from_pretrained("covid_qa_longformer")

def qa(question,text):
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask,return_dict=False)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    return answer

