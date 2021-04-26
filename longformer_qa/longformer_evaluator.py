import pandas as pd
from longformer_inference import qa
from datasets import load_dataset
val = load_dataset("covid_qa_deepset",split='train[90%:]') 
val_contexts, val_questions, val_answers =val["context"], val["question"],val["answers"]
from collections import Counter
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

f1_total=0
em_total=0
p,ans=[],[]
count=0
print(len(val_questions))
for c,q,a in zip(val_contexts, val_questions, val_answers):
    try:
        pred=qa(q,c)
        answer=a["text"][0]
        p.append(pred)
        ans.append(answer)
        f1=f1_score(pred,answer)
        em=exact_match_score(pred,answer)
        if em:
            em_score=1
        else:
            em_score=0
        f1_total+=f1
        em_total+=em_score
        print(f1)
        print(em)
        count+=1
        print(count)
    except:
        continue

df=pd.DataFrame({"pred":p,"actual":ans})
df.to_csv("predvsact.csv",sep="|")

f1_total=f1_total/count
em_total=em_total/count

file1 = open("scores.txt","w")
L = [str(f1_total*100),str(em_total*100)] 
  
# \n is placed to indicate EOL (End of Line)
file1.write("F1 and EM scores \n")
file1.write(L[0]+"\n")
file1.write(L[1])
file1.close()

