# Covid-QA
Fine tuned models for question answering on Covid-19 data.   

# Hosted Inference

This model has been contributed to huggingface.[Click here](https://huggingface.co/abhijithneilabraham/longformer_covid_qa) to see the model in action!   

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

 ```
 from longformer_inference import qa
 print(qa(question,context))
 ```
### Example

```
 print(qa("What is the main cause of HIV-1 infection in children?","Functional Genetic Variants in DC-SIGNR Are Associated with Mother-to-Child Transmission of HIV-1\n\nhttps:\/\/www.ncbi.nlm.nih.gov\/pmc\/articles\/PMC2752805\/\n\nBoily-Larouche, Genevi\u00e8ve; Iscache, Anne-Laure; Zijenah, Lynn S.; Humphrey, Jean H.; Mouland, Andrew J.; Ward, Brian J.; Roger, Michel\n2009-10-07\nDOI:10.1371\/journal.pone.0007211\nLicense:cc-by\n\nAbstract: BACKGROUND: Mother-to-child transmission (MTCT) is the main cause of HIV-1 infection in children worldwide. Given that the C-type lectin receptor, dendritic cell-specific ICAM-grabbing non-integrin-related (DC-SIGNR, also known as CD209L or liver\/lymph node\u2013specific ICAM-grabbing non-integrin (L-SIGN)), can interact with pathogens including HIV-1 and is expressed at the maternal-fetal interface, we hypothesized that it could influence MTCT of HIV-1. METHODS AND FINDINGS: To investigate the potential role of DC-SIGNR in MTCT of HIV-1, we carried out a genetic association study of DC-SIGNR in a well-characterized cohort of 197 HIV-infected mothers and their infants recruited in Harare, Zimbabwe. Infants harbouring two copies of DC-SIGNR H1 and\/or H3 haplotypes (H1-H1, H1-H3, H3-H3) had a 3.6-fold increased risk of in utero (IU) (P = 0.013) HIV-1 infection and a 5.7-fold increased risk of intrapartum (IP) (P = 0.025) HIV-1"))
 
 >>>Mother-to-child transmission (MTCT)
```

## Fine Tuning

Fine tuning script on `longformer_finetune.py`

## Evaluation

Run `evaluator.py` to evaluate F1/EM scores. Current Scores : F1:62.3 | EM:37.83




