import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 파일 읽어 오기
with open('train.txt', mode='r', encoding='utf-8') as file:
    sentence = list()
    raw_sentence = list()
    category = list()
    
    while True:
        data = file.readline()
        
        if data:
            category.append(data.split('\t')[0])
            sentence.append(data.split('\t')[1])
            raw_sentence.append(data.split('\t')[1])
        else:
            break

tokenized_input_text = list()
indexed_input_text = list()

tokenized_output_text = list()
indexed_output_text = list()

for (sentence1, sentence2) in zip(sentence, category):
    sentence1 = "[CLS] " + sentence1 + " [SEP]"
    sentence2 = "[CLS] " + sentence2 + " [SEP]"
    
    tokenized_input_text.append(tokenizer.tokenize(sentence1))
    tokenized_output_text.append(tokenizer.tokenize(sentence2))
    
    indexed_input_text.append(tokenizer.convert_tokens_to_ids(tokenized_input_text))
    indexed_output_text.append(tokenizer.convert_tokens_to_ids(tokenized_output_text))
    
for idx in range(100):
    print('{} {}'.format(tokenized_input_text[idx], indexed_input_text[idx]))
