from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def create_embeddings(chunks):
    texts = [chunk[1] for chunk in chunks]
    embeddings = sentence_model.encode(texts)
    return embeddings

def encode_bert(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
