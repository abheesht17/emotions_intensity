import transformers
from transformers import AutoModel, AutoTokenizer

def compute_embeddings(text,model_name="bert-base-uncased"):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name)
	inputs = tokenizer(text,return_tensors="pt")
	outputs = model(**inputs)
	embeddings = outputs.last_hidden_state
	return embeddings





