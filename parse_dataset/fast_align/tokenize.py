from allennlp.data.tokenizers import SpacyTokenizer

tokenizer = SpacyTokenizer()

x = "How many percent were not female householders?"

print(tokenizer.tokenize(x))