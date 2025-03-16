import json
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
dataset = "cranfield/cran_docs.json"
with open(dataset, 'r') as file:
    docs_json = json.load(file)

# Extract text from documents 
docs = [item["body"] for item in docs_json]  

# Tokenization & Preprocessing
corpus = []
for doc in docs:
    tokens = word_tokenize(doc.lower())  
    tokens = [word for word in tokens if word.isalpha()] 
    corpus.append(set(tokens))  

# Compute Document Frequency (DF)
word_doc_freq = defaultdict(int)  
total_docs = len(corpus)

for doc in corpus:
    for word in doc:
        word_doc_freq[word] += 1  

# Compute IDF values
idf_threshold = 1.0 
corpus_stopwords = {word for word, freq in word_doc_freq.items() if math.log(total_docs / (freq + 1)) <= idf_threshold}

# Load NLTK's English stopwords
nltk_stopwords = set(stopwords.words('english'))

# Find common stopwords
common_stopwords = corpus_stopwords.intersection(nltk_stopwords)


print()
print(f"Cranfield Dataset specific Stopwords ({len(corpus_stopwords)}):\n", corpus_stopwords)
print(f"\nNLTK Stopwords ({len(nltk_stopwords)}):\n", nltk_stopwords)
print(f"\nCommon Stopwords ({len(common_stopwords)}):\n", common_stopwords)

