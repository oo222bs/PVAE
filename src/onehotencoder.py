from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
import numpy as np
#import spacy
#nlp = spacy.load('en_core_web_sm')
# text = "Mary, don’t slap the green witch"
# print([str(token) for token >in nlp(text.lower())])
#words = []
# sentences = []
# for i in range(len(corpus)):
#     for token in nlp(corpus[i].lower()):
#         words.append(token)
#     sentences.append(words)
#     words = []

file = open('../descriptions.txt', 'r')
corpus = file.read().splitlines()
doc = []
for i in range(len(corpus)):
    corpus[i] = "<BOS/EOS> " + corpus[i] + " <BOS/EOS>"
    corpus[i] = corpus[i].lower().split()
    doc = doc + corpus[i]

values = array(doc)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded = np.reshape(onehot_encoded, (72,5,9))
print('Test')