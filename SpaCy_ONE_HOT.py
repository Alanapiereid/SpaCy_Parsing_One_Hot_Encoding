import spacy
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax

#using spaCy, as always

nlp = spacy.load('en_core_web_lg')

#get a text

my_string = '.'

#create a tag lexicon using SpaCy's terms - this list should be exhaustive

tag_lexicon = ['ROOT', 'punct', 'advmod', 'nsubj', 'npadvmod', 'intj', 'poss', 'conj', 'amod', 'cc', 'predet', 'dobj', 'None', 'advcl', 'aux', 'nummod', 'pobj', 'compound', 'relcl', 'neg', 'acomp', 'xcomp', 'prep', 'det']

#define a function to parse # this one is for dependency


def spacy_dep(text):
    text = nlp(text)
    tokens = [token.dep_ for token in text]
    return tokens

print(spacy_dep(my_string))

# integer encoding then one hot

def dep_to_onehot(text):
    text_tags = spacy_dep(text)
    char_to_int = dict((c, i) for i, c in enumerate(tag_lexicon))
    integer_encoded = [char_to_int[char] for char in text_tags]

    onehot_encoded = list()
    for value in integer_encoded:
    	word = [0 for _ in range(len(tag_lexicon))]
    	word[value] = 1
    	onehot_encoded.append(word)
    
    return onehot_encoded



print(dep_to_onehot(my_string))