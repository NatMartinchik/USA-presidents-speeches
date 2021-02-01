import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])

# read each speech file
speeches = [read_file(file) for file in files]

# preprocess each speech
processed_speeches = process_speeches(speeches)

# merge speeches
all_sentences = merge_speeches(processed_speeches)

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom  = all_prez_embeddings.most_similar("freedom", topn=20)

# get President Trump sentences
trump_sentences = get_president_sentences("trump")

# view most frequently used words of Trump
trump_most_freq_words = most_frequent_words(trump_sentences)
#print(trump_most_freq_words)

# create gensim model for Trump
trump_embeddings = gensim.models.Word2Vec(trump_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to americans for Trump
trump_similar_to_freedom = trump_embeddings.most_similar("americans", topn=20)
print(trump_similar_to_freedom)

# get sentences of multiple XXI centuru presidents

xxi_prez_sentences = get_presidents_sentences(["trump", "obama", "george-w-bush"])


# get most frequently used words of presidents
xxi_prez_freq_words = most_frequent_words(xxi_prez_sentences)


# create gensim model for the presidents
xxi_prez_embeddings = gensim.models.Word2Vec(xxi_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# get words similar to power for presidents
xxi_similar_to_word = xxi_prez_embeddings.most_similar("power", topn=20)

