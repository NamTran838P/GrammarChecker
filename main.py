from __future__ import absolute_import, division, print_function, unicode_literals
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import keras.utils as utils
from keras.models import Sequential
import keras.layers
import numpy as np
import nltk

NUM_CHARS = 10000

def main():
	origDataSetFile = '/home/pcori/GrammarChecker/cornell_movie_quotes_corpus/moviequotes.scripts.txt'
	rawDataFile = '/home/pcori/GrammarChecker/rawtext.txt'
	write_raw_data_file(origDataSetFile)
	quit()
	trainText = open(rawDataFile, 'r').read()[:NUM_CHARS]
	testText = open(rawDataFile, 'r').read()[NUM_CHARS:NUM_CHARS + 300]
	# trainText = """ Jack and Jill went up the hill\n
	# 	To fetch a pail of water\n
	# 	Jack fell down and broke his crown\n
	# 	And Jill came tumbling after\n """
	tokenizer = Tokenizer(filters='\n.!?,.')
	tokenizer.fit_on_texts([trainText])
	encoded = tokenizer.texts_to_sequences([trainText])[0]
	vocab_size = len(tokenizer.word_index) + 1
	print(encoded)
	sequences = list()
	for i in range(1, len(encoded)):
		sequence = encoded[i - 1:i + 1]
		sequences.append(sequence)

	# split into X and y elements
	sequences = np.array(sequences)
	X, y = sequences[:, 0], sequences[:, 1]
	y = utils.to_categorical(y, num_classes=vocab_size)
	# define model
	model = Sequential()
	model.add(keras.layers.Embedding(vocab_size, 10, input_length=1))
	model.add(keras.layers.LSTM(50))
	model.add(keras.layers.Dense(vocab_size, activation='softmax'))


	# compile network
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(X, y, epochs=500, verbose=2)

	# evaluate
	in_text = 'Ladies'
	encoded = tokenizer.texts_to_sequences([in_text])[0]
	print(encoded)
	encoded = np.array(encoded)
	print(encoded.shape)
	yhat = model.predict_classes(encoded, verbose=0)
	print(yhat)
	i = 0
	for word, index in tokenizer.word_index.items():
		if index == yhat[i]:
			i += 1
			print(word)

# Use once to generate the raw text data file
def write_raw_data_file(filename):
	file = open(filename, "r")
	newFile = open('rawtext.txt', 'w')
	for line in file:
		parts = line.split(" +++$+++ ")
		dialog_line = parts[-1]
		s = dialog_line.strip().lower()
		preprocessed_line = " ".join(word_tokenize(s))
		newFile.write(preprocessed_line + "\n")

if __name__ == '__main__':
    main()
