from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils as utils
from keras.models import Sequential
import keras.layers
import numpy as np
import tensorflow as tf

NUM_SENTENCES = 1000

REMOVABLE_TOKENS = ['the', '\'ve', '\'m', '\'s', '\'d', '\'ll']

# REMOVABLE_TOKENS = ['the', '\'s']

def main():
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # rawDataFile = '/home/pcori/GrammarChecker/rawtext.txt'
    rawDataFile = 'C:/Users/phili/GrammarChecker/rawtext.txt'
    text = open(rawDataFile, 'r').read()
    sentences = text.split('\n')
    trainSentences = sentences[:NUM_SENTENCES]
    testSentences = sentences[NUM_SENTENCES:NUM_SENTENCES + int(NUM_SENTENCES * 0.2)]

    train_incorrectSentences, train_trueSentences = getIncorrectSentences(trainSentences, 300)
    test_incorrectSentences, test_trueSentences = getIncorrectSentences(testSentences, 300)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.concatenate((trainSentences, testSentences)))
    trainEncoded = tokenizer.texts_to_sequences(trainSentences)
    training_trueSentencesEncoded = tokenizer.texts_to_sequences(train_trueSentences)
    test_trueSentencesEncoded = tokenizer.texts_to_sequences(test_trueSentences)
    vocab_size = len(tokenizer.word_index) + 1
    print('vocab size: ' + str(vocab_size))

    maxLength = max([len(x) for x in np.concatenate((trainEncoded, training_trueSentencesEncoded))])
    x_train, y_train = prepareSequences(trainEncoded, maxLength)
    x_test, y_test = prepareSequences(training_trueSentencesEncoded, maxLength)
    y_train = utils.to_categorical(y_train, num_classes=vocab_size)
    y_test = utils.to_categorical(y_test, num_classes=vocab_size)

    model = Sequential()
    model.add(keras.layers.Embedding(vocab_size, 10, input_length=maxLength - 1))
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=300, verbose=2)
    loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
    print('training loss: ' + str(loss) + ', accuracy: ' + str(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('test loss: ' + str(loss) + ', accuracy: ' + str(accuracy))

    # evaluate
    vocab = tokenizer.word_index
    vocab_inv = {v: k for k, v in vocab.items()}

    training_correctedSentences = processSentences(train_incorrectSentences, model, tokenizer, maxLength)

    print('train data results:')
    for i in range(len(training_correctedSentences)):
        print('orig: ' + train_incorrectSentences[i])
        print('pred: ' + decodeSentence(training_correctedSentences[i], vocab_inv))
        print('true: ' + decodeSentence(training_trueSentencesEncoded[i], vocab_inv))


    test_correctedSentences = processSentences(test_incorrectSentences, model, tokenizer, maxLength)

    print('test data results:')
    for i in range(len(test_correctedSentences)):
        print('orig: ' + test_incorrectSentences[i])
        print('pred: ' + decodeSentence(test_correctedSentences[i], vocab_inv))
        print('true: ' + decodeSentence(test_trueSentencesEncoded[i], vocab_inv))

    print('train accuracy: ' + str(getAccuracy(training_correctedSentences, training_trueSentencesEncoded)))
    print('test accuracy: ' + str(getAccuracy(test_correctedSentences, test_trueSentencesEncoded)))


    # print(getAccuracy(correctedSentences, tokenizer.texts_to_sequences(['he \'ll never get it right . try the log ride !'])))

    # testText = 'do n\'t eat at the console .'
    # testEncoded = tokenizer.texts_to_sequences(incorrectSentences)
    # x_test, y_test, _ = prepareXY(testEncoded, maxLength)
    # probabilities = model.predict(x_test, verbose=0)
    # print(len(probabilities))
    #
    # vocab = tokenizer.word_index
    # vocab_inv = {v: k for k, v in vocab.items()}
    # log_p_sentence = 0
    # for i, prob in enumerate(probabilities):
    #     word = vocab_inv[y_test[i]]
    #     history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
    #     prob_word = prob[y_test[i]]
    #     log_p_sentence += np.log(prob_word)
    #     print('P(w={} | h={})={}'.format(word, history, prob_word))
    # print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))

def prepareSequences(encodedLines, maxLength=None):
    sequences = list()
    for encodedLine in encodedLines:
        for i in range(1, len(encodedLine)):
            seq = encodedLine[:i + 1]
            sequences.append(seq)
    maxlen = maxLength if (maxLength) else max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=maxlen, padding='pre')

    # split into input and output elements
    sequences = np.array(sequences)
    x = sequences[:, :-1]
    y = sequences[:, -1]
    return x, y

def getAccuracy(predSentences, trueSentences):
    accuracy = 0
    for i in range(len(predSentences)):
        if (predSentences[i] == trueSentences[i]):
            accuracy += 1
    return accuracy / len(predSentences)

def processSentences(sentences, model, tokenizer, maxLength):
    correctedSentences = []
    encodedToks = [x[0] for x in tokenizer.texts_to_sequences(REMOVABLE_TOKENS)]
    print(encodedToks)
    numSentences = len(sentences)
    for i, sentence in enumerate(sentences):
        print('correcting sentence ' + str(i) + ' of ' + str(numSentences))
        encoded = tokenizer.texts_to_sequences([sentence])[0]
        x, y = prepareSequences([encoded], maxLength)
        probs = model.predict(x, verbose=0)
        sentence_prob = get_sentence_prob(probs, y)
        bestEncoding = encoded
        for i, token in enumerate(encoded):
            for tok in encodedToks:
                tmpEncoded = list(encoded)
                encoded.insert(i, tok)
                tmp_x, tmp_y = prepareSequences([encoded], maxLength)
                new_probs = model.predict(tmp_x, verbose=0)
                new_sentence_prob = get_sentence_prob(new_probs, tmp_y)
                if (new_sentence_prob > sentence_prob):
                    sentence_prob = new_sentence_prob
                    bestEncoding = encoded
                encoded = tmpEncoded
        correctedSentences.append(bestEncoding)
    return correctedSentences

def decodeSentence(tokens, vocab):
    return ' '.join([vocab[token] for token in tokens])

def get_sentence_prob(probs, encodedSentence):
    log_p_sentence = 0
    for i, prob in enumerate(probs):
        prob_word = prob[encodedSentence[i]]
        log_p_sentence += np.log(prob_word)
    return np.exp(log_p_sentence)

def prepareIncorrectSentences(sentences):
    return [removeToken(x) for x in sentences]

def removeToken(sentence):
    words = sentence.split(' ')
    delete_i = -1
    for i, word in enumerate(words):
        if (word in REMOVABLE_TOKENS):
            print('removing ' + word)
            delete_i = i
            break
    if (delete_i != -1):
        words.pop(delete_i)
    return ' '.join([x for x in words])

def getIncorrectSentences(sentences, num_sentences=None):
    x = []
    y = []
    for sent in sentences:
        new_sent = removeToken(sent)
        if (sent != new_sent):
            x.append(new_sent)
            y.append(sent)
        if (num_sentences and len(x) == num_sentences):
            break
    return x, y


if __name__ == '__main__':
    main()
