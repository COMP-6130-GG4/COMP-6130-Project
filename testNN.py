import csv
import pandas as pd
#from keras import Input, Model
#from keras.activations import softmax
#from keras.layers import Embedding, LSTM, Dense
#from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer

class ChatNN():

	def __init__(self):
		self.dataset = None
		self.tokenizer = None
		self.VOCAB_SIZE = None
		self.tokenized_inputs = None
		self.tokenized_targets = None

	def load_data(self):
		self.dataset = pd.read_csv('./dataset.csv')

		self.inputList = self.dataset['inputs'].to_list()
		self.targetList = self.dataset['targets'].to_list()

	def build_vocabulary(self):
		target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'

		self.tokenizer = Tokenizer(filters=target_regex)
		self.tokenizer.fit_on_texts(self.inputList + self.targetList)
		self.VOCAB_SIZE = len(self.tokenizer.word_index) + 1
		print(f'Vocabulary Size: {self.VOCAB_SIZE}')

		self.tokenized_inputs = self.tokenizer.texts_to_sequences(self.inputList)
		max_length_input = max(len(x) for x in self.tokenized_inputs)
		print(f'Max Length Input: {max_length_input}')
		# pad each input with zeros to the end so that each token is as long as the max
		encoder_input_data = pad_sequences(self.tokenized_inputs, maxlen=max_length_input, padding='post')

		print(encoder_input_data.shape)

		self.tokenized_targets = self.tokenizer.texts_to_sequences(self.targetList)
		max_length_target = max(len(x) for x in self.tokenized_targets)

		decoder_input_data = pad_sequences(self.tokenized_targets, maxlen=max_length_target, padding='post')

		print(f'Max Length Target: {max_length_target}')

		print(decoder_input_data.shape)

		# perform one hot encoding of targets

		for i in range(len(self.tokenized_targets)):
			self.tokenized_targets[i] = self.tokenized_targets[i][1:]

		# pad with zeros
		padded_targets = pad_sequences(self.tokenized_targets, maxlen=max_length_target, padding='post')
		
		self.decoder_output_data = to_categorical(padded_targets, self.VOCAB_SIZE)

		print(self.decoder_output_data.shape)

	# for testing tensorflow install. Remove later
	def test_tensorflow(self):
		cifar = tf.keras.datasets.cifar100
		(x_train, y_train), (x_test, y_test) = cifar.load_data()
		model = tf.keras.applications.ResNet50(
    			include_top=True,
    			weights=None,
    			input_shape=(32, 32, 3),
    			classes=100,)

		loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
		model.fit(x_train, y_train, epochs=5, batch_size=64)






if __name__ == '__main__':
	chatNN = ChatNN()

	chatNN.load_data()
	chatNN.build_vocabulary()
