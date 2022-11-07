import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json

class ChatNN():

	def __init__(self):
		self.dataset = None
		self.tokenizer = None
		self.VOCAB_SIZE = None
		self.tokenized_inputs = None
		self.tokenized_targets = None

	def load_data(self):
		self.dataset = pd.read_csv('./dataset.csv')

		# only use the first 1000 since the dataset is too big
		self.inputList = self.dataset['inputs'].to_list()
		self.targetList = self.dataset['targets'].to_list()

	def build_vocabulary(self):
		target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'

		self.tokenizer = Tokenizer(filters=target_regex)
		self.tokenizer.fit_on_texts(self.inputList + self.targetList)
		self.VOCAB_SIZE = len(self.tokenizer.word_index) + 1
		print(f'Vocabulary Size: {self.VOCAB_SIZE}')

		self.tokenized_inputs = self.tokenizer.texts_to_sequences(self.inputList)
		self.max_length_input = max(len(x) for x in self.tokenized_inputs)
		print(f'Max Length Input: {self.max_length_input}')
		# pad each input with zeros to the end so that each token is as long as the max
		self.encoder_input_data = pad_sequences(self.tokenized_inputs, maxlen=self.max_length_input, padding='post')

		print(self.encoder_input_data.shape)

		self.tokenized_targets = self.tokenizer.texts_to_sequences(self.targetList)
		self.max_length_target = max(len(x) for x in self.tokenized_targets)

		self.decoder_input_data = pad_sequences(self.tokenized_targets, maxlen=self.max_length_target, padding='post')

		print(f'Max Length Target: {self.max_length_target}')

		print(self.decoder_input_data.shape)

		# perform one hot encoding of targets

		for i in range(len(self.tokenized_targets)):
			self.tokenized_targets[i] = self.tokenized_targets[i][1:]

		# pad with zeros
		padded_targets = pad_sequences(self.tokenized_targets, maxlen=self.max_length_target, padding='post')

		self.decoder_output_data = to_categorical(padded_targets, self.VOCAB_SIZE)

		print(self.decoder_output_data.shape)

	def build_model(self):
		# encoder will be used to capture space-dependent 
		# relations between words from the questions
		self.encoder_inputs = Input(shape=(None,), name='encoder_inputs')

		encoder_embedding = Embedding(self.VOCAB_SIZE, 800, mask_zero=True, name="encoder_embedding")(self.encoder_inputs)

		encoder_outputs, state_h, state_c = LSTM(800, return_state=True, name="encoder_lstm")(encoder_embedding)

		self.encoder_states = [state_h, state_c]

		# decoder will be used to capture space-dependent relations 
		# between words from the answers using encoder's 
		# internal state as a context

		self.decoder_inputs = Input(shape=(None,), name="decoder_inputs")

		self.decoder_embedding = Embedding(self.VOCAB_SIZE, 800, mask_zero=True, name="decoder_embedding")(self.decoder_inputs)

		self.decoder_lstm = LSTM(800, return_state=True, return_sequences=True, name="decoder_lstm")

		decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding, initial_state=self.encoder_states)

		# the decoder is connected to the output Dense layer

		self.decoder_dense = Dense(self.VOCAB_SIZE, activation=softmax, name="decoder_dense")
		output = self.decoder_dense(decoder_outputs)

		self.model = Model([self.encoder_inputs, self.decoder_inputs], output)

		self.model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

		self.model.summary()

	def train_encoder_decoder(self):
		history = self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_output_data,
							batch_size=10, epochs=150)

		# summarize history for accuracy and loss
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['loss'])
		plt.title('Accuracy and Loss (Full Dataset)')
		plt.ylabel('Value')
		plt.xlabel('Epoch')
		plt.legend(['Accuracy', 'Loss'], loc='upper left')
		plt.show()


	def make_inference_model(self):
		# two inputs for the state vectors returned by the encoder
		decoder_state_input_h = Input(shape=(800,))
		decoder_state_input_c = Input(shape=(800,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		# these state vectors are used as an initial state
		# for LSTM layer in the inference decoder
		# third input is the Embedding layer
		decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_embedding, initial_state=decoder_states_inputs)

		decoder_states = [state_h, state_c]
		# Dense layer is used to return ONE predicted word
		decoder_outputs = self.decoder_dense(decoder_outputs)

		self.decoder_model = Model(inputs=[self.decoder_inputs] + decoder_states_inputs,
							  outputs=[decoder_outputs] + decoder_states)

		with open('decoder_model.json', 'w', encoding='utf8') as f:
			f.write(self.decoder_model.to_json())
		self.decoder_model.save_weights('decoder_model_weights.h5')


		# single encoder input is an utterance, represented as a sequence 
		# of integers padded with zeros
		self.encoder_model = Model(inputs=self.encoder_inputs, outputs=self.encoder_states)

		with open('encoder_model.json', 'w', encoding='utf8') as f:
			f.write(self.encoder_model.to_json())
		self.encoder_model.save_weights('encoder_model_weights.h5')


	def load_model(self, model_filename, model_weights_filename):
		with open(model_filename, 'r', encoding='utf8') as f:
			model = model_from_json(f.read())
		model.load_weights(model_weights_filename)
		return model

	def load_pretrained_model(self):
		self.encoder_model = self.load_model('encoder_model.json', 'encoder_model_weights.h5')
		self.decoder_model = self.load_model('decoder_model.json', 'decoder_model_weights.h5')

	def tokenizeUtterance(self, utt):
		# convert input string to lowercase
		# then split it by whitespace
		words = utt.lower().split()
		# and then convert to a sequence
		# of integers padded by zeros
		token_list = list()
		for current_word in words:
			result = self.tokenizer.word_index.get(current_word, '')
			if result != '':
				token_list.append(result)

		return pad_sequences([token_list], maxlen=self.max_length_input, padding='post')



	def getResponse(self, userInput):
		#states_values = self.encoder_model.predict(self.tokenizeUtterance(input('Say something: ')))
		states_values = self.encoder_model.predict(self.tokenizeUtterance(userInput))

		empty_target_seq = np.zeros((1,1))
		empty_target_seq[0, 0] = self.tokenizer.word_index['start']
		stop_condition = False
		decoded_response = ''
		while not stop_condition:
			# feed the state vectors and 1-word target sequence
			# to the decoder to produce predictions for the next word
			decoder_outputs, h, c = self.decoder_model.predict([empty_target_seq] + states_values)

			# sample the next word using these predictions
			sampled_word_index = np.argmax(decoder_outputs[0, -1, :])
			sampled_word = None

			# append the sampled word to the target sequence 
			for word, index in self.tokenizer.word_index.items():
				if sampled_word_index == index:
					if word != 'end':
						decoded_response += ' {}'.format(word)
					sampled_word = word

			# repeat until we generate the end-of-sequence word 'end'
			# or we hit the length of answer limit

			if sampled_word == 'end' or len(decoded_response.split()) > self.max_length_target:
				stop_condition = True

			# prepare next iteration 
			empty_target_seq = np.zeros((1,1))
			empty_target_seq[0, 0] = sampled_word_index
			states_values = [h, c]

		return decoded_response




if __name__ == '__main__':
	chatNN = ChatNN()

	chatNN.load_data()
	chatNN.build_vocabulary()
	chatNN.build_model()
	chatNN.train_encoder_decoder()
	chatNN.make_inference_model()

	#chatNN.load_pretrained_model()


	userInput = None
	while not userInput == 'QUIT':
		userInput = input('Say Something: ')
		if not(userInput == 'QUIT'):
			print(chatNN.getResponse(userInput))
