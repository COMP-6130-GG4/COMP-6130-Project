import re
import pandas as pd
import yaml

def processChatBot(fileName=None):

	yamlList = [
		'ai.yml',
		'botprofile.yml',
		'computers.yml',
		'emotion.yml',
		'food.yml',
		'gossip.yml',
		'greetings.yml',
		'health.yml',
		'history.yml',
		'humor.yml',
		'literature.yml',
		'money.yml',
		'movies.yml',
		'politics.yml',
		'psychology.yml',
		'science.yml',
		'sports.yml',
		'trivia.yml'
	]

	data = None

	if fileName:

		with open(f'chatbot-corpus/{fileName}', 'r') as stream:
			try:
				data = yaml.safe_load(stream)
			except yaml.YAMLError as exc:
				print(exc)

	else:
		with open('dataset.csv', 'w', newline='') as csv_file:
			csv_file.write('inputs,targets\n')

			for fileName in yamlList:
				with open("chatbot-corpus/" + fileName, "r") as stream:
					try:
						data = yaml.safe_load(stream)
					except yaml.YAMLError as exc:
						print(exc)

					if(data):

						for convo in data['conversations']:
							try:
								i = cleanUtterance(convo[0])
								r = cleanUtterance(convo[1])
								csv_file.write(f'{i},<START>{r}<END>\n')
							except:
								pass


# Util Function
def cleanUtterance(utt):
	utt = utt.lower()

	# getting rid of common contractions
	#https://www.sjsu.edu/writingcenter/docs/handouts/Contractions.pdf
	utt = re.sub(r"aren't", "are not", utt)
	utt = re.sub(r"hadn't", "had not", utt)
	utt = re.sub(r"hasn't", "has not", utt)
	utt = re.sub(r"haven't", "have not", utt)
	utt = re.sub(r"he'd", "he would", utt)
	utt = re.sub(r"he'll", "he will", utt)
	utt = re.sub(r"he's", "he is", utt)
	utt = re.sub(r"it's", "it is", utt)
	utt = re.sub(r"i'd", "i would", utt)
	utt = re.sub(r"isn't", "is not", utt)
	utt = re.sub(r"doesn't", "does not", utt)
	utt = re.sub(r"didn't", "did not", utt)
	utt = re.sub(r"couldn't", "could not", utt)
	utt = re.sub(r"that's", "that is", utt)
	utt = re.sub(r"there's", "there is", utt)
	utt = re.sub(r"they'd", "they had", utt)
	utt = re.sub(r"they're", "they are", utt)
	utt = re.sub(r"they've", "they have", utt)
	utt = re.sub(r"they'll", "they will", utt)
	utt = re.sub(r"shouldn't", "should not", utt)
	utt = re.sub(r"i'm", "i am", utt)
	utt = re.sub(r"'bout", "about", utt)
	utt = re.sub(r"'til", "until", utt)
	utt = re.sub(r"let's", "let us", utt)
	utt = re.sub(r"can't", "cannot", utt)
	utt = re.sub(r"you're", "you are", utt)
	utt = re.sub(r"you'd", "you had", utt)
	utt = re.sub(r"you'll", "you will", utt)
	utt = re.sub(r"you've", "you have", utt)
	utt = re.sub(r"she's", "she is", utt)
	utt = re.sub(r"she'll", "she will", utt)
	utt = re.sub(r"she'd", "she would", utt)
	utt = re.sub(r"don't", "do not", utt)
	utt = re.sub(r"i've", "i have", utt)
	utt = re.sub(r"we're", "we are", utt)
	utt = re.sub(r"we'd", "we had", utt)
	utt = re.sub(r"we've", "we have", utt)
	utt = re.sub(r"won't", "will not", utt)
	utt = re.sub(r"we'll", "we will", utt)
	utt = re.sub(r"weren't", "were not", utt)
	utt = re.sub(r"what'll", "what will", utt)
	utt = re.sub(r"what're", "what are", utt)
	utt = re.sub(r"what's", "what is", utt)
	utt = re.sub(r"what've", "what have", utt)
	utt = re.sub(r"where's", "where is", utt)
	utt = re.sub(r"who'd", "who had", utt)
	utt = re.sub(r"who'll", "who will", utt)
	utt = re.sub(r"who're", "who are", utt)
	utt = re.sub(r"i'm", "i am", utt)
	utt = re.sub(r"who've", "who have", utt)
	utt = re.sub(r"wouldn't", "would not", utt)
	utt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", utt)

	return utt


if __name__ == '__main__':
	processChatBot()

		
