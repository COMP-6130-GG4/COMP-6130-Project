# COMP-6130-Project
Final Project for the course

- Topic: NLP Sequence to Sequence Learning with Neural Networks 
- Target: Build a chatbot using a Sequence to Sequence NLP Neural Network
- Dataset: https://www.kaggle.com/datasets/hassanamin/chatbot-nlp


## How to run

1. Clone Repo
2. Install requirements

3. If you want to train the model, then have the following lines between lines 216 and 222 uncommented in the __main__ portion of chatNN.py

  ``` Python
  
  chatNN.load_data()
  chatNN.build_vocabulary()
  chatNN.build_model()
  chatNN.train_encoder_decoder()
  chatNN.make_inference_model()
  
  ```
  
  Otherwise if you want to load a pretrained model then have the following lines between lines 216 and 222 uncommented in the __main__ portion of chatNN.py
  
  
  ``` Python
  
  chatNN.load_data()
  chatNN.build_vocabulary()
  chatNN.make_inference_model()
  chatNN.load_pretrained_model()
  
  ```
  
4. run chatNN.py
