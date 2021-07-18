# disaster_response

Introduction

When there is a big disaster, a lot of messages in the web starts to surface. It's a very hard job to filter the messages of people asking for help, offering help, and providing useful information, from the messages that are just reverberating the news. For the autorithies and companies that tries to help the victims somehow, it's really important to understand quickly the nature of the message, so that more people can be helped as fast as possible.

In this project, we have a lot of disaster-related data from Figure Eight, and the goal is to build a Machine Learning Model that processes the messages from the web, and classifies it between some categories (e.g. people asking for medical aid, asking for food, reporting road blocks, etc).

To do so, we are going to use some NLP techniques (e.g. TF-IDF, tokenize, normalize and lemmatize) with the nltk library, and deploy the project in a web app (flask).

This project is part of the Udacity Data Science Nanodegree Program.


Requirements:

Python 3, Pandas, Numpy, Plotly, NLTK, SKLEARN, Sqlalchemy, Flask

Files Description:

DISASTER_RESPONSE
  |-- app
        |-- templates
                |-- go.html
                |-- master.html
        |-- run.py
  |-- data
        |-- disaster_message.csv
        |-- disaster_categories.csv
        |-- DisasterResponse.db
        |-- process_data.py
  |-- models
        |-- classifier.pkl
        |-- train_classifier.py
  |-- README

A) The app folder is reponsible for deploying the model to the web, with flask.
B) The data folder contains the original csv files, the database and the script to clean and process the data.  
C) The models folder contains the script of the ML model, and the model saved as a pkl file.

Running the project:

1) Clone this repository from my github to your machine
2) Make sure you have python 3 and all the libraries installed
3) From your terminal, write "python run.py"
4) Copy the link that appeared, and paste it to your internet browser.

There you go! You can see some charts about the Figure Eight data, or you can insert a text, and the model will try to make the classification. 
