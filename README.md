# disaster_response

**Introduction**

When there is a big disaster, a lot of messages in the web starts to surface. It's a very hard job to filter the messages of people asking for help, offering help, and providing useful information, from the messages that are just reverberating the news. For the autorithies and companies that tries to help the victims somehow, it's really important to understand quickly the nature of the message, so that more people can be helped as fast as possible.

In this project, we have a lot of disaster-related data from Figure Eight, and the goal is to build a Machine Learning Model that processes the messages from the web, and classifies it between some categories (e.g. people asking for medical aid, asking for food, reporting road blocks, etc).

To do so, we are going to use some NLP techniques (e.g. TF-IDF, tokenize, normalize and lemmatize) with the nltk library, combined with a machine learning model and deploy the project in a web app (flask).

This project is part of the Udacity Data Science Nanodegree Program.


**Requirements:**

Python 3, Pandas, Numpy, Plotly, NLTK, SKLEARN, Sqlalchemy, Flask

**Files Description:**

![image](https://user-images.githubusercontent.com/48065052/126054259-b327dfe6-0a28-4015-b59c-9ae042e4ed99.png)

• The app folder is reponsible for deploying the model to the web, with flask.
• The data folder contains the original csv files, the database and the script to clean and process the data.  
• The models folder contains the script of the ML model.

**Running the project:**

1) Clone this repository from my github to your machine
2) Make sure you have python 3 and all the libraries installed
3) Type 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db' in the terminal
4) Type 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl' in the terminal
5) From your terminal, go to the app folder and type "python run.py"
6) Copy the link that appeared, and paste it to your internet browser.

There you go! You can see some charts about the Figure Eight data, or you can insert a text, and the model will try to make the classification. 
