# Disaster-Response

## Introduction
In this project, we'll creat a web app a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
We will creat a machine learning pipeline to categorize these messages and sent to an appropriate disaster relief agency.

## Data
https://www.figure-eight.com/

The files we will use:
 1. disaster_categories.csv
 2. disaster_messages.csv

## Library

1. Numpy and pandas
2.  sqlalchemy
3.  nltk
4.  sklearn
5.  sys
6.  re


## Code files
1. process_data.py : load data from csv files. Save after cleaning.
2. train_classifer.py : load clean data and preprocess the text data. Create a machine learning pipeline include training and 
                        testing. After, save the model.
3. run.py : 


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/





