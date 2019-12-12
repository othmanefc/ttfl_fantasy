# Trashtalk Fantasy League Predictor

This repository is aimed at scrapping at predicting best picks for the TrashTalkfantasy league using its scoring model. All the data will be scraped from Basketball reference

# Files:
*data_scraping.py*: It contains an object Data Scrapper that returns the result of a game and the boxscore giving all the essential statistics from a game.

*scraper.py*: It uses the object from data_scraping.py to initialize the data_set for the whole season of nba. For this study only the 2018 season is used as you do not really need old seasons for current predictions. It might be used in the future of the project for applying Transfer Learning.

*feature_enginnering.py*: creating the different features and transforming the dataset returned by the scraper

# How to Run Locally:

## From Virtual Env:

Initialize the environment
```bash
virtualenv -p python3.7 env
source env/bin/activate
pip install -r requirements.txt
```

Activate the environment and run
```bash
source env/bin/activate # If not already in the virtual env
streamlit run app.py
```
## From Docker:

Make sure that Docker is running beforehand
```bash
docker build -t ttfl_fantasy:latest .
docker run -p 8501:8501 ttfl_fantasy:latest
```
