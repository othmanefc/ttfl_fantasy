import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime

from constants import DATA_DIR, SEASON_DATES, METRICS, VARS
from data_scraping import Data_scrapper
import feature_engineering
from nn_model import nn_model


def check_season_available(season):
    list_files = os.listdir(DATA_DIR)
    file_searched = f'season_{season}.csv'
    file_searched_cleaned = f'season_{season}_cleaned.csv'

    if file_searched in list_files:
        return 'Season processed available'
    if file_searched_cleaned in list_files:
        return 'Season available'
    else:
        return 'Season not available'


def check_date_available(df, date):
    if 'date_dt' not in df.columns:
        df['date_dt'] = pd.to_datetime(df.date, format='%Y%m%d')
    max_date_av = df.date_dt.max()
    return date - max_date_av


def scrape_df(start, end, date, season):
    scrapper = Data_scrapper(start, end)
    df = scrapper.get_timeframe_data(sleep=10, write=False)
    return df


def append_df(season, df_scraped, write=True):
    initial_df = pd.read_csv(path)
    initial_df = initial_df.sort_values('date', ascending=True)
    df_scraped = df_scraped.sort_values('date', ascending=True)
    appended = pd.concat([initial_df, df_scraped])
    if write:
        appended.to_csv(path, index=False)
    return appended

def load_dataset(season, data):
    data = pd.read_csv(os.path.join(DATA_DIR, f'season_{season}.csv'))
    data = data[data.date < date]
    return data

def feature_engineer(df):
    season = Season(data=df, read=False)
    metrics_agg = {metric: 'mean' for metric in METRICS}
    df_engineered = season.feature_cleaning(metrics_agg)
    return df_engineered


def run_model(df):
    X, y = df[VARS], df['ttfl']
    nn_mod = nn_model(X, y)
    model = nn_mod.run_saved_model(X, y)
    return model


def players_availables(date):
    players = Data_scrapper.get_next_games_player(date)
    return players


def init_stats(df):
    return False


###
# Sidebar
available_seasons = list(SEASON_DATES.keys())
season_selected = st.sidebar.selectbox('Season', available_seasons, index=1)
season_selected_tf = SEASON_DATES[season_selected]
season_start, season_end = season_selected_tf
season_start_dt = datetime.datetime.strptime(season_start, "%Y%m%d")
season_end_dt = datetime.datetime.strptime(season_end, "%Y%m%d")

check_season = check_season_available(season_selected)
st.sidebar.markdown(f'**{check_season}**')

if check_season == 'Season available':
    load_dataset()
    clean_dataset()

today = st.sidebar.markdown(f"today's date: {datetime.datetime.now().date()}")
date = st.sidebar.date_input('prediction date desired', 
                                season_end_dt)
if date > season_end_dt.date():
    st.error('Cannot predict outside the season, there is no game available')

check_date = check_date_available(season)

previous_seasons = st.sidebar.checkbox('Predict on used model')

predict_btn = st.sidebar.button('Predict!')

def predict(date, season):
    players = players_availables(date)
    if check_season=='Season available':
        df = load_dataset(season, date)
        append_df()
        feature_engineer()

    if check_season=='Season not available':
        scrape_df()
        players_availables()
        append_df()
        feature_engineer()
















