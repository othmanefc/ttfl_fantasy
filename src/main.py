import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime

from constants import DATA_DIR, SEASON_DATES, METRICS, VARS
from data_scraping import Data_scrapper
import feature_engineering
from nn_model import nn_model
'''
-Season wanted
-dates wanted
-Go check if we have dataset for that Season if not generate it
-If season available, check max date from ds = date wanted
-Potentially predict full week(incrementally)
'''


def check_season_available(season):
    list_files = os.listdir(DATA_DIR)
    file_searched = f'season_{season}.csv'
    file_searched_cleaned = f'season_{season}_cleaned.csv'
    if file_searched_cleaned in list_files:
        return 'Sea available'
    if file_searched in list_files:
        return 'Sea processed available'
    else:
        return 'not available'


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
    path = os.path.join(DATA_DIR, f'season_{season}.csv')
    initial_df = pd.read_csv(path)
    initial_df = initial_df.sort_values('date', ascending=True)
    df_scraped = df_scraped.sort_values('date', ascending=True)
    appended = pd.concat([initial_df, df_scraped])
    if write:
        appended.to_csv(path, index=False)
    return appended


def feature_engineer(df):
    season = feature_engineering.Season(df)
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


def init_stats(players):
    
    return False
