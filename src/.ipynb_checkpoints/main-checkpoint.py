import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime

from constants import DATA_DIR, SEASON_DATES, METRICS, VARS
from data_scraping import Data_scrapper
from feature_engineering import Season
from nn_model import nn_model

from joblib import Memory
memory_path = './tmp'
memory1 = Memory(memory_path, verbose=0)
memory2 = Memory(memory_path, verbose=0)
memory3 = Memory(memory_path, verbose=0)


def check_season_available(season):
    list_files = os.listdir(DATA_DIR)
    file_searched = f'season_{season}.csv'
    # file_searched_cleaned = f'season_{season}_cleaned.csv'
    if file_searched in list_files:
        return 'Season available'
    else:
        return 'Season not available'


def check_date_available(df, date):
    if 'date_dt' not in df.columns:
        df['date_dt'] = pd.to_datetime(df.date, format='%Y%m%d')
    max_date_av = df.date_dt.max()
    print(max_date_av, date)
    if date - max_date_av == datetime.timedelta(days=1):
        return 1
    elif date - max_date_av == datetime.timedelta(days=0):
        return 0
    else:
        return 2


def get_diff(df1, df2):
    start = df1.date_dt + datetime.timedelta(days=1)
    end = df2.date_dt - datetime.timedelta(days=1)
    start_str = datetime.datetime.strptime(start, '%Y%m%d')
    end_str = datetime.datetime.strptime(end, '%Y%m%d')
    return start_str, end_str


def get_end_date(today, date):
    min = np.minimum(today, date)
    if min == today:
        return datetime.datetime.strftime(today, '%Y%m%d')
    elif min == date:
        return datetime.datetime.strftime(date - datetime.timedelta(days=1),
                                          '%Y%m%d')


# @memory1.cache
def scrape_df(start, end):
    print('Started Scraping...')
    scrapper = Data_scrapper(start, end)
    df = scrapper.get_timeframe_data(sleep=10, write=False)
    df['date_dt'] = pd.to_datetime(df.date, format='%Y%m%d')
    print('Scraping Done')
    return df


def append_df(initial_df, players_df, write=True):
    initial_df = initial_df.sort_values('date', ascending=True)
    players_df = players_df.sort_values('date', ascending=True)
    print(players_df.date_dt.max(), initial_df.date_dt.max())
    assert (players_df.date_dt.max() -
            initial_df.date_dt.max() == datetime.timedelta(days=1))
    appended = pd.concat([initial_df, players_df])
    return appended


@memory2.cache
def load_dataset(season, data):
    data = pd.read_csv(os.path.join(DATA_DIR, f'season_{season}.csv'))
    if 'date_dt' not in data.columns:
        data.date_dt = pd.to_datetime(data.date, format='%Y%m%d')
    data = data[data.date_dt <= date]
    return data


def feature_engineer(df):
    season = Season(data=df, read=False)
    metrics_agg = {metric: 'mean' for metric in METRICS}
    df_engineered = season.feature_cleaning(metrics_agg)
    return df_engineered


def run_model(X, y, saved_model=True):
    nn_mod = nn_model(X, y)
    if saved_model:
        model = nn_mod.run_saved_model()
    else:
        model = nn_mod.run_models()
    return model


def players_available(date, season_year):
    print('Getting available players...')
    date = datetime.datetime.strftime(date, '%Y%m%d')
    players = Data_scrapper.get_next_games_player(date, season_year)
    players['date_dt'] = pd.to_datetime(players.date, format='%Y%m%d')
    print('Getting players done')
    return players


def init_stats(df):
    return False


@memory3.cache
def get_df_prediction(date, season, season_year):
    if check_season == 'Season available':
        df = load_dataset(season, date)
        print('check date:', check_date_available(df, date))
        if check_date_available(df, date) == 1:
            players = players_available(date, season_year)
            appended = append_df(df, players)
            df_to_predict = feature_engineer(appended)

        elif check_date_available(df, date) == 0:
            df_to_predict = feature_engineer(df)

        elif check_date_available(df, date) == 2:
            players = players_available(date, season_year)
            start, end = get_diff(df, players)
            scraped = scrape_df(start, end)
            df_f = append_df(df, scraped)
            appended = append_df(df_f, players)
            df_to_predict = feature_engineer(appended)

    elif check_season == 'Season not available':
        end = get_end_date(today_dt, date)
        print(end)
        scraped = scrape_df(season_start, end)
        players = players_available(date, season_year)
        appended = append_df(scraped, players)
        df_to_predict = feature_engineer(appended)

    return df_to_predict


def split_df(df, date_wanted):
    assert (df.date_dt.max() == date_wanted)
    df_train = df[df.date_dt < date_wanted]
    df_test = df[df.date_dt == date_wanted]
    X_train, X_test = df_train[VARS], df_test[VARS]
    y_train = df_train['ttfl']
    return X_train, X_test, y_train


def predict(model, X):
    y = model.predict(X)
    df = pd.concat([X, y], axis=1)
    df.columns = list(X.columns) + ['ttfl']
    return df


def main(season_selected, date_selected, previous_seasons, season_year):
    # Get dataset
    with st.spinner('Building dataset...'):
        df_to_predict = get_df_prediction(date_selected, season_selected,
                                          season_year)
    st.success('Dataset built')
    # Split
    X_train, X_test, y_train = split_df(df_to_predict, date_selected)
    # Train or retrain
    with st.spinner('Training model'):
        model = run_model(X_train, y_train, previous_seasons)
    # predict
    predicted_df = predict(model, X_test)
    predicted_df = predicted_df.sort_values('ttfl', ascending=False)
    st.write(predicted_df)
    st.balloons()
    return predicted_df


###
# Sidebar
available_seasons = list(SEASON_DATES.keys())
season_selected = st.sidebar.selectbox('Season', available_seasons, index=1)
season_selected_tf = SEASON_DATES[season_selected]
season_start, season_end = season_selected_tf
season_start_dt = datetime.datetime.strptime(season_start, "%Y%m%d")
season_end_dt = datetime.datetime.strptime(season_end, "%Y%m%d")
season_year = season_end[0:4]

check_season = check_season_available(season_selected)
st.sidebar.markdown(f'**{check_season}**')
st.sidebar.markdown(f'Season goes from **{season_start_dt.date()}** to \n'
                    f'**{season_end_dt.date()}**')

today = st.sidebar.markdown(
    f"Today's date: **{datetime.datetime.now().date()}**")
today_dt = datetime.datetime.now()
date = st.sidebar.date_input('prediction date:', season_end_dt)
date = datetime.datetime.combine(date, datetime.time())
if (date < season_start_dt) or (date > season_end_dt):
    st.error('Cannot predict outside the season, there is no game available')

previous_seasons = st.sidebar.checkbox('Predict on used model')

predict_btn = st.sidebar.button('🚀 Predict!')
stop_btn = st.sidebar.button('Stop')

if predict_btn:
    main(season_selected, date, previous_seasons, season_year)
if stop_btn:
    os._exit(0)