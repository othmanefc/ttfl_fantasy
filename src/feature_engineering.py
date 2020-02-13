#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict, Union, Optional, Any, Sequence, Tuple

import pandas as pd
import os
import datetime
from tqdm import tqdm as tqdm
from src.constants import DATA_DIR, METRICS, TEAM_ID, CACHE_DIR
from joblib import Memory

memory_path: str = CACHE_DIR
memory1: Memory = Memory(memory_path, verbose=0)
memory2: Memory = Memory(memory_path, verbose=0)
memory3: Memory = Memory(memory_path, verbose=0)
memory4: Memory = Memory(memory_path, verbose=0)
memory5: Memory = Memory(memory_path, verbose=0)


def split_df_nan(df: pd.DataFrame, cols: List[str]) -> List[pd.DataFrame]:
    cols = [col for col in cols if col in df.columns]
    df_no_nan: pd.DataFrame = df.loc[~df[cols].isna().any(axis=1), :]
    df_nan: pd.DataFrame = df.loc[df[cols].isna().any(axis=1), :]
    return [df_no_nan, df_nan]


def compute_ttfl(df: pd.DataFrame, missed: bool = True) -> Union[int, float]:
    data = df[~df.isna()]
    if missed:
        ttfl_missed: Union[int,
                           float] = (data["pts"] + data[" trb"] + data["ast"] +
                                     data["blk"] + data["fg"] + data["fg3"] +
                                     data["ft"] - data["tov"] - data["fgm"] -
                                     data["fg3m"] - data["ftm"])
        return ttfl_missed
    else:
        ttfl: Union[int, float] = (data["pts"] + data[" trb"] + data["ast"] +
                                   data["blk"] + data["fg"] + data["fg3"] +
                                   data["ft"] - data["tov"] -
                                   (data["fga"] - data["fg"]) -
                                   (data["fg3a"] - data["fg3"]) -
                                   (data["fta"] - data["ft"]))
        return ttfl


def fix_column_dtypes(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    sample: pd.DataFrame = df[columns].copy()
    sample = sample.astype(str)
    sample = sample.apply(lambda x: x.split(".")[0])
    return sample


def team_data_to_date_cache(data: pd.DataFrame,
                            team: str,
                            to_date: str,
                            metrics: Dict[str, Any],
                            add_cols: List[str] = []) -> pd.DataFrame:
    to_date_dt: datetime.datetime = datetime.datetime.strptime(
        to_date, "%Y%m%d")
    df: pd.DataFrame = data.copy()
    # df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df = df[(df.date_dt < to_date_dt) & (df.mp > 0)]
    groupby_cols: List[str] = ["date", "date_dt", "team", "team_id"] + add_cols

    if df.empty:
        df_empty: pd.DataFrame = pd.DataFrame(columns=groupby_cols +
                                              list(metrics.keys()))
        df_empty.loc[0, ["date", "date_dt", "team", "team_id"]] = (
            to_date,
            to_date_dt,
            team,
            data.loc[0, "team_id"],
        )
        return df_empty

    df_per_game: pd.DataFrame = df.groupby(groupby_cols,
                                           as_index=False).agg(metrics)
    # df_to_date =
    return df_per_game


def team_record_to_date_cache(
        data: pd.DataFrame,
        to_date: Union[str, datetime.datetime]) -> Sequence[Union[int, float]]:
    if isinstance(to_date, str):
        to_date = str(int(float(to_date)))  # Make sure there's no dec
        to_date = datetime.datetime.strptime(to_date, "%Y%m%d")
    df: pd.DataFrame = data.copy()
    # df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df = df[df.date_dt < to_date]
    if df.empty:
        return 0, 0, 0
    df.drop_duplicates("date", inplace=True)
    results: Dict = dict(df.result.value_counts())
    try:
        wins: Union[int, float] = results[1]
    except Exception:
        wins = 0
    try:
        losses: Union[int, float] = results[0]
    except Exception:
        losses = 0

    return wins, losses, wins + losses


def player_current_team_cache(data: pd.DataFrame,
                              to_date: datetime.datetime) -> Tuple[str, int]:
    df: pd.DataFrale = data
    # df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df = df[df.date_dt <= to_date]
    current_team: str = df[df.date_dt == df.date_dt.max()]["team"].values[0]
    current_team_id: int = df[df.date_dt ==
                              df.date_dt.max()]["team_id"].values[0]

    return current_team, current_team_id


def player_weekly_data_cache(data: pd.DataFrame,
                             name: str,
                             current_team: Tuple[str, int],
                             to_date: str,
                             metrics: Dict[str, str],
                             add_cols: List[str] = []) -> pd.Series:
    to_date_dt: datetime.datetime = datetime.datetime.strptime(
        to_date, "%Y%m%d")
    df: pd.DataFrame = data.copy()
    # df.date = pd.to_datetime(df.date, format='%Y%m%d')
    curr_team, curr_team_id = current_team
    df_week: pd.DataFrame = df[
        (datetime.timedelta(days=0) < (to_date_dt - df.date_dt))
        & (to_date_dt - df.date_dt <= datetime.timedelta(days=7))]
    groupby_cols: List[str] = ["date", "date_dt", "name", "team", "team_id"
                               ] + add_cols
    if df_week.empty:
        df_empty: pd.Series = pd.Series(
            index=groupby_cols + [metric + "_lw" for metric in metrics.keys()])
        df_empty["date"] = to_date
        df_empty["date_dt"] = to_date_dt
        df_empty["name"] = name
        df_empty["team"] = curr_team
        df_empty["team_id"] = curr_team_id
        df_empty[[metric + "_lw" for metric in metrics.keys()]] = 0
        return df_empty

    df_metrics: pd.DataFrame = df_week[list(metrics.keys())]
    std: pd.Series = df_metrics.std() if len(df_week) > 1 else 1
    mean: pd.Series = df_metrics.mean()
    agg: pd.Series = (mean) / (std)
    agg["date"] = to_date
    agg["date_dt"] = to_date_dt
    agg["name"] = name
    agg["team"] = curr_team
    agg["team_id"] = curr_team_id
    agg.index = [metric + "_lw" for metric in metrics.keys()] + groupby_cols
    return agg


def player_season_stat_to_date_cache(data: pd.DataFrame,
                                     name: str,
                                     current_team: Tuple[str, int],
                                     to_date: str,
                                     metrics: Dict[str, str],
                                     add_cols: List[str] = []) -> pd.Series:
    to_date_dt: datetime.datetime = datetime.datetime.strptime(
        to_date, "%Y%m%d")
    df: pd.DataFrame = data.copy()
    # df.date = pd.to_datetime(df.date, format='%Y%m%d')
    curr_team, curr_team_id = current_team

    df = df[df.date_dt < to_date_dt]
    groupby_cols: List[str] = ["date", "date_dt", "name", "team", "team_id"
                               ] + add_cols

    if df.empty:
        df_empty: pd.Series = pd.Series(
            index=groupby_cols + [metric + "_sn" for metric in metrics.keys()])
        df_empty["date"] = to_date
        df_empty["date_dt"] = to_date_dt
        df_empty["name"] = name
        df_empty["team"] = curr_team
        df_empty["team_id"] = curr_team_id
        df_empty[[metric + "_sn" for metric in metrics.keys()]] = 0
        # print('empty', df_empty)
        return df_empty

    df_metrics: pd.DataFrame = df[list(metrics.keys())]
    std: pd.Series = df_metrics.std() if len(df) > 1 else 1
    mean: pd.Series = df_metrics.mean()
    agg: pd.Series = (mean) / (std)
    agg["date"] = to_date
    agg["date_dt"] = to_date_dt
    agg["name"] = name
    agg["team"] = curr_team
    agg["team_id"] = curr_team_id
    agg.index = [metric + "_sn" for metric in metrics.keys()] + groupby_cols
    return agg


class Season(object):
    def __init__(self,
                 path_data: str = "",
                 data: Optional[pd.DataFrame] = None,
                 read: bool = True) -> None:
        if read:
            self.data: pd.DataFrame = pd.read_csv(path_data)
        else:
            self.data = data

        self.teams: Sequence[str] = self.data.team.unique()

    def feature_cleaning(self, metrics: Dict[str, str]) -> pd.DataFrame:

        data: pd.DataFrame = self.data.copy()
        print("length data beginning fc: ", len(data))
        data["date"] = data["date"].astype(str)

        # Split into Nan and Not nan
        df_no_nan, df_nan = split_df_nan(data, METRICS)
        print('splitted')
        df_no_nan.loc[:, "mp"] = df_no_nan["mp"].apply(
            lambda x: int(str(x).split(":")[0]))
        print('mp changed')

        # Base_stats
        self.float_cols(df_no_nan)
        self.base_stats(df_no_nan)
        df_no_nan.loc[:, "ttfl"] = compute_ttfl(df_no_nan)

        # Concatenate the data back
        data = pd.concat([df_no_nan, df_nan], sort=True)
        print('length data after concat', len(data))

        # Datetime format
        data["date_dt"] = pd.to_datetime(data.date, format="%Y%m%d")
        data["date"] = data["date"].astype(float).astype(int).astype(str)
        # data.drop(columns='pts_team', inplace=True)

        # Get ID
        data["team_id"] = data.team.map(TEAM_ID)
        print("Got ID")

        # Team Record
        team_record: pd.DataFrame = self.get_teams_records(data)
        data = data.merge(team_record, on=["team", "date", "date_dt"])
        print("Got Team Record")
        # Get record
        self.record(data)
        # Get opponents
        data_opp: pd.DataFrame = self.get_opponents(data)
        data = data.merge(data_opp, on=["opp", "team", "date", "date_dt"])
        print("Got opponents")

        # Player record last_week
        player_record: pd.DataFrame = self.get_players_rec(
            data, metrics, "week")
        player_record["date"] = fix_column_dtypes(player_record, "date")
        data = data.merge(player_record,
                          on=["name", "team", "team_id", "date", "date_dt"])
        print("Got PW Player Record")

        # Player record season
        player_record_season: pd.DataFrame = self.get_players_rec(
            data, metrics, "season")
        player_record_season["date"] = fix_column_dtypes(
            player_record_season, "date")
        data = data.merge(player_record_season,
                          on=["name", "team", "team_id", "date", "date_dt"])
        print("Got Season Player Record")

        # Player last game played
        player_last_game: pd.DataFrame = self.get_players_last_game(data)
        player_last_game["date"] = fix_column_dtypes(player_last_game, "date")
        data = data.merge(player_last_game, on=["name", "date", "date_dt"])
        print(data)

        return data

    def float_cols(self, data: pd.DataFrame) -> None:
        cols: List[str] = [col for col in METRICS if col in data.columns]
        float_cols: List[str] = [col for col in cols if type(col) != int]
        data[float_cols] = data[float_cols].astype(float)

    def base_stats(self, data: pd.DataFrame) -> None:
        print('fetching base stats...')
        data.loc[:, "fgm"] = data["fga"] - data["fg"]
        data.loc[:, "fg3m"] = data["fg3a"] - data["fg3"]
        data.loc[:, "ftm"] = data["fta"] - data["ft"]
        print('fetching stats done')

    def record(self, data: pd.DataFrame) -> None:
        data.loc[:, "record"] = data["wins"] - data["losses"]

    def get_opponents(self, data: pd.DataFrame) -> pd.DataFrame:
        data_opp: pd.DataFrame = data.copy()[[
            "team", "record", "opp", "date", "date_dt"
        ]]
        data_opp = data_opp.rename(columns={
            "team": "opp",
            "opp": "team",
            "record": "opp_record"
        })
        data_opp = data_opp.drop_duplicates(
            subset=["opp", "team", "date", "date_dt", "opp_record"])
        return data_opp

    def get_max_date(self, column: str, data: pd.DataFrame) -> str:
        data_already_processed: pd.DataFrame = data[~data[column].isna()]
        max_date: int = data_already_processed.date.astype(int).max()
        return str(max_date)

    def get_teams_records(self, data: pd.DataFrame) -> pd.DataFrame:
        # already = False
        # if 'wins' in data.columns:
        #     max_date = self.get_max_date('wins', data)
        #     data_c = data[data.date > max_date]
        #     already = True
        # else:
        #     max_date = str(data.date.astype(int).min() - 1)
        #     data_c = data.copy()

        teams_list: List[str] = list(data.team.unique())
        teams_record: List[Dict] = []
        for team in tqdm(teams_list, desc="Teams"):
            team_obj: Team = Team(team, data)
            date_list: List[str] = list(team_obj.data.date.unique())
            for date in tqdm(date_list, desc=team):

                team_record: Dict[str,
                                  Optional[Union[float, str,
                                                 datetime.datetime]]] = {}
                wins, losses, total_games = team_obj.record_to_date(
                    to_date=date)
                date = str(int(float(date)))
                team_record.update({
                    "team":
                    team,
                    "wins":
                    wins,
                    "losses":
                    losses,
                    "tot_game":
                    total_games,
                    "date":
                    date,
                    "date_dt":
                    datetime.datetime.strptime(date, "%Y%m%d"),
                })
                teams_record.append(team_record)

        teams_record_df: pd.DataFrame = pd.DataFrame(teams_record)
        # if already:
        #     teams_record_f = pd.concat(
        #         [teams_record, data_c[teams_record.columns]])
        #     return teams_record_f
        return teams_record_df

    def get_players_rec(self, data: pd.DataFrame, metrics: Dict[str, str],
                        granularity: str) -> pd.DataFrame:
        # already = False
        # if 'pts_sn' in data.columns:
        #     max_date = self.get_max_date('pts_sn', data)
        #     data_c = data[data.date > max_date]
        #     already = True
        # else:
        #     max_date = str(data.date.astype(int).min - 1)
        #     data_c = data.copy()

        player_list: List[str] = list(data.name.unique())
        players_df: List[pd.Series] = []
        for player in tqdm(player_list, desc=f"Player by {granularity}"):
            player_obj: Player = Player(player, data)
            date_list: List[str] = list(player_obj.data.date.unique())

            for date in tqdm(date_list, desc=player):
                # if date > max_date:
                if granularity == "week":
                    players_df.append(player_obj.weekly_data(date, metrics))
                if granularity == "season":
                    players_df.append(
                        player_obj.season_stat_to_date(date, metrics))

        players_record: pd.DataFrame = pd.concat(players_df,
                                                 axis=1,
                                                 sort=False).T
        # if already:
        #     players_record_f = pd.concat(
        #         [players_record, data_c[players_record.columns]])
        #     return players_record_f
        return players_record

    def get_players_last_game(self, data: pd.DataFrame) -> pd.DataFrame:
        player_list: List[str] = list(data.name.unique())
        players_df: List[pd.DataFrame] = []
        for player in tqdm(player_list, desc=f"Player Last Game"):
            player_obj: pd.DataFrame = Player(player, data)
            player_df: pd.DataFrame = player_obj.data[[
                "name", "date", "date_dt", "last_game"
            ]]
            players_df.append(player_df)
        return pd.concat(players_df)


class Team(object):
    def __init__(self, team: str, season_data: pd.DataFrame, cached: bool = False) -> None:
        self.team: str = team
        self.data: pd.DataFrame = self.get_team_data(season_data)
        self.cached = cached
        # self.team_id =

    def get_team_data(self, season_data: pd.DataFrame) -> pd.DataFrame:
        team_data = season_data[season_data.team == self.team].reset_index(
            drop=True)
        return team_data

    def team_data_to_date(self,
                          to_date: str,
                          metrics: Dict[str, str],
                          add_cols: List[str] = []) -> pd.DataFrame:
        if self.cached:
            tdtd = memory1.cache(team_data_to_date_cache)
        else:
            tdtd = team_data_to_date_cache
        return tdtd(self.data, self.team, to_date, metrics, add_cols)

    def record_to_date(self, to_date):
        if self.cached:
            trtd = memory2.cache(team_record_to_date_cache)
        else:
            trtd = team_record_to_date_cache

        return trtd(self.data, to_date)


class Player(object):
    def __init__(self, name, season_data, cached: bool = True):
        self.name = name
        self.data = self.get_player_data(season_data)
        self.cached = cached

    def current_team(self, to_date):
        if self.cached:
            pct = memory3.cache(player_current_team_cache)
        else:
            pct = player_current_team_cache
        return pct(self.data, to_date)

    def get_player_data(self, season_data):
        player_data = season_data[season_data.name == self.name].reset_index(
            drop=True)
        player_data = player_data.sort_values(["date_dt"])
        player_data["last_game"] = (
            player_data["date_dt"] -
            player_data["date_dt"].shift()).dt.days.fillna(7)
        return player_data

    def weekly_data(self, to_date, metrics, add_cols=[]):
        to_date_dt = datetime.datetime.strptime(to_date, "%Y%m%d")
        current_team = self.current_team(to_date_dt)
        if self.cached:
            pwd = memory4.cache(player_weekly_data_cache)
        else:
            pwd = player_weekly_data_cache
        return pwd(self.data, self.name, current_team, to_date, metrics,
                   add_cols)

    def season_stat_to_date(self, to_date, metrics, add_cols=[]):
        to_date_dt = datetime.datetime.strptime(to_date, "%Y%m%d")
        current_team = self.current_team(to_date_dt)
        if self.cached:
            sstd = memory5.cache(player_season_stat_to_date_cache)
        else:
            sstd = player_season_stat_to_date_cache
        return sstd(self.data, self.name, current_team, to_date, metrics,
                    add_cols)


if __name__ == "__main__":
    path_data = os.path.join(DATA_DIR, "season_2018.csv")

    metrics_agg = {metric: "mean" for metric in METRICS}
    test = Season(path_data)
    test_data = test.feature_cleaning(metrics_agg)

    test_data.to_csv("season_2018_cleaned.csv", index=False)
    print("data cleaned and saved")
