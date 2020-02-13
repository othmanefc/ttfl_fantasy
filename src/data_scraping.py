#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Union, Optional, Callable, Sequence

from bs4 import BeautifulSoup, Comment, element
import pandas as pd
import re
from urllib.request import urlopen
import os
import datetime
from tqdm import tqdm as tqdm_notebook
import time
from src.constants import DATA_DIR


def get_scores(date: str, metrics: List[str]) -> pd.DataFrame:
    path_check = os.path.join(DATA_DIR, "dates", f"{date}.csv")
    if os.path.exists(path_check):
        df_games = pd.read_csv(path_check)
        return df_games

    url_parent: str = "https://www.basketball-reference.com"
    url: str = (f"https://www.basketball-reference.com/boxscores/?month="
                f"{date[4:6]}&day={date[6:8]}&year={date[0:4]}")
    soup: BeautifulSoup = BeautifulSoup(urlopen(url), "lxml")
    games: Sequence[Optional[element.Tag]] = soup.find_all(
        "div", class_="game_summary expanded nohover")
    if len(games) == 0:
        return pd.DataFrame(columns=metrics)
    df_games: List[Any] = []
    for game in tqdm_notebook(games, desc=f"Date: {date}", total=len(games)):
        summary: Dict[str, List[Any]] = {}
        # host = game.find_all('table')[1].find_all('a')[1]['href'][7:10]
        # away = game.find_all('table')[1].find_all('a')[0]['href'][7:10]

        winner: Sequence[Optional[element.Tag]] = game.find(
            "tr", class_="winner").find_all("td")
        loser: Sequence[Optional[element.Tag]] = game.find(
            "tr", class_="loser").find_all("td")

        summary["winner"] = [
            winner[0].find("a")["href"][7:10],
            int(winner[1].get_text()),
        ]
        summary["loser"] = [
            loser[0].find("a")["href"][7:10],
            int(loser[1].get_text())
        ]
        url_game: str = url_parent + game.find("a", text="Box Score")["href"]
        soup_game: BeautifulSoup = BeautifulSoup(urlopen(url_game), "lxml")
        box_score: Optional[element.Tag] = game.find("a",
                                                     text="Box Score")["href"]
        date = re.findall(r"\d\d\d\d\d\d\d\d", box_score)[0]

        for result, (side, score) in summary.items():
            game_result: Optional[element.Tag] = soup_game.find(
                "table",
                class_="sortable stats_table",
                id=f"box-{side}-game-basic")
            player_list: List[Any] = game_result.find_all("tr",
                                                          class_=None)[1:-1]
            team: List[Dict[str, Optional[Union[float, int, str]]]] = []
            for player in player_list:
                player_name: Optional[str] = player.find("th")["csk"]
                player_dict: Dict[str, Optional[Union[str, int, str]]] = {
                    "name": player_name,
                    "date": date
                }
                for metric in metrics:
                    try:
                        res: Union[str, int, float] = player.find(
                            "td", {
                                "data-stat": metric
                            }).contents[0]
                    except Exception:
                        res: Union[str, int, float] = 0
                    player_dict.update({metric: res})
                if result == "winner":
                    player_dict.update({
                        "result": 1,
                        "score": score,
                        "team": summary["winner"][0],
                        "opp": summary["loser"][0],
                        "opp_score": summary["loser"][1],
                    })
                if result == "loser":
                    player_dict.update({
                        "result": 0,
                        "score": score,
                        "team": summary["winner"][0],
                        "opp": summary["winner"][0],
                        "opp_score": summary["winner"][1],
                    })
                if int(str(player_dict["mp"]).split(":")[0]) >= 10:
                    team.append(player_dict)
            team_df: pd.DataFrame = pd.DataFrame(team)
            team_df["score"] = score
            df_games.append(pd.DataFrame(team_df))
    df_games_df: pd.DataFrame = pd.concat(df_games)
    if ' trb' in df_games_df.columns:
        df_games_df.rename({' trb': 'trb'}, inplace=True)

    Data_scrapper.write_csv(df=df_games_df, name=date, extra_path="dates")
    return df_games_df


class Data_scrapper(object):
    def __init__(self, start: str, end: str) -> None:
        self.metrics: List[str] = [
            "mp",
            "fg",
            "fga",
            "fg_pct",
            "fg3",
            "fg3a",
            "fg3_pct",
            "ft",
            "fta",
            "ft_pct",
            "orb",
            "drb",
            " trb",
            "ast",
            "stl",
            "blk",
            "tov",
            "pf",
            "pts",
            "plus_minus",
        ]
        self.start: datetime.datetime = datetime.datetime.strptime(
            start, "%Y%m%d")
        self.end: datetime.datetime = datetime.datetime.strptime(end, "%Y%m%d")
        self.timeframe: pd.DataFrame = self.generate_time_frame()

    @staticmethod
    def write_csv(df: pd.DataFrame, name: str, extra_path: str = None) -> None:
        if extra_path is not None:
            path_data: str = os.path.join(DATA_DIR, extra_path)
        else:
            path_data = os.path.join(DATA_DIR)

        if not os.path.exists(path_data):
            os.mkdir(path_data)
        full_path: str = os.path.join(path_data, f"{name}.csv")
        df.to_csv(full_path, index=False)

    def get_timeframe_data(self,
                           sleep: int = 0,
                           name: str = "default",
                           write: bool = True,
                           get_scores: Callable = get_scores) -> pd.DataFrame:
        full_time_list: List[pd.DataFrame] = []
        for date in tqdm_notebook(self.timeframe,
                                  total=len(self.timeframe),
                                  desc="Main Frame"):
            # get_scores_cached: Callable = memory1.cache(get_scores)
            # date_df: pd.DataFrame = get_scores_cached(date, self.metrics)
            date_df: pd.DataFrame = get_scores(date, self.metrics)
            full_time_list.append(date_df)
            time.sleep(sleep)
        full_time_df: pd.DataFrame = pd.concat(full_time_list, sort=True)
        if write:
            Data_scrapper.write_csv(full_time_df, name=name)
        return full_time_df

    def generate_time_frame(self) -> List[str]:
        date_range: List[str] = [
            (self.start + datetime.timedelta(days=x)).strftime("%Y%m%d")
            for x in range(0, (self.end - self.start).days + 1)
        ]
        return date_range

    @staticmethod
    def get_next_games(
            date: str,
            season_year: Union[str, int]) -> List[Dict[str, Optional[str]]]:
        month: str = datetime.datetime.strptime(
            date, "%Y%m%d").strftime("%B").lower()
        url_games: str = (f"https://www.basketball-reference.com/leagues/"
                          f"NBA_{season_year}_games-{month}.html")
        print(url_games)
        soup: BeautifulSoup = BeautifulSoup(urlopen(url_games), "lxml")
        month_games: Sequence[Any] = soup.find_all("tr")
        match_ups: List[Dict[str, Optional[str]]] = []
        for month_game in month_games:
            try:
                check_date: bool = month_game.find("th")["csk"].startswith(
                    date)
            except Exception:
                continue

            if check_date:
                visitor: Optional[str] = month_game.find(
                    "td", {
                        "data-stat": "visitor_team_name"
                    }).find("a")["href"][7:10]
                home: Optional[str] = month_game.find(
                    "td", {
                        "data-stat": "home_team_name"
                    }).find("a")["href"][7:10]
                match_ups.append({"home": home, "visitor": visitor})
        return match_ups

    @staticmethod
    def get_all_players(
            team: Optional[str], date: str,
            season_year: Union[str, int]) -> List[Dict[str, Optional[str]]]:
        url: str = (f"https://www.basketball-reference.com/"
                    f"teams/{team}/{season_year}.html")
        print(url)
        soup: BeautifulSoup = BeautifulSoup(urlopen(url), "lxml")
        table_players: Optional[element.Tag] = soup.find("tbody")
        players: List[Dict[str, Optional[element.Tag]]] = []
        for player in table_players.find_all("tr"):
            name: Optional[str] = player.find("td",
                                              {"data-stat": "player"})["csk"]
            players.append({"name": name, "team": team, "date": date})
        return players

    @staticmethod
    def get_injured_players(team: Optional[str], date: str,
                            season_year: Union[str, int]) -> List:
        url: str = (f"https://www.basketball-reference.com/"
                    f"teams/{team}/{season_year}.html")
        soup: BeautifulSoup = BeautifulSoup(urlopen(url), "lxml")
        div_inj: Optional[element.Tag] = soup.find("div", id="all_injury")
        try:
            comments: Sequence[Optional[element.Tag]] = div_inj.find_all(
                string=lambda text: isinstance(text, Comment))
            comms: Optional[str] = re.sub("\n", "", comments[0]).strip()
            soup = BeautifulSoup(comms, "lxml")
            body: Optional[element.Tag] = soup.find("tbody")
            players: List[Dict[str, Optional[str]]] = []
            for player in body.find_all("tr"):
                name: Optional[str] = player.find(
                    "th", {"data-stat": "player"})["csk"]
                players.append({"name": name, "team": team, "date": date})
            return players
        except Exception:
            return list()

    @staticmethod
    def get_next_games_player(date: str,
                              season_year: Union[str, int]) -> pd.DataFrame:
        match_ups: List[Dict[str,
                             Optional[str]]] = Data_scrapper.get_next_games(
                                 date, season_year)
        all_players_list: List = []
        for match_up in match_ups:
            for i, team in enumerate(match_up.values()):
                all_players: List[Dict[
                    str, Optional[str]]] = Data_scrapper.get_all_players(
                        team, date, season_year)

                injured_players: List = Data_scrapper.get_injured_players(
                    team, date, season_year)
                injured_players_names: List = ([
                    player["name"] for player in injured_players
                ] if len(injured_players) > 0 else [])

                available_players: List = [
                    player for player in all_players
                    if player["name"] not in injured_players_names
                ]

                for player in available_players:
                    ind: int = 1 if i == 0 else 0
                    player["opp"] = list(match_up.values())[ind]

                all_players_list.extend(available_players)

        return pd.DataFrame(all_players_list)
