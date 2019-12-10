from bs4 import BeautifulSoup, Comment
import pandas as pd
import re
from urllib.request import urlopen
import os
import datetime
from tqdm import tqdm as tqdm_notebook
import time
from constants import DATA_DIR
from joblib import Memory

memory_path = "./tmp"
memory1 = Memory(memory_path, verbose=0)


def get_scores(date, metrics):

    url_parent = "https://www.basketball-reference.com"
    url = (f"https://www.basketball-reference.com/boxscores/?month="
           f"{date[4:6]}&day={date[6:8]}&year={date[0:4]}")
    soup = BeautifulSoup(urlopen(url), "lxml")
    games = soup.find_all("div", class_="game_summary expanded nohover")
    if len(games) == 0:
        return pd.DataFrame(columns=metrics)
    df_games = []
    for game in tqdm_notebook(games, desc=f"Date: {date}", total=len(games)):
        summary = {}
        # host = game.find_all('table')[1].find_all('a')[1]['href'][7:10]
        # away = game.find_all('table')[1].find_all('a')[0]['href'][7:10]

        winner = game.find("tr", class_="winner").find_all("td")
        loser = game.find("tr", class_="loser").find_all("td")

        summary["winner"] = [
            winner[0].find("a")["href"][7:10],
            int(winner[1].get_text()),
        ]
        summary["loser"] = [
            loser[0].find("a")["href"][7:10],
            int(loser[1].get_text())
        ]
        url_game = url_parent + game.find("a", text="Box Score")["href"]
        soup_game = BeautifulSoup(urlopen(url_game), "lxml")
        box_score = game.find("a", text="Box Score")["href"]
        date = re.findall(r"\d\d\d\d\d\d\d\d", box_score)[0]

        for result, (side, score) in summary.items():
            game_result = soup_game.find("table",
                                         class_="sortable stats_table",
                                         id=f"box-{side}-game-basic")
            player_list = game_result.find_all("tr", class_=None)[1:-1]
            team = []
            for player in player_list:
                player_name = player.find("th")["csk"]
                player_dict = {"name": player_name, "date": date}
                for metric in metrics:
                    try:
                        res = player.find("td", {
                            "data-stat": metric
                        }).contents[0]
                    except Exception:
                        res = 0
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
            team = pd.DataFrame(team)
            team["score"] = score
            df_games.append(pd.DataFrame(team))
    df_games = pd.concat(df_games)
    Data_scrapper.write_csv(df=df_games, name=date)
    return df_games


class Data_scrapper(object):
    def __init__(self, start, end):
        self.metrics = [
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
        self.start = datetime.datetime.strptime(start, "%Y%m%d")
        self.end = datetime.datetime.strptime(end, "%Y%m%d")
        self.timeframe = self.generate_time_frame()

    @staticmethod
    def write_csv(df, name):
        path_data = os.path.join(DATA_DIR)
        if not os.path.exists(path_data):
            os.mkdir(path_data)
        full_path = os.path.join(path_data, f"{name}.csv")
        df.to_csv(full_path, index=False)

    def get_timeframe_data(self,
                           sleep=0,
                           name="default",
                           write=True,
                           get_scores=get_scores):
        full_time_list = []
        for date in tqdm_notebook(self.timeframe,
                                  total=len(self.timeframe),
                                  desc="Main Frame"):
            get_scores = memory1.cache(get_scores)
            date_df = get_scores(date, self.metrics)
            full_time_list.append(date_df)
            time.sleep(sleep)
        full_time_df = pd.concat(full_time_list, sort=True)
        if write:
            Data_scrapper.write_csv(full_time_df, name=name)
        return full_time_df

    def generate_time_frame(self):
        date_range = [
            (self.start + datetime.timedelta(days=x)).strftime("%Y%m%d")
            for x in range(0, (self.end - self.start).days + 1)
        ]
        return date_range

    @staticmethod
    def get_next_games(date, season_year):
        month = datetime.datetime.strptime(date,
                                           "%Y%m%d").strftime("%B").lower()
        url_games = (f"https://www.basketball-reference.com/leagues/"
                     f"NBA_{season_year}_games-{month}.html")
        print(url_games)
        soup = BeautifulSoup(urlopen(url_games), "lxml")
        month_games = soup.find_all("tr")
        match_ups = []
        for month_game in month_games:
            try:
                check_date = month_game.find("th")["csk"].startswith(date)
            except Exception:
                continue

            if check_date:
                visitor = month_game.find("td", {
                    "data-stat": "visitor_team_name"
                }).find("a")["href"][7:10]
                home = month_game.find("td", {
                    "data-stat": "home_team_name"
                }).find("a")["href"][7:10]
                match_ups.append({"home": home, "visitor": visitor})
        return match_ups

    @staticmethod
    def get_all_players(team, date, season_year):
        url = (f"https://www.basketball-reference.com/"
               f"teams/{team}/{season_year}.html")
        print(url)
        soup = BeautifulSoup(urlopen(url), "lxml")
        table_players = soup.find("tbody")
        players = []
        for player in table_players.find_all("tr"):
            name = player.find("td", {"data-stat": "player"})["csk"]
            players.append({"name": name, "team": team, "date": date})
        return players

    @staticmethod
    def get_injured_players(team, date, season_year):
        url = (f"https://www.basketball-reference.com/"
               f"teams/{team}/{season_year}.html")
        soup = BeautifulSoup(urlopen(url), "lxml")
        div_inj = soup.find("div", id="all_injury")
        try:
            comments = div_inj.find_all(
                string=lambda text: isinstance(text, Comment))
            comms = re.sub("\n", "", comments[0]).strip()
            soup = BeautifulSoup(comms, "lxml")
            body = soup.find("tbody")
            players = []
            for player in body.find_all("tr"):
                name = player.find("th", {"data-stat": "player"})["csk"]
                players.append({"name": name, "team": team, "date": date})
            return players
        except Exception:
            return []

    @staticmethod
    def get_next_games_player(date, season_year):
        match_ups = Data_scrapper.get_next_games(date, season_year)
        all_players_list = []
        for match_up in match_ups:
            for i, team in enumerate(match_up.values()):
                all_players = Data_scrapper.get_all_players(
                    team, date, season_year)

                injured_players = Data_scrapper.get_injured_players(
                    team, date, season_year)
                injured_players_names = ([
                    player["name"] for player in injured_players
                ] if len(injured_players) > 0 else [])

                available_players = [
                    player for player in all_players
                    if player["name"] not in injured_players_names
                ]

                for player in available_players:
                    ind = 1 if i == 0 else 0
                    player["opp"] = list(match_up.values())[ind]

                all_players_list.extend(available_players)

        return pd.DataFrame(all_players_list)
