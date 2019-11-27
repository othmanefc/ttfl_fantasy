from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.request import urlopen
import os
import datetime
from tqdm import tqdm as tqdm_notebook
import time


class Data_scrapper(object):
    def __init__(self, start, end):
        self.metrics = [
            'mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct', 'ft', 'fta',
            'ft_pct', 'orb', 'drb', ' trb', 'ast', 'stl', 'blk', 'tov', 'pf',
            'pts', 'plus_minus'
        ]
        self.start = datetime.datetime.strptime(start, "%Y%m%d")
        self.end = datetime.datetime.strptime(end, "%Y%m%d")
        self.timeframe = self.generate_time_frame()

    def get_scores(self, date):

        url_parent = "https://www.basketball-reference.com"
        url = f"https://www.basketball-reference.com/boxscores/?month={date[4:6]}&day={date[6:8]}&year={date[0:4]}"
        soup = BeautifulSoup(urlopen(url), 'lxml')
        games = soup.find_all('div', class_='game_summary expanded nohover')
        if len(games) == 0:
            return pd.DataFrame(columns=self.metrics)
        df_games = []
        for game in tqdm_notebook(games,
                                  desc=f'Date: {date}',
                                  total=len(games)):
            summary = {}
            # host = game.find_all('table')[1].find_all('a')[1]['href'][7:10]
            # away = game.find_all('table')[1].find_all('a')[0]['href'][7:10]

            winner = game.find('tr', class_='winner').find_all('td')
            loser = game.find('tr', class_='loser').find_all('td')

            summary['winner'] = [
                winner[0].find('a')['href'][7:10],
                int(winner[1].get_text())
            ]
            summary['loser'] = [
                loser[0].find('a')['href'][7:10],
                int(loser[1].get_text())
            ]
            url_game = url_parent + game.find('a', text='Box Score')['href']
            soup_game = BeautifulSoup(urlopen(url_game), 'lxml')
            box_score = game.find('a', text='Box Score')['href']
            date = re.findall(r'\d\d\d\d\d\d\d\d', box_score)[0]

            for result, (side, score) in summary.items():
                game_result = soup_game.find('table',
                                             class_='sortable stats_table',
                                             id=f'box-{side}-game-basic')
                player_list = game_result.find_all('tr', class_=None)[1:-1]
                team_result = game_result.find_all('tr', class_=None)[-1:][0]
                team = []
                for player in player_list:
                    player_name = player.find('th')['csk']
                    player_dict = {'name': player_name, 'date': date}
                    for metric in self.metrics:
                        try:
                            res = player.find('td', {
                                'data-stat': metric
                            }).contents[0]
                        except:
                            res = 0
                        player_dict.update({metric: res})
                    if result == 'winner':
                        player_dict.update({'result': 1, 'score': score,
                                    'opp': summary['loser'][0],
                                    'opp_score': summary['loser'][1]})
                    if result == 'loser':
                        player_dict.update({'result': 0, 'score': score, 
                                    'opp': summary['winner'][0], 
                                    'opp_score': summary['winner'][1]})
                    if int(str(player_dict['mp']).split(':')[0]) >= 10:
                        team.append(player_dict)
                team = pd.DataFrame(team)
                team['score'] = score
                team['pts_team'] = team_result.find('td', {'data_stat': 'fg'})
                df_games.append(pd.DataFrame(team))
        df_games = pd.concat(df_games)
        self.write_csv(df=df_games, name=date)
        return df_games

    def write_csv(self, df, name):
        current_dir = os.getcwd()
        path_data = os.path.join(current_dir, 'data')
        if not os.path.exists(path_data):
            os.mkdir(path_data)
        full_path = os.path.join(path_data, f'{name}.csv')
        df.to_csv(full_path, index=False)

    def get_timeframe_data(self, sleep=0, name='default'):
        full_time_list = []
        for date in tqdm_notebook(self.timeframe,
                                  total=len(self.timeframe),
                                  desc='Main Frame'):
            date_df = self.get_scores(date)
            full_time_list.append(date_df)
            time.sleep(sleep)
        full_time_df = pd.concat(full_time_list)
        self.write_csv(full_time_df, name=name)
        return full_time_df

    def generate_time_frame(self):
        date_range = [
            (self.start + datetime.timedelta(days=x)).strftime('%Y%m%d')
            for x in range(0, (self.end - self.start).days)
        ]
        return date_range
