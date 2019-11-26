import pandas as pd
import os
import itertools
import datetime



class Season(object):
    def __init__(self, path_data):
        self.data = pd.read_csv(path_data)
        self.teams = self.data.team.unique()

    def feature_cleaning(self, metrics):
        data = self.data.copy()
        data['mp'] = data['mp'].apply(lambda x: int(str(x).split(":")[0]))
        data['ttfl'] = data['pts'] + data[' trb'] + data['ast'] + data[
            'blk'] + data['fg'] + data['fg3'] + data['ft'] - data['tov'] - (
                data['fga'] - data['fg']) - (data['fg3a'] - data['fg3']) - (
                    data['fta'] - data['ft'])
        data['date'] = data['date'].astype(str)
        # data.drop(columns='pts_team', inplace=True)
        # Get ID
        teams = list(data.team.unique())
        teams_dict = {team: id for id, team in enumerate(teams)}
        data['team_id'] = data.team.map(teams_dict)
        # Team Record
        team_record = self.get_teams_records(data)
        data = data.merge(team_record, on=['team', 'date'])
        # Player record
        player_record = self.get_players_weekly_rec(data, metrics)
        print(player_record)
        data = data.merge(player_record, on=['name', 'team', 'date'])
        print(data)
        return data

    def get_teams_records(self, data):
        teams_list = list(data.team.unique())
        date_list = list(data.date.unique())
        teams_record = []
        for team, date in itertools.product(teams_list, date_list):
            team_record = {}
            team_obj = Team(team, data)
            wins, losses, total_games = team_obj.record_to_date(to_date=date)
            team_record.update({
                'team': team,
                'wins': wins,
                'losses': losses,
                'tot_game': total_games,
                'date': date
            })
            teams_record.append(team_record)
        teams_record = pd.DataFrame(teams_record)
        return teams_record

    def get_players_weekly_rec(self, data, metrics):
        player_list = list(data.name.unique())
        # date_list = list(data.date.unique())
        # players_record = []
        # for player, date in itertools.product(player_list, date_list):
        #     player_record = {}
        #     player_obj = Player(player, data)
        players_df = []
        for player in player_list:
            player_obj = Player(player, data)
            players_df.append(player_obj.weekly_data(metrics))
        return pd.concat(players_df)


class Team(object):
    def __init__(self, team, season_data):
        self.team = team
        self.data = self.get_team_data(season_data)
        # self.team_id =

    def get_team_data(self, season_data):
        team_data = season_data[season_data.team == self.team].reset_index(
            drop=True)
        return team_data

    def team_data_to_date(self, to_date, metrics, add_cols=[]):
        if isinstance(to_date, str):
            to_date = datetime.datetime.strptime(to_date, '%Y%m%d')
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        df = df[(df.date <= to_date) & (df.mp > 0)]
        groupby_cols = ['date', 'team', 'team_id'] + add_cols
        df_per_game = df.groupby(groupby_cols, as_index=False).agg(metrics)
        # df_to_date =
        return df_per_game

    def record_to_date(self, to_date):
        if isinstance(to_date, str):
            to_date = datetime.datetime.strptime(to_date, '%Y%m%d')
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        df = df[df.date <= to_date]
        if df.empty:
            return 0, 0, 0
        df.drop_duplicates('date', inplace=True)
        results = dict(df.result.value_counts())
        try:
            wins = results[1]
        except:
            wins = 0
        try:
            losses = results[0]
        except:
            losses = 0

        return wins, losses, wins + losses


class Player(object):
    def __init__(self, name, season_data):
        self.name = name
        self.data = self.get_player_data(season_data)

    def get_player_data(self, season_data):
        player_data = season_data[season_data.name == self.name]
        return player_data

    def weekly_data(self, metrics, add_cols=[]):
        df = self.data.copy()
        df['week'] = pd.to_datetime(df.date, format='%Y%m%d').dt.week
        df['year'] = pd.to_datetime(df.date, format='%Y%m%d').dt.year
        groupby_cols = ['name', 'date', 'team', 'team_id'] + add_cols
        agg = df.groupby(groupby_cols, as_index=False).agg(metrics)
        agg.columns = groupby_cols + [metric + '_lw' for metric in metrics.keys()]
        return agg


if __name__='__main__':
    path_data = os.path.join(os.getcwd(), 'data', 'season_2018.csv')
    metrics = [
        'mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct', 'ft', 'fta',
        'ft_pct', 'orb', 'drb', ' trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts',
        'plus_minus', 'score', 'opp_score'
    ]
    metrics_agg = {metric: 'mean' for metric in metrics}
    test = Season(path_data)
    test_data = test.feature_cleaning(metrics_agg)

    test_data.to_csv('./data/season_2018_cleaned.csv', index=False)
    print('data cleaned and saved')
# player_test = Player('Curry,Stephen', test.data)

# team_test = Team('GSW', test.data)

# team_test.team_data_to_date('20181201', metrics_agg)