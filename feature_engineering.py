import pandas as pd
import os
import itertools
import datetime
from tqdm import tqdm as tqdm


def compute_ttfl(data):
    ttfl = data['pts'] + data[' trb'] + data['ast'] + data['blk'] + data[
        'fg'] + data['fg3'] + data['ft'] - data['tov'] - data['fgm'] - data[
            'fg3m'] - data['ftm']
    return ttfl


def fix_column_dtypes(df, columns):
    sample = df[columns].copy()
    sample = sample.astype(str)
    sample = sample.apply(lambda x: x.split('.')[0])
    return sample


class Season(object):
    def __init__(self, path_data):
        self.data = pd.read_csv(path_data)
        self.teams = self.data.team.unique()

    def feature_cleaning(self, metrics):
        data = self.data.copy()
        data['mp'] = data['mp'].apply(lambda x: int(str(x).split(":")[0]))
        data['ttfl'] = compute_ttfl(data)
        data['date'] = data['date'].astype(str)
        data['fgm'] = data['fga'] - data['fg']
        data['fg3m'] = data['fg3a'] - data['fg3']
        data['ftm'] = data['fta'] - data['ft']
        # Get record & opponent record
        data['record'] = data['wins'] - data['losses']
        data_opp = data.copy()[['team', 'record', 'opp', 'date']]
        data_opp = data_opp.rename(columns={
            'team': 'opp',
            'opp': 'team',
            'record': 'opp_record'
        })
        data_opp = data_opp.drop_duplicates(
            subset=['opp', 'team', 'date', 'opp_record'])
        data = data.merge(data_opp, on=['opp', 'team', 'date'])

        # data.drop(columns='pts_team', inplace=True)
        # Get ID
        teams = list(data.team.unique())
        teams_dict = {team: id for id, team in enumerate(teams)}
        data['team_id'] = data.team.map(teams_dict)
        print('Got ID')

        # Team Record
        team_record = self.get_teams_records(data)
        data = data.merge(team_record, on=['team', 'date'])
        print('Got Team Record')

        # Player record last_week
        player_record = self.get_players_rec(data, metrics, 'week')
        player_record['date'] = fix_column_dtypes(player_record, 'date')
        print(player_record.head(), player_record.shape)
        data = data.merge(player_record,
                          on=['name', 'team', 'team_id', 'date'])
        print('Got PW Player Record')

        # Player record season
        player_record_season = self.get_players_rec(data, metrics, 'season')
        player_record_season['date'] = fix_column_dtypes(
            player_record_season, 'date')
        data = data.merge(player_record_season,
                          on=['name', 'team', 'team_id', 'date'])
        print('Got Season Player Record')

        # Player last game played
        player_last_game = self.get_players_last_game(data)
        player_last_game['date'] = fix_column_dtypes(player_last_game, 'date')
        data = data.merge(player_last_game, on=['name', 'date'])

        return data

    def get_teams_records(self, data):
        teams_list = list(data.team.unique())
        teams_record = []
        for team in tqdm(teams_list, desc='Teams'):
            team_obj = Team(team, data)
            date_list = list(team_obj.data.date.unique())
            for date in tqdm(date_list, desc=team):
                team_record = {}
                wins, losses, total_games = team_obj.record_to_date(
                    to_date=date)
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

    def get_players_rec(self, data, metrics, granularity):

        player_list = list(data.name.unique())
        players_df = []
        for player in tqdm(player_list, desc=f'Player by {granularity}'):
            player_obj = Player(player, data)
            date_list = list(player_obj.data.date.unique())
            for date in tqdm(date_list, desc=player):
                if granularity == 'week':
                    players_df.append(player_obj.weekly_data(date, metrics))
                if granularity == 'season':
                    players_df.append(
                        player_obj.season_stat_to_date(date, metrics))

        return pd.concat(players_df, axis=1, sort=False).T

    def get_players_last_game(self, data):
        player_list = list(data.name.unique())
        players_df = []
        for player in tqdm(player_list, desc=f'Player Last Game'):
            player_obj = Player(player, data)
            player_df = player_obj.data[['name', 'date', 'last_game']]
            players_df.append(player_df)
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
        to_date_dt = datetime.datetime.strptime(to_date, '%Y%m%d')
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        df = df[(df.date < to_date_dt) & (df.mp > 0)]
        groupby_cols = ['date', 'team', 'team_id'] + add_cols

        if df.empty:
            df_empty = pd.DataFrame(columns=groupby_cols +
                                    list(metrics.keys()))
            df_empty.loc[0, [
                'date', 'team', 'team_id'
            ]] = to_date, self.team, self.data.loc[0, 'team_id']
            return df_empty

        df_per_game = df.groupby(groupby_cols, as_index=False).agg(metrics)
        # df_to_date =
        return df_per_game

    def record_to_date(self, to_date):
        if isinstance(to_date, str):
            to_date = datetime.datetime.strptime(to_date, '%Y%m%d')
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        df = df[df.date < to_date]
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

    def current_team(self, to_date):
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        df = df[df.date <= to_date]
        current_team = df[df.date == df.date.max()]['team'].values[0]
        current_team_id = df[df.date == df.date.max()]['team_id'].values[0]

        return current_team, current_team_id

    def get_player_data(self, season_data):
        player_data = season_data[season_data.name == self.name].reset_index(
            drop=True)
        player_data['date_dt'] = pd.to_datetime(player_data.date,
                                                format='%Y%m%d')
        player_data = player_data.sort_values(['date_dt'])
        player_data['last_game'] = (
            player_data['date_dt'] -
            player_data['date_dt'].shift()).dt.days.fillna(7)
        return player_data

    def weekly_data(self, to_date, metrics, add_cols=[]):

        to_date_dt = datetime.datetime.strptime(to_date, '%Y%m%d')
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        curr_team, curr_team_id = self.current_team(to_date_dt)
        # df['week'] = pd.to_datetime(df.date, format='%Y%m%d').dt.week
        # df['year'] = pd.to_datetime(df.date, format='%Y%m%d').dt.year
        # cweek, cyear = to_date_df.isocalendar()[1], to_date_df.isocalendar()[0]
        df_week = df[(datetime.timedelta(days=0) < (to_date_dt - df.date))
                     & (to_date_dt - df.date <= datetime.timedelta(days=7))]
        groupby_cols = ['date', 'name', 'team', 'team_id'] + add_cols
        if df_week.empty:
            df_empty = pd.Series(index=groupby_cols +
                                 [metric + '_lw' for metric in metrics.keys()])
            df_empty['date'] = to_date
            df_empty['name'] = self.name
            df_empty['team'] = curr_team
            df_empty['team_id'] = curr_team_id
            df_empty[[metric + '_lw' for metric in metrics.keys()]] = 0
            return df_empty

        df_metrics = df_week[list(metrics.keys())]
        std = df_metrics.std() if len(df_week) > 1 else 1
        mean = df_metrics.mean()
        agg = ((mean * 0.5 * len(df_week)) / (std))
        agg['date'] = to_date
        agg['name'] = self.name
        agg['team'] = curr_team
        agg['team_id'] = curr_team_id
        agg.index = [metric + '_lw'
                     for metric in metrics.keys()] + groupby_cols

        return agg

    def season_stat_to_date(self, to_date, metrics, add_cols=[]):

        to_date_dt = datetime.datetime.strptime(to_date, '%Y%m%d')
        df = self.data.copy()
        df.date = pd.to_datetime(df.date, format='%Y%m%d')
        curr_team, curr_team_id = self.current_team(to_date_dt)

        df = df[df.date < to_date_dt]
        groupby_cols = ['date', 'name', 'team', 'team_id'] + add_cols

        if df.empty:
            df_empty = pd.Series(index=groupby_cols +
                                 [metric + '_sn' for metric in metrics.keys()])
            df_empty['date'] = to_date
            df_empty['name'] = self.name
            df_empty['team'] = curr_team
            df_empty['team_id'] = curr_team_id
            df_empty[[metric + '_sn' for metric in metrics.keys()]] = 0
            # print('empty', df_empty)
            return df_empty

        df_metrics = df[list(metrics.keys())]
        std = df_metrics.std() if len(df) > 1 else 1
        mean = df_metrics.mean()
        agg = ((mean * 0.5 * len(df)) / (std))
        agg['date'] = to_date
        agg['name'] = self.name
        agg['team'] = curr_team
        agg['team_id'] = curr_team_id
        agg.index = [metric + '_sn'
                     for metric in metrics.keys()] + groupby_cols
        return agg


if __name__ == '__main__':
    path_data = os.path.join(os.getcwd(), 'data', 'season_2018.csv')
    metrics = [
        'mp', 'fg', 'fga', 'fg_pct', 'fgm', 'fg3', 'fg3a', 'fg3_pct', 'fg3m',
        'ft', 'fta', 'ft_pct', 'ftm', 'orb', 'drb', ' trb', 'ast', 'stl',
        'blk', 'tov', 'pf', 'pts', 'plus_minus', 'score', 'record',
        'opp_score', 'opp_record'
    ]
    metrics_agg = {metric: 'mean' for metric in metrics}
    test = Season(path_data)
    test_data = test.feature_cleaning(metrics_agg)

    test_data.to_csv('./data/season_2018_cleaned.csv', index=False)
    print('data cleaned and saved')

# player_test = Player('Curry,Stephen', test.data)

# team_test = Team('GSW', test.data)

# team_test.team_data_to_date('20181201', metrics_agg)

# df = test.data.copy()
# df['mp'] = df['mp'].apply(lambda x: int(str(x).split(":")[0]))
# df['ttfl'] = compute_ttfl(df)
# df['date'] = df.date.astype(str)
# teams = list(df.team.unique())
# teams_dict = {team: id for id, team in enumerate(teams)}
# df['team_id'] = df.team.map(teams_dict)
# team_record = test.get_teams_records(df)
# df_team = df.merge(team_record, on=['team', 'date'])

# player_record = test.get_players_rec(df, metrics_agg, 'week')
# player_record['date'] = fix_column_dtypes(player_record, 'date')
# df_player = df_team.merge(player_record,
#                           on=['name', 'team', 'date', 'team_id'])
# ##
# player_list = list(df_team.name.unique())[:5]
# players_df = []
# for player in player_list:
#     player_obj = Player(player, df_team)
#     date_list = list(player_obj.data.date.unique())
#     print(date_list)
#     for date in date_list:
#         players_df.append(player_obj.weekly_data(date, metrics_agg))
# players_df = pd.concat(players_df, axis=1, sort=False).T
# players_df['date'] = fix_column_dtypes(players_df, 'date')
# print(player_record.head(), player_record.shape)
# df_player = df_team.merge(players_df,
#                 on=['name', 'team', 'team_id', 'date'])

# def anti_join(tableA, tableB, on):

#     #if joining on index, make it into a column
#     if tableB.index.name is not None:
#         dummy = tableB.reset_index()[on]
#     else:
#         dummy = tableB[on]

#     #create a dummy columns of 1s
#     if isinstance(dummy, pd.Series):
#         dummy = dummy.to_frame()

#     dummy.loc[:, 'dummy_col'] = 1

#     #preserve the index of tableA if it has one
#     if tableA.index.name is not None:
#         idx_name = tableA.index.name
#         tableA = tableA.reset_index(drop=False)
#     else:
#         idx_name = None

#     #do a left-join
#     merged = tableA.merge(dummy, on=on, how='left')

#     #keep only the non-matches
#     output = merged.loc[merged.dummy_col.isna(), tableA.columns.tolist()]

#     #reset the index (if applicable)
#     if idx_name is not None:
#         output = output.set_index(idx_name)

#     return (output)

# data_cleaned = pd.read_csv('./data/season_2018_cleaned.csv')