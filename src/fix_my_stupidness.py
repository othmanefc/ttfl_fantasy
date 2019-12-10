import os
import pandas as pd
import numpy as np

path = os.path.join(os.getcwd().replace('/src', ''), 'data')
list_files = [f for f in os.listdir(path) if 'season' not in f and '.csv' in f]

for f in list_files:
    print(f)
    df = pd.read_csv(os.path.join(path, f))
    team = []
    for index in range(len(df)):
        for index2 in range(len(df)):
            if np.all([
                    df.loc[index, 'score'] == df.loc[index2, 'opp_score'],
                    df.loc[index, 'opp_score'] == df.loc[index2, 'score']
            ]):
                team.append(df.loc[index2, 'opp'])
                break
    assert (len(team) == len(df))
    df['team'] = team
    df.to_csv(os.path.join(path, f), index=False)

full_df = pd.concat([pd.read_csv(os.path.join(path, f)) for f in list_files])
full_df.to_csv(os.path.join(path, 'season_2018.csv'), index=False)
