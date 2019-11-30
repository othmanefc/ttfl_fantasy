import os

TARGETS = [
    'fg', 'fga', 'fg3', 'fg3a', 'ft', 'fta', 'orb', 'drb', 'ast', 'stl', 'blk',
    'tov', 'pts', ' trb'
]

VARS = [
    ' trb_lw', ' trb_sn', 'ast_lw', 'ast_sn', 'blk_lw', 'blk_sn', 'drb_lw',
    'drb_sn', 'fg3_lw', 'fg3_pct_lw', 'fg3_pct_sn', 'fg3_sn', 'fg3a_lw',
    'fg3a_sn', 'fg_lw', 'fg_pct_lw', 'fg_pct_sn', 'fg_sn', 'fga_lw', 'fga_sn',
    'ft_lw', 'ft_pct_lw', 'ft_pct_sn', 'ft_sn', 'fta_lw', 'fta_sn',
    'last_game', 'mp_lw', 'mp_sn', 'opp_record', 'opp_score_lw',
    'opp_score_sn', 'orb_lw', 'orb_sn', 'pf_lw', 'pf_sn', 'plus_minus_lw',
    'plus_minus_sn', 'pts_lw', 'pts_sn', 'record', 'score_lw', 'score_sn',
    'stl_lw', 'stl_sn', 'tot_game', 'tov_lw', 'tov_sn'
]

IDS = ['name', 'team', 'team_id', 'date']

#Dirs

MAIN_DIR = os.getcwd().replace('/src', '')
DATA_DIR = os.path.join(MAIN_DIR, 'data')
LOGS_DIR = os.path.join(MAIN_DIR, 'logs')