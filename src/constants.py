#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

TARGETS = [
    "fg",
    "fga",
    "fg3",
    "fg3a",
    "ft",
    "fta",
    "orb",
    "drb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pts",
    "trb",
]

VARS = [
    " trb_lw",
    " trb_sn",
    "ast_lw",
    "ast_sn",
    "blk_lw",
    "blk_sn",
    "drb_lw",
    "drb_sn",
    "fg3_lw",
    "fg3_pct_lw",
    "fg3_pct_sn",
    "fg3_sn",
    "fg3a_lw",
    "fg3a_sn",
    "fg_lw",
    "fg_pct_lw",
    "fg_pct_sn",
    "fg_sn",
    "fga_lw",
    "fga_sn",
    "ft_lw",
    "ft_pct_lw",
    "ft_pct_sn",
    "ft_sn",
    "fta_lw",
    "fta_sn",
    "last_game",
    "mp_lw",
    "mp_sn",
    "opp_record",
    "opp_score_lw",
    "opp_score_sn",
    "orb_lw",
    "orb_sn",
    "pf_lw",
    "pf_sn",
    "plus_minus_lw",
    "plus_minus_sn",
    "pts_lw",
    "pts_sn",
    "record",
    "score_lw",
    "score_sn",
    "stl_lw",
    "stl_sn",
    "tot_game",
    "tov_lw",
    "tov_sn",
]

METRICS = [
    "mp",
    "fg",
    "fga",
    "fg_pct",
    "fgm",
    "fg3",
    "fg3a",
    "fg3_pct",
    "fg3m",
    "ft",
    "fta",
    "ft_pct",
    "ftm",
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
    "score",
    "record",
    "opp_score",
    "opp_record",
]

IDS = ["name", "team", "team_id", "date", "opp"]

SEASON_DATES = {
    "2018": ["20181016", "20190410"],
    "2019": ["20191022", "20200415"]
}

TEAM_ID = {
    "GSW": 0,
    "BOS": 1,
    "DEN": 2,
    "PHI": 3,
    "MEM": 4,
    "IND": 5,
    "SAS": 6,
    "NOP": 7,
    "POR": 8,
    "ATL": 9,
    "MIL": 10,
    "CHO": 11,
    "DET": 12,
    "BRK": 13,
    "HOU": 14,
    "LAC": 15,
    "NYK": 16,
    "ORL": 17,
    "MIA": 18,
    "PHO": 19,
    "DAL": 20,
    "UTA": 21,
    "SAC": 22,
    "MIN": 23,
    "TOR": 24,
    "CLE": 25,
    "LAL": 26,
    "WAS": 27,
    "CHI": 28,
    "OKC": 29,
}

# Dirs

MAIN_DIR = os.getcwd().replace("/src", "")
DATA_DIR = os.path.join(MAIN_DIR, "data")
LOGS_DIR = os.path.join(MAIN_DIR, "logs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
