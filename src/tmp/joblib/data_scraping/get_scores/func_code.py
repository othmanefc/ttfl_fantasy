# first line: 16
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
