import os
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time

# Set up the Chrome WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--log-level=3")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)

# List of leagues with English URLs
leagues = [
    ('https://fbref.com/en/comps/9/Premier-League-Stats', 'Premier-League', '9'),
    ('https://fbref.com/en/comps/12/La-Liga-Stats', 'La-Liga', '12'),
    ('https://fbref.com/en/comps/20/Bundesliga-Stats', 'Bundesliga', '20'),
    ('https://fbref.com/en/comps/11/Serie-A-Stats', 'Serie-A', '11'),
    ('https://fbref.com/en/comps/13/Ligue-1-Stats', 'Ligue-1', '13')
]

# Function to extract a player statistics table by XPath
def extract_player_table(xpath, headers):
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
        table = driver.find_element(By.XPATH, xpath)
        soup = BeautifulSoup(table.get_attribute("outerHTML"), 'html.parser')

        rows = []
        for row in soup.find_all('tr', attrs={"data-row": True}):
            cells = row.find_all(['th', 'td'])
            row_data = []
            link_player = None
            link_matches = None

            anchors = row.find_all('a', href=True)
            if anchors:
                link_player = anchors[0]['href']
                if len(anchors) > 1:
                    link_matches = anchors[-1]['href']

            for cell in cells:
                row_data.append(cell.get_text(strip=True))

            row_data.extend([link_player, link_matches])
            rows.append(row_data)

        if rows:
            assert len(rows[0]) == len(headers), f"Mismatch in column count: {len(rows[0])} vs {len(headers)}"

        df = pd.DataFrame(rows, columns=headers)
        return df

    except (TimeoutException, NoSuchElementException):
        print(f"⚠️ Failed to load table: {xpath}")
        return None

# Main scraping loop per league
for league_url, league_name, league_code in leagues:
    print(f"\n⚽ Scraping league: {league_name}")
    driver.get(league_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, f'//*[@id="results2024-2025{league_code}1_overall"]')))

    try:
        standard_path = os.path.join("data", "fbref", league_name.replace(" ", "-").lower(), league_name.replace(" ", "-").lower() + "_stats", "standard.csv")
        df_teams = pd.read_csv(standard_path)
    except FileNotFoundError:
        print(f"❌ Missing standard.csv for {league_name}")
        continue

    for team, link in zip(df_teams['Squad'], df_teams['link']):
        team_url = f"https://fbref.com{link}"
        print(f"\n🏟️  Scraping team: {team} -> {team_url}")

        driver.get(team_url)
        time.sleep(5)

        output_team_dir = os.path.join("data", "fbref", league_name.replace(" ", "-").lower(), f"{league_name.lower()}_teams", team.replace(" ", "_"))
        stats_dir = os.path.join(output_team_dir, "team_stats")
        os.makedirs(stats_dir, exist_ok=True)

        # 🖼️ Download team logo
        try:
            logo_img = driver.find_element(By.CSS_SELECTOR, "img.teamlogo")
            logo_url = logo_img.get_attribute("src")
            logo_ext = os.path.splitext(logo_url)[-1]
            logo_path = os.path.join(output_team_dir, f"logo{logo_ext}")
            with open(logo_path, "wb") as f:
                f.write(requests.get(logo_url).content)
            print("🖼️  Logo saved")
        except:
            print("⚠️ Logo not found")

        # Define tables to scrape from team page
        tables_info = {
            'standard': (f'//*[@id="stats_standard_{league_code}"]', ["Player", "Nation", "Pos", "Age", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "npxG+xAG", "PrgC", "PrgP", "PrgR", "Gls/90", "Ast/90", "G+A/90", "G-PK/90", "G+A-PK/90", "xG/90", "xAG/90", "xG+xAG/90", "npxG/90", "npxG+xAG/90", "Matches", "link_player", "link_matches"]),
            'matchlogs': (f'//*[@id="matchlogs_for"]', ["Date", "Time", "Competition", "Round", "Day", "Venue", "Result", "GF", "GA", "Opponent", "xG", "xGA", "Poss", "Attendance", "Captain", "Formation", "OppFormation", "Referee", "MatchReport", "Notes", "link_player", "link_matches"]),
            'goalkeeping': (f'//*[@id="stats_keeper_{league_code}"]', ["Player", "Nation", "Pos", "Age", "MP", "Starts", "Min", "90s", "GA", "GA90", "SoTA", "Saves", "Save%", "W", "D", "L", "CS", "CS%", "PKatt", "PKA", "PKsv", "PKm", "PKSave%", "Matches", "link_player", "link_matches"]),
            'goalkeeping_advanced': (f'//*[@id="stats_keeper_adv_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "GA", "PD", "TL", "TE", "OG", "PSxG", "PSxG/SoT", "PSxG+/-", "PSxG+/-/90", "Cmp", "Int", "Cmp%", "PassAttGK", "ThrAttGK", "Launch%GK", "AvgLenGK", "GoalKicks", "Launch%GKicks", "AvgLenGKicks", "CrossesFaced", "CrossesStopped", "CrossesStopped%", "#OPA", "#OPA/90", "AvgDistOPA", "Matches", "link_player", "link_matches"]),
            'shooting': (f'//*[@id="stats_shooting_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "Gls", "Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "Dist", "FK", "PK", "PKatt", "xG", "npxG", "npxG/Sh", "G-xG", "np:G-xG", "Matches", "link_player", "link_matches"]),
            'passing': (f'//*[@id="stats_passing_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "Cmp", "Att", "Cmp%", "TotDist", "PrgDist", "CmpShort", "AttShort", "Cmp%Short", "CmpMedium", "AttMedium", "Cmp%Medium", "CmpLong", "AttLong", "Cmp%Long", "Ast", "xAG", "xA", "A-xAG", "KP", "FinalThird", "PPA", "CrsPA", "PrgP", "Matches", "link_player", "link_matches"]),
            'pass_types': (f'//*[@id="stats_passing_types_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "Att", "Live", "Dead", "FK", "TB", "Sw", "Crs", "TI", "CK", "CKIn", "CKOut", "CKStr", "Cmp", "Off", "Blocks", "Matches", "link_player", "link_matches"]),
            'gca': (f'//*[@id="stats_gca_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "SCA", "SCA90", "PassLive", "PassDead", "TO", "Sh", "Fld", "Def", "GCA", "GCA90", "PassLiveGCA", "PassDeadGCA", "TOGCA", "ShGCA", "FldGCA", "DefGCA", "Matches", "link_player", "link_matches"]),
            'defensive': (f'//*[@id="stats_defense_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "Tkl", "TklW", "Def3rd", "Mid3rd", "Att3rd", "DribblersTackled", "DribblesChallenged", "DribblersTackled%", "ChallengesLost", "Blocks", "ShotsBlocked", "PassesBlocked", "Interceptions", "Tkl+Int", "Clearances", "Errors", "Matches", "link_player", "link_matches"]),
            'possession': (f'//*[@id="stats_possession_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "Touches", "DefPen", "Def3rd", "Mid3rd", "Att3rd", "AttPen", "TouchesLive", "TakeOns", "TakeOnSucc", "TakeOnSucc%", "Tkld", "Tkld%", "Carries", "TotDistCarries", "PrgDistCarries", "PrgCarries", "Final3rd", "CPA", "Miscontrols", "Dispossessed", "PassesReceived", "PrgPassesReceived", "Matches", "link_player", "link_matches"]),
            'playing_time': (f'//*[@id="stats_playing_time_{league_code}"]', ["Player", "Nation", "Pos", "Age", "MP", "Min", "Min/MP", "Min%", "90s", "Starts", "Min/Start", "Compl", "Subs", "Min/Sub", "SubMatches", "PPM", "onG", "onGA", "+/-", "+/-90", "OnOff","onxG", "onxGA", "xG+/-", "xG+/-90", "xGOnOff", "Matches", "link_player", "link_matches"]),
            'misc': (f'//*[@id="stats_misc_{league_code}"]', ["Player", "Nation", "Pos", "Age", "90s", "CrdY", "CrdR", "2CrdY", "Fls", "Fld", "Off", "Crs", "Int", "TklW", "PKwon", "PKcon", "OG", "Recov", "AerialsWon", "AerialsLost", "AerialsWon%", "Matches", "link_player", "link_matches"])
        }

        for name, (xpath, headers) in tables_info.items():
            df = extract_player_table(xpath, headers)
            if df is not None:
                df.to_csv(os.path.join(stats_dir, f"{name}.csv"), index=False)
                print(f"✅ {name} table saved")

# Close the browser
driver.quit()
