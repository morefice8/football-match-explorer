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

# Create a global list to collect all player links across leagues
all_players = []

# Main loop to collect player links from each team's standard.csv
for league_url, league_name, league_code in leagues:
    print(f"\n🔎 Looking for players in league: {league_name}")

    league_teams_dir = os.path.join("data", "fbref", league_name.replace(" ", "-").lower(), f"{league_name.replace(" ", "-").lower()}_teams")
    if not os.path.exists(league_teams_dir):
        print(f"❌ Directory not found for teams: {league_teams_dir}")
        continue

    for team_folder in os.listdir(league_teams_dir):
        team_path = os.path.join(league_teams_dir, team_folder, "team_stats", "standard.csv")
        if not os.path.exists(team_path):
            print(f"⚠️ No standard.csv for {team_folder} in {league_name}")
            continue

        df = pd.read_csv(team_path)

        if 'link_player' in df.columns:
            for _, row in df.iterrows():
                player_name = row['Player']
                player_url = row['link_player']
                if pd.notna(player_url):
                    full_url = f"https://fbref.com{player_url}"
                    all_players.append({"name": player_name, "url": full_url})

# Save the full player list for later processing
output_path = os.path.join("data", "fbref", "player_top5_europe", "all_players_links.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(all_players).drop_duplicates().to_csv(output_path, index=False)
print(f"\n✅ Player links saved to {output_path}")

# Close the browser
driver.quit()
