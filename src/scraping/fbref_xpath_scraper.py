import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# Configure the Chrome WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--log-level=3")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)

# List of leagues with their URLs, names, and competition codes
leagues = [
    ('https://fbref.com/en/comps/9/Premier-League-Stats', 'Premier League', '9'),
    ('https://fbref.com/en/comps/12/La-Liga-Stats', 'La Liga', '12'),
    ('https://fbref.com/en/comps/20/Bundesliga-Stats', 'Bundesliga', '20'),
    ('https://fbref.com/en/comps/11/Serie-A-Stats', 'Serie A', '11'),
    ('https://fbref.com/en/comps/13/Ligue-1-Stats', 'Ligue 1', '13')
]

# Generic function to extract an HTML table by XPath
def extract_table(xpath, headers):
    table = driver.find_element(By.XPATH, xpath)
    outerHTML = table.get_attribute("outerHTML")
    soup = BeautifulSoup(outerHTML, 'html.parser')

    # Extract table rows
    rows = []
    for row in soup.find_all('tr', attrs={"data-row": True}):  # Only rows with actual data
        cells = row.find_all(['th', 'td'])
        row_data = []
        link = None
        for cell in cells:
            # Extract link if present
            anchor = cell.find('a')
            if anchor and 'href' in anchor.attrs:
                link = anchor['href']
            row_data.append(cell.get_text(strip=True))
        row_data.append(link)  # Append link as last column
        rows.append(row_data)

    # Return dataframe with provided headers
    return pd.DataFrame(rows, columns=headers)

# Main scraping function
def scrape_all_leagues():
    for url, league_name, league_code in leagues:
        print(f"\n⚽ Scraping {league_name}...")

        # Define the structure of each table to scrape for the league
        tables_info = {
            'table': {
                'xpath': f'//*[@id="results2024-2025{league_code}1_overall"]',
                'headers': [
                    'RK', 'Squad', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP',
                    'xG', 'xGA', 'xGD', 'xGD/90', 'Attendance', 'Top Team Scorer', 'Goalkeeper', 'Notes', 'link'
                ]
            },
            'standard': {
                'xpath': '//*[@id="stats_squads_standard_for"]',
                'headers': [
                    'Squad', '#Pl', 'Age', 'Poss', 'MP', 'Starts', 'Min', '90s', 'Gls.', 'Ast', 'G+A',
                    'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP',
                    'Gls/90', 'Ast/90', 'G+A/90', 'G-PK/90', 'G+A-PK/90', 'xG/90', 'xAG/90', 'xG+xAG/90',
                    'npxG/90', 'npxG+xAG/90', 'link'
                ]
            },
            'goalkeeping': {
                'xpath': '//*[@id="stats_squads_keeper_for"]',
                'headers': [
                    'Squad', '#Pl', 'MP', 'Starts', 'Min', '90s', 'GA', 'GA90', 'SoTA', 'Saves',
                    'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PKatt', 'PKA', 'PKsv', 'PKm', 'PKSave%', 'link'
                ]
            },
            'goalkeeping_advanced': {
                'xpath': '//*[@id="stats_squads_keeper_adv_for"]',
                'headers': [
                    'Squad', '#Pl', '90s', 'GA', 'PKA', 'FK', 'CK', 'OG', 'PSxG', 'PSxG/SoT', 'PSxG-GA', 'PSxG-GA/90',
                    'Cmp(Launched)', 'Att(Launched)', 'Cmp%(Launched)', 'PassAtt(GK)', 'ThrAtt(GK)', 'Launch%(GK)', 'AvgLen(GK)', 'GoalKicks',
                    'Launch%(GoalKicks)', 'AvgLen(GoalKicks)', 'CrossesFaced', 'CrossesStopped', 'CrossesStopped%', '#OPA',
                    '#OPA/90', 'AvgDistOPA', 'link'
                ]
            },
            'shooting': {
                'xpath': '//*[@id="stats_squads_shooting_for"]',
                'headers': [
                    'Squad', '#PlUsed', '90s', 'Gls', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT',
                    'Dist', 'FK', 'PK', 'PKatt', 'xG', 'npxG', 'npxG/Sh', 'G-xG', 'np:G-xG', 'link'
                ]
            },
            'passing': {
                'xpath': '//*[@id="stats_squads_passing_for"]',
                'headers' : [
                    'Squad', '#PlUsed', '90s', 'Cmp', 'Att', 'Cmp%', 'TotDist', 'PrgDist',
                    'Cmp(Short)', 'Att(Short)', 'Cmp%(Short)', 'Cmp(Medium)', 'Att(Medium)', 'Cmp%(Medium)',
                    'Cmp(Long)', 'Att(Long)', 'Cmp%(Long)', 'Ast', 'xAG', 'xA', 'A-xAG', 'KP',
                    'PassesFinal3rd', 'PPA', 'CrsPA', 'PrgP', 'link'
                ]
            },
            'pass_type': {
                'xpath': '//*[@id="stats_squads_passing_types_for"]',
                'headers' : [
                    'Squad', '#PlUsed', '90s', 'Att', 'Live', 'Dead', 'FK', 'TB', 'Sw',
                    'Crs', 'TI', 'CK', 'CKIn', 'CKOut', 'CKStr', 'Cmp', 'Off', 'Blocks', 'link'
                ]
            },
            'gca': {
                'xpath': '//*[@id="stats_squads_gca_for"]',
                'headers': [
                    'Squad', '#PlUsed', '90s', 'SCA', 'SCA90', 'PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def',
                    'GCA', 'GCA90', 'PassLive_GCA', 'PassDead_GCA', 'TO_GCA', 'Sh_GCA', 'Fld_GCA', 'Def_GCA', 'link'
                ]
            },
            'defensive': {
                'xpath': '//*[@id="stats_squads_defense_for"]',
                'headers' : [
                    'Squad', '#PlUsed', '90s', 'Tkl', 'TklW', 'Def3rd', 'Mid3rd', 'Att3rd',
                    'DribblersTackled', 'DribblersChallenged', 'DribblersTackled%', 'ChallengesLost',
                    'Blocks', 'ShotsBlocked', 'PassesBlocked',
                    'Interceptions', 'Tkl+Int', 'Clearances', 'Errors', 'link'
                ]
            },
            'possession': {
                'xpath': '//*[@id="stats_squads_possession_for"]',
                'headers': [
                    'Squad', '#PlUsed', 'Poss', '90s', 'Touches', 'DefPen', 'Def3rd', 'Mid3rd', 'Att3rd',
                    'AttPen', 'ToucesLive', 'AttTakeOn', 'SuccTakeOn', 'SuccTakeOn%', 'TkldTakeOn', 'TkldTakeOn%', 'Carries',
                    'TotDistCarries', 'PrgDistCarries', 'PrgCarries', 'CarriesFinal3rd', 'CPA', 'Miscontrols', 'Dispossessed', 'PassesReceived', 'PrgPassesReceived', 'link'
                ]
            },
            'playing_time': {
                'xpath': '//*[@id="stats_squads_playing_time_for"]',
                'headers': [
                    'Squad', '#PlUsed', 'Age', 'MP', 'Min', 'Mn/MP', 'Min%', '90s', 'Starts',
                    'Mn/Start', 'Compl', 'Subs', 'Mn/Sub', 'unSub', 'PPM',
                    'onG', 'onGA', '+/-', '+/-90', 'onxG', 'onxGA', 'xG+/-', 'xG+/-90', 'link'
                ]
            },
            'miscellaneous': {
                'xpath': '//*[@id="stats_squads_misc_for"]',
                'headers': [
                    'Squad', '#PlUsed', '90s', 'CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld', 'Off', 'Crs',
                    'Int', 'TklW', 'PKwon', 'PKcon', 'OG', 'Recov', 'AerialsWon', 'AerialsLost', 'AerialsWon%',
                    'link'
                ]
            }
        }

        # Navigate to the league page
        driver.get(url)
        time.sleep(5)

        # Extract and save all specified tables
        dataframes = {}
        for name, info in tables_info.items():
            dataframes[name] = extract_table(info['xpath'], info['headers'])

        # Create a folder for the league
        output_dir = os.path.join("data", "fbref", league_name.replace(" ", "-").lower(), league_name.replace(" ", "-").lower() + "_stats")
        os.makedirs(output_dir, exist_ok=True)

        # Save each table as a CSV
        for name, df in dataframes.items():
            output_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved CSV for {name} in {output_path}")

    # Close the browser once done
    driver.quit()
