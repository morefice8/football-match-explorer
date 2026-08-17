import os
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


SEASON = "2025-2026"
FBREF_BASE_URL = "https://fbref.com"

# Start with the Premier League only. Uncomment the other leagues after
# confirming that the complete pipeline works as expected.
LEAGUES = [
    (
        f"{FBREF_BASE_URL}/en/comps/9/{SEASON}/{SEASON}-Premier-League-Stats",
        "Premier League",
        "9",
    ),
    # (
    #     f"{FBREF_BASE_URL}/en/comps/12/{SEASON}/{SEASON}-La-Liga-Stats",
    #     "La Liga",
    #     "12",
    # ),
    # (
    #     f"{FBREF_BASE_URL}/en/comps/20/{SEASON}/{SEASON}-Bundesliga-Stats",
    #     "Bundesliga",
    #     "20",
    # ),
    # (
    #     f"{FBREF_BASE_URL}/en/comps/11/{SEASON}/{SEASON}-Serie-A-Stats",
    #     "Serie A",
    #     "11",
    # ),
    # (
    #     f"{FBREF_BASE_URL}/en/comps/13/{SEASON}/{SEASON}-Ligue-1-Stats",
    #     "Ligue 1",
    #     "13",
    # ),
]


def create_driver():
    """Create Chrome with a persistent profile dedicated to this project."""
    options = webdriver.ChromeOptions()
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("The LOCALAPPDATA environment variable is not available.")

    profile_dir = os.path.join(
        local_app_data,
        "FootballMatchExplorer",
        "ChromeProfile",
    )
    os.makedirs(profile_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={profile_dir}")

    return webdriver.Chrome(options=options)


def extract_table(driver, xpath, headers):
    """Extract one FBref HTML table and return it as a DataFrame."""
    table = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    soup = BeautifulSoup(table.get_attribute("outerHTML"), "html.parser")

    rows = []
    for row in soup.find_all("tr", attrs={"data-row": True}):
        cells = row.find_all(["th", "td"])
        row_data = []
        team_link = None

        for cell in cells:
            # The first link in these squad tables is normally the team link.
            if team_link is None:
                anchor = cell.find("a", href=True)
                if anchor:
                    team_link = anchor["href"]

            row_data.append(cell.get_text(strip=True))

        row_data.append(team_link)

        if len(row_data) != len(headers):
            raise ValueError(
                "FBref table structure has changed: "
                f"found {len(row_data)} columns, expected {len(headers)}."
            )

        rows.append(row_data)

    return pd.DataFrame(rows, columns=headers)


def get_tables_info(league_code):
    """Return table selectors and output column names for one league."""
    return {
        "table": {
            "xpath": f'//*[@id="results{SEASON}{league_code}1_overall"]',
            "headers": [
                "RK",
                "Squad",
                "MP",
                "W",
                "D",
                "L",
                "GF",
                "GA",
                "GD",
                "Pts",
                "Pts/MP",
                "xG",
                "xGA",
                "xGD",
                "xGD/90",
                "Attendance",
                "Top Team Scorer",
                "Goalkeeper",
                "Notes",
                "link",
            ],
        },
        "standard": {
            "xpath": '//*[@id="stats_squads_standard_for"]',
            "headers": [
                "Squad",
                "#Pl",
                "Age",
                "Poss",
                "MP",
                "Starts",
                "Min",
                "90s",
                "Gls.",
                "Ast",
                "G+A",
                "G-PK",
                "PK",
                "PKatt",
                "CrdY",
                "CrdR",
                "xG",
                "npxG",
                "xAG",
                "npxG+xAG",
                "PrgC",
                "PrgP",
                "Gls/90",
                "Ast/90",
                "G+A/90",
                "G-PK/90",
                "G+A-PK/90",
                "xG/90",
                "xAG/90",
                "xG+xAG/90",
                "npxG/90",
                "npxG+xAG/90",
                "link",
            ],
        },
        "goalkeeping": {
            "xpath": '//*[@id="stats_squads_keeper_for"]',
            "headers": [
                "Squad",
                "#Pl",
                "MP",
                "Starts",
                "Min",
                "90s",
                "GA",
                "GA90",
                "SoTA",
                "Saves",
                "Save%",
                "W",
                "D",
                "L",
                "CS",
                "CS%",
                "PKatt",
                "PKA",
                "PKsv",
                "PKm",
                "PKSave%",
                "link",
            ],
        },
        "goalkeeping_advanced": {
            "xpath": '//*[@id="stats_squads_keeper_adv_for"]',
            "headers": [
                "Squad",
                "#Pl",
                "90s",
                "GA",
                "PKA",
                "FK",
                "CK",
                "OG",
                "PSxG",
                "PSxG/SoT",
                "PSxG-GA",
                "PSxG-GA/90",
                "Cmp(Launched)",
                "Att(Launched)",
                "Cmp%(Launched)",
                "PassAtt(GK)",
                "ThrAtt(GK)",
                "Launch%(GK)",
                "AvgLen(GK)",
                "GoalKicks",
                "Launch%(GoalKicks)",
                "AvgLen(GoalKicks)",
                "CrossesFaced",
                "CrossesStopped",
                "CrossesStopped%",
                "#OPA",
                "#OPA/90",
                "AvgDistOPA",
                "link",
            ],
        },
        "shooting": {
            "xpath": '//*[@id="stats_squads_shooting_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "90s",
                "Gls",
                "Sh",
                "SoT",
                "SoT%",
                "Sh/90",
                "SoT/90",
                "G/Sh",
                "G/SoT",
                "Dist",
                "FK",
                "PK",
                "PKatt",
                "xG",
                "npxG",
                "npxG/Sh",
                "G-xG",
                "np:G-xG",
                "link",
            ],
        },
        "passing": {
            "xpath": '//*[@id="stats_squads_passing_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "90s",
                "Cmp",
                "Att",
                "Cmp%",
                "TotDist",
                "PrgDist",
                "Cmp(Short)",
                "Att(Short)",
                "Cmp%(Short)",
                "Cmp(Medium)",
                "Att(Medium)",
                "Cmp%(Medium)",
                "Cmp(Long)",
                "Att(Long)",
                "Cmp%(Long)",
                "Ast",
                "xAG",
                "xA",
                "A-xAG",
                "KP",
                "PassesFinal3rd",
                "PPA",
                "CrsPA",
                "PrgP",
                "link",
            ],
        },
        "pass_type": {
            "xpath": '//*[@id="stats_squads_passing_types_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "90s",
                "Att",
                "Live",
                "Dead",
                "FK",
                "TB",
                "Sw",
                "Crs",
                "TI",
                "CK",
                "CKIn",
                "CKOut",
                "CKStr",
                "Cmp",
                "Off",
                "Blocks",
                "link",
            ],
        },
        "gca": {
            "xpath": '//*[@id="stats_squads_gca_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "90s",
                "SCA",
                "SCA90",
                "PassLive",
                "PassDead",
                "TO",
                "Sh",
                "Fld",
                "Def",
                "GCA",
                "GCA90",
                "PassLive_GCA",
                "PassDead_GCA",
                "TO_GCA",
                "Sh_GCA",
                "Fld_GCA",
                "Def_GCA",
                "link",
            ],
        },
        "defensive": {
            "xpath": '//*[@id="stats_squads_defense_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "90s",
                "Tkl",
                "TklW",
                "Def3rd",
                "Mid3rd",
                "Att3rd",
                "DribblersTackled",
                "DribblersChallenged",
                "DribblersTackled%",
                "ChallengesLost",
                "Blocks",
                "ShotsBlocked",
                "PassesBlocked",
                "Interceptions",
                "Tkl+Int",
                "Clearances",
                "Errors",
                "link",
            ],
        },
        "possession": {
            "xpath": '//*[@id="stats_squads_possession_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "Poss",
                "90s",
                "Touches",
                "DefPen",
                "Def3rd",
                "Mid3rd",
                "Att3rd",
                "AttPen",
                "ToucesLive",
                "AttTakeOn",
                "SuccTakeOn",
                "SuccTakeOn%",
                "TkldTakeOn",
                "TkldTakeOn%",
                "Carries",
                "TotDistCarries",
                "PrgDistCarries",
                "PrgCarries",
                "CarriesFinal3rd",
                "CPA",
                "Miscontrols",
                "Dispossessed",
                "PassesReceived",
                "PrgPassesReceived",
                "link",
            ],
        },
        "playing_time": {
            "xpath": '//*[@id="stats_squads_playing_time_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "Age",
                "MP",
                "Min",
                "Mn/MP",
                "Min%",
                "90s",
                "Starts",
                "Mn/Start",
                "Compl",
                "Subs",
                "Mn/Sub",
                "unSub",
                "PPM",
                "onG",
                "onGA",
                "+/-",
                "+/-90",
                "onxG",
                "onxGA",
                "xG+/-",
                "xG+/-90",
                "link",
            ],
        },
        "miscellaneous": {
            "xpath": '//*[@id="stats_squads_misc_for"]',
            "headers": [
                "Squad",
                "#PlUsed",
                "90s",
                "CrdY",
                "CrdR",
                "2CrdY",
                "Fls",
                "Fld",
                "Off",
                "Crs",
                "Int",
                "TklW",
                "PKwon",
                "PKcon",
                "OG",
                "Recov",
                "AerialsWon",
                "AerialsLost",
                "AerialsWon%",
                "link",
            ],
        },
    }


def scrape_all_leagues(driver):
    verification_completed = False

    for url, league_name, league_code in LEAGUES:
        print(f"\nScraping {league_name} ({SEASON})...")
        tables_info = get_tables_info(league_code)

        driver.get(url)

        if not verification_completed:
            print("\nComplete any human-verification step in the Chrome window.")
            input(
                "When the FBref league page and its tables are visible, "
                "return here and press ENTER..."
            )
            verification_completed = True

        output_slug = league_name.replace(" ", "-").lower()
        output_dir = os.path.join(
            "data",
            "fbref",
            output_slug,
            f"{output_slug}_stats",
        )
        os.makedirs(output_dir, exist_ok=True)

        saved_tables = 0
        for table_name, info in tables_info.items():
            try:
                dataframe = extract_table(driver, info["xpath"], info["headers"])
            except (TimeoutException, NoSuchElementException) as error:
                print(f"Table '{table_name}' not found: {error.__class__.__name__}")
                continue
            except ValueError as error:
                print(f"Table '{table_name}' not saved: {error}")
                continue

            output_path = os.path.join(output_dir, f"{table_name}.csv")
            dataframe.to_csv(output_path, index=False)
            saved_tables += 1
            print(
                f"Saved {table_name}: {len(dataframe)} rows -> {output_path}"
            )

        print(
            f"Completed {league_name}: "
            f"{saved_tables}/{len(tables_info)} tables saved."
        )

        # Avoid immediately requesting the next league page.
        time.sleep(5)


def main():
    driver = create_driver()
    try:
        scrape_all_leagues(driver)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()