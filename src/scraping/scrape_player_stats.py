import os
import json
import re
import html
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import random
import shutil
import logging

# --- SETUP ---
options = webdriver.ChromeOptions()
options.add_argument("--log-level=3")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

player_csv = os.path.join("data", "fbref", "player_top5_europe", "all_players_links.csv")
df_players = pd.read_csv(player_csv).drop_duplicates()

output_root = os.path.join("data", "fbref", "player_top5_europe")
os.makedirs(output_root, exist_ok=True)


def handle_cookie_consent():
    try:
        accept_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Accept')]")))
        accept_button.click()
    except TimeoutException:
        pass

def extract_all_tables_on_page():
    # ... (questa funzione è già corretta e rimane invariata) ...
    all_dataframes = {}
    table_wrappers = driver.find_elements(By.CSS_SELECTOR, "div.table_wrapper")
    for wrapper in table_wrappers:
        try:
            title_element = wrapper.find_element(By.TAG_NAME, 'h2')
            raw_title = title_element.text
            clean_title = raw_title.strip().lower().replace(' ', '_').replace('/', '_')
            if not clean_title: continue
            table_element = wrapper.find_element(By.TAG_NAME, 'table')
            soup = BeautifulSoup(table_element.get_attribute('outerHTML'), 'html.parser')
            header_row = soup.select_one('thead tr:last-of-type')
            if not header_row: continue
            headers = [th.get_text(strip=True) for th in header_row.select('th')]
            link_column_present = False
            if 'Match Report' in headers:
                link_column_present = True
                headers.append('match_report_link')
            elif 'Matches' in headers:
                link_column_present = True
                headers.append('matches_link')
            data_rows = []
            body_rows = soup.select('tbody tr')
            for row in body_rows:
                if not row.find('th', scope='row'):
                    continue
                cells = row.find_all(['th', 'td'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if link_column_present:
                    match_link = None
                    link_cell = row.select_one("td[data-stat='match_report'], td[data-stat='matches']")
                    if link_cell and (link_tag := link_cell.find('a')) and link_tag.has_attr('href'):
                        match_link = 'https://fbref.com' + link_tag['href']
                    row_data.append(match_link)
                if len(row_data) == len(headers):
                    data_rows.append(row_data)
            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                all_dataframes[clean_title] = df
        except (NoSuchElementException, IndexError):
            continue
    return all_dataframes

def extract_player_profile():
    """
    Estrae i dati del profilo del giocatore, partendo dal selettore più 
    specifico e affidabile (il tooltip dello stipendio) e risalendo da lì.
    """
    try:
        meta_soup = BeautifulSoup(driver.find_element(By.ID, "meta").get_attribute("outerHTML"), 'html.parser')
        info_div = meta_soup.select_one("div:not(.media-item)")
        if not info_div: return None
        
        profile = {}

        # Dati anagrafici (già corretti)
        name_tag = info_div.select_one('h1 > span:first-of-type')
        profile['full_name'] = name_tag.text.strip() if name_tag else None

        pos_p = info_div.find(lambda tag: tag.name == 'p' and 'Position:' in tag.get_text())
        if pos_p:
            text = pos_p.get_text(strip=True).replace("Position:", "").replace("Footed:", "")
            parts = [p.strip() for p in text.split('▪')]
            profile['position'] = parts[0] if len(parts) > 0 else None
            profile['footed'] = parts[1] if len(parts) > 1 else None

        hw_p = info_div.find(lambda tag: tag.name == 'p' and 'cm' in tag.get_text() and 'kg' in tag.get_text())
        if hw_p:
            text_parts = hw_p.get_text(strip=True).split('(')[0].split(',')
            if len(text_parts) >= 2:
                profile['height_cm'] = text_parts[0].strip()
                profile['weight_kg'] = text_parts[1].strip()

        birth_span = info_div.select_one('#necro-birth')
        if birth_span:
            profile['birth_date'] = birth_span.get('data-birth')
            birth_place_span = birth_span.find_next("span", string=lambda text: text and " in " in text)
            if birth_place_span:
                profile['birth_place'] = birth_place_span.text.strip().replace("in ", "")

        def get_linked_item(label):
            strong_tag = info_div.find('strong', string=lambda s: s and label in s)
            if strong_tag and (link_tag := strong_tag.find_next_sibling('a')):
                return {'name': link_tag.text.strip(), 'link': 'https://fbref.com' + link_tag['href']}
            return {'name': None, 'link': None}

        profile['national_team'] = get_linked_item('National Team:')
        profile['club'] = get_linked_item('Club:')
        
        # --- LOGICA CORRETTA E ROBUSTA PER STIPENDIO E CONTRATTO ---
        # 1. Cerca direttamente il tooltip nell'intero blocco delle informazioni
        tooltip_span = info_div.select_one('span[data-tip*="Wages in Euros"]')
        
        if tooltip_span:
            try:
                # 2. Estrai lo stipendio in Euro dal tooltip.
                tooltip_html = html.unescape(tooltip_span['data-tip'])
                wage_match = re.search(r"Wages in Euros:.*?Weekly:\s*(€\s*[\d,]+)", tooltip_html, re.DOTALL)
                if wage_match:
                    profile['wages_weekly_euro'] = wage_match.group(1).strip()
            except Exception as e:
                print(f"⚠️ Failed to parse wages tooltip: {e}")
            
            # 3. Risali al paragrafo genitore per trovare la scadenza del contratto.
            parent_p = tooltip_span.find_parent('p')
            if parent_p:
                p_text = parent_p.get_text()
                expires_match = re.search(r"Expires\s+([A-Za-z]+\s+\d{4})", p_text)
                if expires_match:
                    profile['contract_expires'] = expires_match.group(1).strip()
        
        insta_strong = info_div.find('strong', string=lambda s: s and 'Instagram:' in s)
        if insta_strong and (link_tag := insta_strong.find_next('a')):
            profile['instagram'] = link_tag['href']

        return profile
    except Exception as e:
        print(f"⚠️ Failed to extract profile: {e}")
        return None

# --- CICLO PRINCIPALE ---
total_players = len(df_players)
for index, row in df_players.iterrows():
    name = row['name'].replace("/", "-").replace(" ", "_")
    base_url = row['url']
    
    print(f"\n🎯 [{index+1}/{total_players}] Processing: {name}")

    try:
        player_dir = os.path.join(output_root, name)
        os.makedirs(player_dir, exist_ok=True)

        # --- MODIFICA CHIAVE: ESTRAZIONE ROBUSTA DELL'ID ---
        # Questo metodo non si basa più sulla posizione fissa nell'URL.
        if "/players/" not in base_url:
            print(f"⚠️ Invalid URL for {name}: {base_url}. Skipping.")
            continue
        
        # Trova la parte dell'URL dopo "/players/"
        path_after_players = base_url.split('/players/')[1]
        # L'ID è la prima parte di ciò che resta, prima del successivo "/"
        player_id = path_after_players.split('/')[0]

        if not player_id:
            print(f"⚠️ Could not extract Player ID for {name}. Skipping.")
            continue
        
        # Costruisci un URL "All Comps" che funziona sempre, anche solo con l'ID.
        # all_comps_url = f"https://fbref.com/en/players/{player_id}/all_comps/"
        
        # print(f"🔗 Navigating to: {all_comps_url}")
        # driver.get(all_comps_url)
        # handle_cookie_consent()
        # --------------------------------------------------------

        # # Estrai l'immagine del profilo se disponibile
        # print("🖼️ Checking for player photo...")
        # img_path = os.path.join(player_dir, f"{name}.png")
        # if not os.path.exists(img_path):
        #     try:
        #         img_tag = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#meta .media-item img")))
        #         img_tag.screenshot(img_path)
        #         print(f"🖼️  New photo saved for {name}.")
        #     except Exception:
        #         print(f"⚠️  Could not save photo for {name}.")
        # else:
        #     print(f"🖼️  Photo for {name} already exists. Skipping download.")
        
        # # Estrai le tabelle
        # print("📊 Extracting player tables...")
        # player_tables = extract_all_tables_on_page()
        # if player_tables:
        #     print(f"   - Found {len(player_tables)} tables. Overwriting CSVs...")
        #     for table_name, df_table in player_tables.items():
        #         table_path = os.path.join(player_dir, f"{table_name}.csv")
        #         df_table.to_csv(table_path, index=False)
        #     print(f"📊 Tables for {name} successfully updated.")
        # else:
        #     print("⚠️ No tables were extracted for this player.")

        # Estrai i dati del profilo
        # print("📝 Extracting player profile data...")
        # profile_data = extract_player_profile()
        # if profile_data:
        #     json_path = os.path.join(player_dir, "profile.json")
        #     with open(json_path, 'w', encoding='utf-8') as f:
        #         json.dump(profile_data, f, ensure_ascii=False, indent=2)
        #     print("🧾 profile.json saved")
        # else:
        #     print("⚠️ profile.json not created")

        # Estrai le statistiche del giocatore
        season = "2024-2025"
        player_slug = name.replace('_', '-')
        matchlogs_url = f"https://fbref.com/en/players/{player_id}/matchlogs/{season}/{player_slug}-Match-Logs"
        print(f"Navigating to Match Logs page: {matchlogs_url}")
        driver.get(matchlogs_url)
        handle_cookie_consent()

        print("Extracting player match logs...")
        all_tables = extract_all_tables_on_page()
        matchlog_table = None
        for table_name, df_table in all_tables.items():
            if 'match_logs' in table_name:
                matchlog_table = df_table
                break

        if matchlog_table is not None and not matchlog_table.empty:
            matchlog_path = os.path.join(player_dir, "player_matchlogs.csv")
            matchlog_table.to_csv(matchlog_path, index=False)
            print(f"🧾 player_matchlogs.csv saved.")

    except Exception as e:
        print(f"🚨🚨🚨 CRITICAL ERROR processing {name}: {e}")
        print("     Moving to the next player.")
        continue

    # Pausa casuale
    sleep_time = random.uniform(1, 3)
    time.sleep(sleep_time)

driver.quit()
print("\n\n✅✅✅ UPDATE PROCESS COMPLETE! ✅✅✅")