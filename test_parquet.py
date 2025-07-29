import pandas as pd

# Carica il file
df = pd.read_parquet("data/processed/player_stats_2024-2025.parquet")

# Trova il giocatore
kolo_muani_stats = df[df['Player'] == 'Randal Kolo Muani']

# Stampa la riga per vedere i valori
print(kolo_muani_stats.to_dict('records')) 