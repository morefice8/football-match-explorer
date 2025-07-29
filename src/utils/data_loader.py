import os
import pandas as pd

def load_player_stats(base_path="data/fbref/player_top5_europe"):
    all_players = []
    for player in os.listdir(base_path):
        player_dir = os.path.join(base_path, player, "2024-2025")
        if not os.path.isdir(player_dir):
            continue
        try:
            stats = {}
            for file in os.listdir(player_dir):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(player_dir, file))
                    df = df.dropna(axis=1, how="all")  # remove empty columns
                    stats.update(df.iloc[0].to_dict())  # keep only first row

            stats["Player"] = player.replace("_", " ")
            stats["Squad"] = stats.get("Squad", "Unknown")
            all_players.append(stats)
        except Exception as e:
            print(f"Failed loading {player}: {e}")
            continue

    return pd.DataFrame(all_players)
