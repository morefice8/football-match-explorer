# src/metrics/league_metrics.py
import pandas as pd
import numpy as np

def add_advanced_metrics(df):
    """
    Aggiunge metriche calcolate che descrivono lo stile di gioco di una squadra.
    """
    df_new = df.copy()

    # --- METRICHE DI STILE ---
    if 'Total Passes' in df_new.columns and 'Successful Dribbles' in df_new.columns and 'Unsuccessful Dribbles' in df_new.columns:
        total_dribbles = df_new['Successful Dribbles'] + df_new['Unsuccessful Dribbles']
        total_touches_proxy = df_new['Total Passes'] + total_dribbles
        df_new['Passing Tempo'] = (df_new['Total Passes'] / total_touches_proxy * 100).replace([np.inf, -np.inf], 0).fillna(0)

    if 'Successful Short Passes' in df_new.columns and 'Successful Dribbles' in df_new.columns:
        df_new['Pass vs Carry Index'] = (df_new['Successful Short Passes'] / df_new['Successful Dribbles']).replace([np.inf, -np.inf], 0).fillna(0)

    if 'Successful Short Passes' in df_new.columns and 'Successful Long Passes' in df_new.columns:
        df_new['Short vs Long Ratio'] = (df_new['Successful Short Passes'] / df_new['Successful Long Passes']).replace([np.inf, -np.inf], 0).fillna(0)

    # --- METRICHE DIFENSIVE ---
    if 'Aerial Duels won' in df_new.columns and 'Aerial Duels lost' in df_new.columns:
        total_aerials = df_new['Aerial Duels won'] + df_new['Aerial Duels lost']
        df_new['Aerial Duels Won %'] = (df_new['Aerial Duels won'] / total_aerials * 100).fillna(0)

    if 'Tackles Won' in df_new.columns and 'Interceptions' in df_new.columns:
        df_new['Defensive Actions'] = df_new['Tackles Won'] + df_new['Interceptions']
        
        if 'Total Shots Conceded' in df_new.columns and 'Defensive Actions' in df_new.columns:
            # Calcola l'efficienza: quanti tiri concedi per ogni azione difensiva?
            df_new['Shots Conceded per DA'] = (df_new['Total Shots Conceded'] / df_new['Defensive Actions']).replace([np.inf, -np.inf], 0).fillna(0)
        
    return df_new
        
    return df_new