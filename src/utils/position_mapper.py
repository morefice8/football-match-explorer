# src/utils/position_mapper.py
import pandas as pd # Optional, might not be needed

# --- Manual Mapping based on Opta F24 Formation Diagrams ---
# Key: formation_id (int)
# Value: dictionary where Key=position_num (int), Value=role_name (str)

FORMATION_ROLE_MAP = {
    # Formation 2: 442 (Standard)
    2: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        7: 'RM', 4: 'RCM', 8: 'LCM', 11: 'LM',
        10: 'RCF', 9: 'LCF'},
    # Formation 3: 41212 (Diamond)
    3: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        4: 'DM', 7: 'RCM', 11: 'LCM', 8: 'ACM',
        10: 'RCF', 9: 'LCF'},
    # Formation 4: 433 (Standard)
    4: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        7: 'RCM', 4: 'CM', 8: 'LCM', # Or 4=DM? Check typical usage
        10: 'RW', 9: 'ST', 11: 'LW'},
    # Formation 5: 451 / 4141 (Need to distinguish or merge?) - Assuming 4141 structure here
    # Check Formation 7 for alternative interpretation
    5: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        4: 'DM', # Assuming 4 is DM in 4141 like F7
        7: 'RM', 8: 'RCM', 10: 'LCM', 11: 'LM',
        9: 'ST'},
    # Formation 6: 4411
    6: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        7: 'RM', 4: 'RCM', 8: 'LCM', 11: 'LM',
        10: 'CAM/SS', 9: 'ST'},
    # Formation 7: 4141
    7: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        4: 'DM', 7: 'RM', 8: 'RCM', 10: 'LCM', 11: 'LM',
        9: 'ST'},
    # Formation 8: 4231
    8: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
        8: 'RDM', 4: 'LDM', # Double pivot
        7: 'RAM', 10: 'CAM', 11: 'LAM',
        9: 'ST'},
    # Formation 9: 4321 (Christmas Tree)
    9: {1: 'GK', 2: 'RB', 5: 'RCB', 6: 'LCB', 3: 'LB',
         8: 'RCM', 4: 'CM', 7: 'LCM', # Midfield 3
         10: 'RAM/SS', 11: 'LAM/SS', # Behind striker
         9: 'ST'},
    # Formation 10: 532 / 352 (Need to check wingback numbering) - Assuming 352 based on F12 diagram
    10:{1: 'GK', 6: 'RCB', 5: 'CB', 4: 'LCB', # Back 3
         2: 'RWB', 7: 'RCM', 8: 'LCM', 3: 'LWB', # Mid 5
         11: 'SUB/?', # F10 diagram doesn't show 11 in midfield? Recheck Opta doc/data for typical 532
         10: 'RCF', 9: 'LCF'}, # F10 diagram looks more like 352 than 532 layout-wise
    # Formation 11: 541
    11:{1: 'GK', 2: 'RWB', 6: 'RCB', 5: 'CB', 4: 'LCB', 3: 'LWB', # Back 5
         7: 'RM', 8: 'RCM', 10: 'LCM', 11: 'LM', # Mid 4
         9: 'ST'},
    # Formation 12: 352
    12:{1: 'GK', 6: 'RCB', 5: 'CB', 4: 'LCB', # Back 3
         2: 'RWB', 7: 'RCM', 8: 'LCM', 3: 'LWB', # Midfield (assuming 11 is AM/Sub)
         11: 'CAM/SS', # Often the player behind strikers
         10: 'RCF', 9: 'LCF'},
    # Formation 13: 343
    13:{1: 'GK', 6: 'RCB', 5: 'CB', 4: 'LCB', # Back 3
         2: 'RWB/RM', 7: 'RCM', 8: 'LCM', 3: 'LWB/LM', # Mid 4
         10: 'RW', 9: 'ST', 11: 'LW'}, # Front 3
    # Formation 14: 31312 - Very specific, map carefully
    14:{1: 'GK', 6:'RCB', 5:'CB', 7:'LCB', 4:'DM', # Def block + DM
         2:'RM', 8:'CM', 3:'LM', # Mid 3
         10:'CAM', # Att Mid
         9:'RCF', 11:'LCF'}, # Strikers
     # Formation 15: 4222
    15:{1:'GK', 2:'RB', 5:'RCB', 6:'LCB', 3:'LB', # Back 4
         4:'RDM', 7:'LDM', # DMs
         8:'RAM', 11:'LAM', # AMs
         10:'RCF', 9:'LCF'}, # Strikers
     # Formation 16: 3511
    16:{1:'GK', 6:'RCB', 5:'CB', 4:'LCB', # Back 3
         2:'RWB', 7:'RCM', 8:'LCM', 3:'LWB', # Midfield (11 is CAM)
         11:'CAM', 10:'SS/F9', # Player behind main striker
         9:'ST'}, # Striker
     # Formation 17: 3421
    17:{1:'GK', 6:'RCB', 5:'CB', 4:'LCB', # Back 3
         2:'RWB/RM', 7:'RCM', 8:'LCM', 3:'LWB/LM', # Mid 4
         10:'RAM/IF', 9:'LAM/IF', # Attacking mids / Inside forwards
         11:'ST'}, # Striker
     # Formation 18: 3412
    18:{1:'GK', 6:'RCB', 5:'CB', 4:'LCB', # Back 3
         2:'RWB/RM', 7:'RCM', 8:'LCM', 3:'LWB/LM', # Mid 4
         9:'CAM', # Central AM
         10:'RCF', 11:'LCF'}, # Strikers

    # Add other formations as needed
}

def get_role_from_formation(formation_id, position_num):
    """
    Maps an Opta formation ID and positional number (1-11) to a role name.

    Args:
        formation_id (int or None): Opta Formation ID (e.g., 8 for 4231).
        position_num (int or None): Opta positional number (1-11 for starters).

    Returns:
        str: A descriptive role name (e.g., 'RCB', 'ST', 'Unknown') or None if inputs invalid.
    """
    if pd.isna(formation_id) or pd.isna(position_num) or position_num == 0: # Position 0 often means sub/not on pitch
        return 'Sub/Unknown'

    formation_map = FORMATION_ROLE_MAP.get(int(formation_id))
    if formation_map:
        role = formation_map.get(int(position_num))
        return role if role else 'UnknownPosNum' # Role found for formation, but number invalid?
    else:
        return 'UnknownFormation' # Formation ID not in our map