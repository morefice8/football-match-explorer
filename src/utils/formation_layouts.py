# src/utils/formation_layouts.py
import numpy as np
import pandas as pd  # Keep for pd.isna in get_formation_layout_coords

# Approximate X, Y coordinates for Opta pitch (0-100)
# Key: Opta Formation ID (int)
# Value: Dictionary where Key=Opta Positional Number (1-11), Value=(X, Y) tuple

# YOU MUST FINE-TUNE THESE COORDINATES FOR VISUAL ACCURACY!
FORMATION_COORDINATES = {
    # Formation 2: 4-4-2
    2: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        7: (50, 15),  # RM
        4: (50, 40),  # RCM
        8: (50, 60),  # LCM
        11: (50, 85), # LM
        10: (75, 40), # RCF
        9: (75, 60)   # LCF
    },
    # Formation 3: 4-1-2-1-2 (Diamond)
    3: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        4: (40, 50),  # DM
        7: (55, 25),  # RCM (wide CM)
        11: (55, 75),  # LCM (wide CM)
        8: (65, 50),  # CAM
        10: (80, 40),  # RCF
        9: (80, 60)   # LCF
    },
    # Formation 4: 4-3-3
    4: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        7: (50, 30),  # RCM/RDM
        4: (50, 50),  # CM (central)
        8: (50, 70),  # LCM/LDM
        10: (75, 20), # RW
        9: (80, 50),  # ST
        11: (75, 80)  # LW
    },
    # Formation 5: 4-5-1 (Often similar to 4-1-4-1 or flat 4-5-1)
    # Using a 4-1-4-1 structure based on commonality & your F7
    5: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        4: (40, 50),  # DM
        7: (55, 15),  # RM
        8: (55, 40),  # RCM
        10: (55, 60), # LCM (Note: F5 diagram has 10 and 11 swapped vs F7)
        11: (55, 85), # LM
        9: (75, 50)   # ST
    },
    # Formation 6: 4-4-1-1
    6: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        7: (50, 15),  # RM
        4: (50, 40),  # RCM
        8: (50, 60),  # LCM
        11: (50, 85), # LM
        10: (65, 50), # CAM/SS (Number 10 player)
        9: (80, 50)   # ST (Number 9 player)
    },
    # Formation 7: 4-1-4-1
    7: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        4: (40, 50),  # DM
        7: (55, 15),  # RM
        8: (55, 40),  # RCM
        10: (55, 60), # LCM
        11: (55, 85), # LM
        9: (75, 50)   # ST
    },
    # Formation 8: 4-2-3-1
    8: {
        1: (5, 50),   # GK
        2: (30, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (30, 85),  # LB
        8: (45, 40),  # RDM
        4: (45, 60),  # LDM
        7: (65, 20),  # RAM
        10: (65, 50), # CAM
        11: (65, 80), # LAM
        9: (80, 50)   # ST
    },
    # Formation 9: 4-3-2-1 (Christmas Tree)
    9: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        8: (45, 30),  # RCM
        4: (45, 50),  # CM
        7: (45, 70),  # LCM
        10: (65, 40), # RF/RAM (inner forward)
        11: (65, 60), # LF/LAM (inner forward)
        9: (80, 50)   # ST
    },
    # Formation 10: 5-3-2 (often looks like 3-5-2 with WBs)
    10: { # Assuming 3 CBs, 2 Wingbacks, 3 CMs, 2 STs
        1: (5, 50),   # GK
        2: (45, 10),  # RWB
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        3: (45, 90),  # LWB
        7: (50, 35),  # RCM
        8: (50, 50),  # CM
        11: (50, 65), # LCM (Diagram shows 11 in mid, unlike F12)
        10: (75, 40), # RCF
        9: (75, 60)   # LCF
    },
    # Formation 11: 5-4-1
    11: {
        1: (5, 50),   # GK
        2: (35, 10),  # RWB
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        3: (35, 90),  # LWB
        7: (50, 25),  # RM
        8: (50, 45),  # RCM
        10: (50, 55), # LCM (Diagram shows 10 and 8 central)
        11: (50, 75), # LM
        9: (75, 50)   # ST
    },
    # Formation 12: 3-5-2
    12: {
        1: (5, 50),   # GK
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        2: (50, 10),  # RWB
        7: (40, 35),  # RCM
        8: (40, 65),  # LCM
        3: (50, 90),  # LWB
        11: (60, 50), # CAM (Number 11 in diagram)
        10: (75, 40), # RCF
        9: (75, 60)   # LCF
    },
    # Formation 13: 3-4-3
    13: {
        1: (5, 50),   # GK
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        2: (45, 15),  # RWB/RM
        7: (45, 40),  # RCM
        8: (45, 60),  # LCM
        3: (45, 85),  # LWB/LM
        10: (70, 20), # RW
        9: (75, 50),  # ST
        11: (70, 80)  # LW
    },
    # Formation 14: 3-1-3-1-2 (This is very specific!)
    14: {
        1: (5, 50),   # GK
        6: (20, 25),  # RCB
        5: (20, 50),  # CB
        7: (20, 75),  # LCB
        4: (35, 50),  # DM
        2: (50, 15),  # RM
        8: (50, 50),  # CM (central of the 3)
        3: (50, 85),  # LM
        10: (65, 50), # CAM
        9: (80, 35),  # RCF
        11: (80, 65)  # LCF
    },
    # Formation 15: 4-2-2-2 (Box Midfield)
    15: {
        1: (5, 50),   # GK
        2: (25, 15),  # RB
        5: (25, 35),  # RCB
        6: (25, 65),  # LCB
        3: (25, 85),  # LB
        4: (45, 40),  # RDM
        7: (45, 60),  # LDM
        8: (65, 25),  # RAM (wide attacking mid)
        11: (65, 75), # LAM (wide attacking mid)
        10: (80, 40), # RCF
        9: (80, 60)   # LCF
    },
    # Formation 16: 3-5-1-1
    16: {
        1: (5, 50),   # GK
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        2: (45, 10),  # RWB
        7: (50, 35),  # RCM
        8: (50, 65),  # LCM
        3: (45, 90),  # LWB
        11: (60, 50), # CAM (Number 11 in diagram)
        10: (70, 50), # SS (Number 10 slightly ahead of CAM)
        9: (85, 50)   # ST
    },
    # Formation 17: 3-4-2-1
    17: {
        1: (5, 50),   # GK
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        2: (45, 15),  # RWB/RM
        7: (45, 40),  # RCM
        8: (45, 60),  # LCM
        3: (45, 85),  # LWB/LM
        10: (65, 40), # RF/RAM (inner forward)
        9: (65, 60),  # LF/LAM (inner forward)
        11: (80, 50)  # ST
    },
    # Formation 18: 3-4-1-2
    18: {
        1: (5, 50),   # GK
        6: (20, 30),  # RCB
        5: (20, 50),  # CB
        4: (20, 70),  # LCB
        2: (45, 15),  # RWB/RM
        7: (45, 40),  # RCM
        8: (45, 60),  # LCM
        3: (45, 85),  # LWB/LM
        9: (65, 50),  # CAM
        10: (80, 40), # RCF
        11: (80, 60)  # LCF
    },
}

# --- *** Map Opta Formation ID to Name *** ---
OPTA_FORMATION_ID_TO_NAME = {
    1: "Unknown/Other", # Usually Opta ID 1 is not a standard formation
    2: "4-4-2",
    3: "4-1-2-1-2", # Diamond
    4: "4-3-3",
    5: "4-5-1", # Can also be 4-1-4-1 depending on roles
    6: "4-4-1-1",
    7: "4-1-4-1", # Often used interchangeably with 4-5-1
    8: "4-2-3-1",
    9: "4-3-2-1", # Christmas Tree
    10: "5-3-2", # Or 3-5-2 with Wingbacks
    11: "5-4-1",
    12: "3-5-2",
    13: "3-4-3",
    14: "3-1-3-1-2", # Very specific
    15: "4-2-2-2", # Box Midfield
    16: "3-5-1-1",
    17: "3-4-2-1",
    18: "3-4-1-2",
    # Add any other IDs you encounter or have specific names for
}

def get_formation_layout_coords(formation_id, position_num):
    """
    Returns the (X, Y) plotting coordinates for a given Opta formation ID
    and positional number (1-11).
    """
    if pd.isna(formation_id) or pd.isna(position_num) or not isinstance(position_num, (int, np.integer)) or not (1 <= position_num <= 11):
        return (None, None)

    # Ensure formation_id is int for dictionary lookup
    try:
        fid = int(formation_id)
    except (ValueError, TypeError):
        # print(f"Warning: Invalid formation_id type: {formation_id}")
        return (None, None)

    formation_layout = FORMATION_COORDINATES.get(fid)
    if formation_layout:
        # Ensure position_num is int for dictionary lookup
        try:
            pid = int(position_num)
            coords = formation_layout.get(pid)
            return coords if coords else (None, None)
        except (ValueError, TypeError):
            # print(f"Warning: Invalid position_num type: {position_num}")
            return (None, None)
    else:
        # print(f"Warning: Layout coordinates not defined for formation ID {fid}")
        return (None, None)
    
# --- *** Helper function to get formation name *** ---
def get_formation_name(formation_id):
    """
    Returns the textual representation of a formation ID.
    """
    if pd.isna(formation_id):
        return "N/A"
    try:
        fid_int = int(formation_id)
        return OPTA_FORMATION_ID_TO_NAME.get(fid_int, f"ID {fid_int}") # Fallback to ID if not in map
    except (ValueError, TypeError):
        return f"ID {formation_id}" # If not convertible to int