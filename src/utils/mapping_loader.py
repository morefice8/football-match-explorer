import pandas as pd
import json
import os

def load_opta_event_mapping(excel_path):
    """Loads the Opta EventCode -> EventName mapping from an Excel file."""
    try:
        events_df = pd.read_excel(excel_path)
        # Create a dictionary for faster lookups (Code -> Event)
        # Assumes columns are named 'Code' and 'Event'
        event_map = pd.Series(events_df.Event.values, index=events_df.Code).to_dict()
        print(f"Loaded {len(event_map)} event mappings from {excel_path}")
        return event_map
    except FileNotFoundError:
        print(f"Error: Event mapping file not found at {excel_path}")
        return {}
    except KeyError:
        print(f"Error: Expected columns 'Code' and 'Event' not found in {excel_path}")
        return {}
    except Exception as e:
        print(f"An error occurred loading event mappings: {e}")
        return {}

def load_opta_qualifier_mapping(json_path):
    """Loads the Opta QualifierID -> QualifierInfo mapping from a JSON file."""
    try:
        with open(json_path, "r", encoding='utf-8') as f:
            qualifiers_map = json.load(f)
        # Ensure keys are strings, as they often come from string-based IDs in data
        string_key_map = {str(k): v for k, v in qualifiers_map.items()}
        print(f"Loaded {len(string_key_map)} qualifier mappings from {json_path}")
        return string_key_map
    except FileNotFoundError:
        print(f"Error: Qualifier mapping file not found at {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return {}
    except Exception as e:
        print(f"An error occurred loading qualifier mappings: {e}")
        return {}