import json
import os # Recommended for handling file paths

def clean_json(file_path):
    """
    Reads a text file, attempts to find and extract the main JSON block,
    and loads it into a Python dictionary.

    Args:
        file_path (str): The path to the input text file containing JSON data.

    Returns:
        dict or None: The loaded JSON data as a dictionary, or None if an error occurs.
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Find the start and end of the main JSON object
        start_pos = file_content.find('{')
        end_pos = file_content.rfind('}')

        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            # Extract the potential JSON string
            json_string = file_content[start_pos:end_pos + 1]

            # Attempt to load the extracted string as JSON
            try:
                data = json.loads(json_string)
                print(f"Successfully cleaned and loaded JSON from {file_path}")
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
                # Optionally, you could try more robust cleaning here if needed
                return None
        else:
            print(f"Error: Could not find valid JSON start/end braces in {file_path}.")
            return None

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while cleaning JSON from {file_path}: {e}")
        return None