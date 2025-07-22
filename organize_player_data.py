# --- START OF FILE organize_player_data.py ---
import os
import shutil

def organize_player_files():
    """
    Scans the player data directory and moves all seasonal CSV stat files
    into a '2024-2025' subfolder for each player.
    
    This script assumes it is run from the root of your project directory
    (e.g., 'football-match-explorer').
    """
    
    # Path to the main directory containing all player folders
    root_player_dir = os.path.join("data", "fbref", "player_top5_europe")
    
    # The name of the subfolder to create for seasonal data
    season_folder_name = "2024-2025"
    
    print(f"Starting organization process for directory: {root_player_dir}\n")
    
    if not os.path.isdir(root_player_dir):
        print(f"Error: Directory not found -> {root_player_dir}")
        print("Please make sure you are running this script from your project's root folder.")
        return

    # Get a list of all player folders
    player_folders = [d for d in os.listdir(root_player_dir) if os.path.isdir(os.path.join(root_player_dir, d))]
    
    if not player_folders:
        print("No player folders found to organize.")
        return
        
    total_players = len(player_folders)
    print(f"Found {total_players} player folders to process.\n")
    
    # Iterate over each player's folder
    for i, player_name in enumerate(player_folders):
        player_path = os.path.join(root_player_dir, player_name)
        
        # Define the path for the new season-specific subfolder
        season_path = os.path.join(player_path, season_folder_name)
        
        print(f"[{i+1}/{total_players}] Processing: {player_name}")
        
        # Create the season subfolder if it doesn't exist
        if not os.path.exists(season_path):
            os.makedirs(season_path)
            print(f"  -> Created subfolder: {season_folder_name}")
            
        # List all files in the player's root directory
        try:
            files_to_move = [f for f in os.listdir(player_path) if os.path.isfile(os.path.join(player_path, f))]
        except OSError as e:
            print(f"  -> Could not read files in {player_path}. Error: {e}. Skipping.")
            continue

        moved_count = 0
        for filename in files_to_move:
            # We only want to move the statistical CSV files
            if filename.endswith(".csv"):
                source_path = os.path.join(player_path, filename)
                destination_path = os.path.join(season_path, filename)
                
                try:
                    # Move the file
                    shutil.move(source_path, destination_path)
                    moved_count += 1
                except Exception as e:
                    print(f"  -> ERROR moving {filename}: {e}")

        if moved_count > 0:
            print(f"  -> Successfully moved {moved_count} CSV file(s) to '{season_folder_name}'.")
        else:
            print(f"  -> No new CSV files to move.")
            
    print("\nOrganization complete! All CSV files have been moved to their respective season folders.")

# --- Run the script ---
if __name__ == "__main__":
    organize_player_files()

# --- END OF FILE organize_player_data.py ---