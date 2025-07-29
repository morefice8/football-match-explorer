# scripts/process_images.py
import os
from PIL import Image, ImageDraw

# Assicurati che il path sia corretto, partendo dalla radice del tuo progetto
DATA_PATH = os.path.join("data", "fbref", "player_top5_europe")

def create_circular_headshot(image_path):
    """
    Converte un'immagine quadrata in una circolare con sfondo trasparente.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # Crea una maschera circolare
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + img.size, fill=255)
        
        # Applica la maschera
        output = Image.new('RGBA', img.size)
        output.paste(img, (0, 0), mask)
        
        # Salva la nuova immagine
        output_path = image_path.replace('headshot.png', 'headshot_circular.png')
        output.save(output_path)
        return True
    except Exception as e:
        print(f"  -> Errore nel processare {image_path}: {e}")
        return False

if __name__ == "__main__":
    print("--- Inizio processamento immagini dei giocatori ---")
    if not os.path.isdir(DATA_PATH):
        print(f"Errore: La cartella {DATA_PATH} non è stata trovata.")
    else:
        player_folders = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
        total_players = len(player_folders)
        processed_count = 0
        
        for i, player_folder in enumerate(player_folders):
            if (i + 1) % 100 == 0:
                print(f"  ...analizzando giocatore {i+1}/{total_players}")
                
            headshot_path = os.path.join(DATA_PATH, player_folder, f"{player_folder}.png")
            
            if os.path.exists(headshot_path):
                if create_circular_headshot(headshot_path):
                    processed_count += 1
        
        print(f"--- Processamento completato. {processed_count}/{total_players} immagini create con successo. ---")