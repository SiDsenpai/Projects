
import lyricsgenius
import os

GENIUS_TOKEN = "kTGkJqqXgAw9nezrb7m7qgtwHCpWlridVOJocLNgCekwxj0ESAu4HNAMYe-WMDOB"

genius = lyricsgenius.Genius(GENIUS_TOKEN)

# Removes exessive stuff like chorus
genius.remove_section_headers = True 
genius.skip_non_songs = True

def download_lyrics(artist_name, max_songs=50):
    print(f"📥 {artist_name} Fetiching lyrics...")
    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
            
        file_path = f"dataset/{artist_name.replace(' ', '_').lower()}_lyrics.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            for song in artist.songs:
                f.write(f"--- {song.title} ---\n")
                f.write(song.lyrics)
                f.write("\n\n")
        
        print(f"✅ {artist_name} data is saved: {file_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

# --- EXECUTE ---
artists_to_track = ["KR$NA", "Seedhe Maut"]

for artist in artists_to_track:
    download_lyrics(artist, max_songs=50)