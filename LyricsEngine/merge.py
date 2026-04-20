import os

input_folder = "Sidlyrics"
output_file = "dataset/siddharth_lyrics.txt" 

if not os.path.exists('dataset'):
    os.makedirs('dataset')

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                outfile.write(f"--- {filename} ---\n")
                outfile.write(infile.read())
                outfile.write("\n\n")

print("✅All Lyrics have been merged")