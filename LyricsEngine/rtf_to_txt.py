# pip3 install striprtf
import os
from striprtf.striprtf import rtf_to_text

def convert_rtf_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.endswith(".rtf"):
            filepath = os.path.join(input_folder, filename)
            
            # RTF read 
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                rtf_content = file.read()
                
            # Plain Text Conversion
            plain_text = rtf_to_text(rtf_content)
            
            # New .txt file save 
            new_filename = filename.replace(".rtf", ".txt")
            output_path = os.path.join(output_folder, new_filename)
            
            with open(output_path, 'w', encoding='utf-8') as new_file:
                new_file.write(plain_text)
                
            print(f"✅ Converted: {new_filename}")

input_directory = "/Users/siddharthagarwal/Downloads/Duo/Lyrics" 
output_directory = "/Users/siddharthagarwal/Downloads/Lyrics"

convert_rtf_folder(input_directory, output_directory)
print("🎯 All RTF files are converted to plain text!")