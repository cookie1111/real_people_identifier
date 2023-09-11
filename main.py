import os
import csv
import keyboard
import matplotlib.pyplot as plt
from PIL import Image
#import readchar



# WXPYTHON -> https://stackoverflow.com/questions/54503994/how-to-make-window-overlay-on-top-of-browser-games-exc-with-wxpython
# QT -> https://stackoverflow.com/questions/25950049/creating-a-transparent-overlay-with-qt

# Define the path to the main folder
main_folder = '/media/sebastjan/663C012C3C00F8B7/Users/Sebastjan/Downloads/InstaCities1M/InstaCities1M'
img_folder = os.path.join(main_folder, 'img')
caption_folder = os.path.join(main_folder, 'captions')

# Define the output CSV file and log file
csv_file = 'output.csv'
log_file = 'log.txt'

# Define the subfolders
subfolders = ['train', 'test', 'val']
cities = ['london', 'melbourne', 'newyork', 'sanfrancisco', 'singapore', 'sydney', 'toronto']

# Check the log file to see where we left off
try:
    with open(log_file, 'r') as f:
        last_processed = f.read().strip()
        last_processed, last_city = last_processed.split(',')
        print(last_processed, last_city)
except FileNotFoundError:
    last_processed = None



# Open the CSV file for writing
with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)

    # If the CSV file is empty, write the header
    if file.tell() == 0:
        writer.writerow(['ID', 'City', 'Person'])

    # Loop through the subfolders and cities
    subfolder = "train"
    for city in cities:
        if last_city is not None and city != last_city:
            continue
        # Define the path to the images and captions for this city
        img_path = os.path.join(img_folder, subfolder, city)
        caption_path = os.path.join(caption_folder, subfolder, city)
        print(img_path)

        # Loop through the images in this city's folder
        for i, filename in enumerate(sorted(os.listdir(img_path))):
            print(i)
            if i>1000:
                break
            # Only process .jpg files
            if filename.endswith('.jpg'):
                # If we have a last processed image and we haven't reached it yet, skip this image
                if last_processed is not None and filename != last_processed:
                    continue

                # Update the log file to reflect last unprocessed
                with open(log_file, 'w') as f:
                    last_processed = None
                    last_city = None
                    f.write(filename + f",{city}")
                # Display the image
                img = Image.open(os.path.join(img_path, filename))
                plt.imshow(img)
                plt.show(block=False)

                # Display the caption
                with open(os.path.join(caption_path, filename[:-4] + '.txt'), 'r') as f:
                    caption = f.read()
                    comp = caption.lower()
                    #print(caption)
                if "www." in comp or "forsale" in comp or "link in bio" in comp \
                    or "to book" in comp or "available for" in comp \
                    or "download the" in comp or "check out" in comp:
                    person = False
                else:
                    print(caption)
                    # Wait for the user's input
                    while True:
                        #print("Press 'r' for real person, 'n' for not: ")
                        #c = repr(readchar.readkey())
                        c = input("Press 'Enter' for real person, 'n' for not: ")
                        if c == '':
                            person = True
                            break
                        elif c == 'n':
                            person = False
                            break

                # Close the image
                plt.close()

                # Write the result to the CSV file
                writer.writerow([filename[:-4], city, person])



        #last_processed = None