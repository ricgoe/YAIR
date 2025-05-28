import os 
from PIL import Image

root_path = "/Volumes/Big Data/data/image_data"

with open('image_info/w_h_data.csv', 'w') as f:
    f.write('path,width,height\n')
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            try:
                p = os.path.join(dirpath, filename)
                im = Image.open(p)
                width, height = im.size
                f.write(f'{p},{width},{height}\n')
            except Exception as e:
                print(f'{filename} is not an image')
                continue
        
