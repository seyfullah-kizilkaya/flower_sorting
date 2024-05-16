import glob




folder_dir = 'model/*'

for images in glob.iglob(f'{folder_dir}/*'):

    if (images.endswith(".jpg")):
        #print(images)









