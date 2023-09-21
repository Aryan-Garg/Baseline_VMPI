import os
import random


store_dirs = 'TAMULF+Stanford_corr/'
os.makedirs(store_dirs, exist_ok=True)
train_file = open(os.path.join(store_dirs, 'train_files.txt'), 'w')
val_file = open(os.path.join(store_dirs, 'val_files.txt'), 'w')
    
img_root = '/media/data/prasan/datasets/LF_datasets/'
directories = os.listdir(img_root)

for directory in ['TAMULF_corr', 'Stanford']:
    imgs = sorted(os.listdir(os.path.join(img_root, directory, 'train')))
    num_imgs = len(imgs)
    for img in imgs:
        prob = random.random()
        img_path = os.path.join(directory, 'train', img)
        if prob > 0.9:
            val_file.write('{0}\n'.format(img_path))
        else:
            train_file.write('{0}\n'.format(img_path))

train_file.close()
val_file.close()
