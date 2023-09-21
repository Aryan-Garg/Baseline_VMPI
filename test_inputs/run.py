import os
import random


    
img_root = '/media/data/prasan/datasets/LF_video_datasets/'
directories = os.listdir(img_root)

for directory in ['Hybrid', 'Kalantari', 'Stanford', 'TAMULF']:
    store_dirs = directory
    os.makedirs(store_dirs, exist_ok=True)
    set = 'test'
    test_file = open(os.path.join(store_dirs, f'{set}_files.txt'), 'w')

    try:
        seqs = sorted(os.listdir(os.path.join(img_root, directory, set)))
        for seq in seqs:
            imgs = sorted(os.listdir(os.path.join(img_root, directory, set, seq)))
            for img in imgs:
                img_path = os.path.join(directory, set, seq, img)
                test_file.write('{0}\n'.format(img_path))

    except:
        continue

    test_file.close()
