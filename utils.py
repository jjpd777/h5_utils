import h5py as h5
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def keys(f):
    return [key for key in f.keys()]

def load_dataset(h5_name):
    ''' Load in images from Hdf5 file as np.arrays.
    '''
    f = h5.File(h5_name,'r')
    h5_keys = keys(f)
    return_list = []
    test_titles = []
    for key in h5_keys:
        extracted_group = f[key]
        extract_images = []
        for i in extracted_group.keys():
            if(key == 'test_cases'):
                test_titles.append(i)

            img = np.array(extracted_group[i])
            extract_images.append(img)

        return_list.append(extract_images)
    return_list.append(test_titles)

    return return_list

def create_dataset(keys_input,values_path, h5_name):
    '''Create an Hdf5 file from a dictionary of images'''
    key_path_pairs = {}
    for i in range(len(keys_input)):
        key_path_pairs[keys_input[i]] = values_path[i]

    for key, value in key_path_pairs.items():
        h5_key = key
        path = Path(value)

        images = path.glob('*.jpg')
        f = h5.File(h5_name,'a')

        mines_group = f.create_group(h5_key)

        for image in images:
            vector_img = cv2.imread(str(image))
            name = str(image).split('/')[-1]
            mines_group.create_dataset(name, data= vector_img, dtype='uint8')


# check = load_dataset()
© 2019 Git
