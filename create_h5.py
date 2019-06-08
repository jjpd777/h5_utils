import h5py as h5
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def extract_labels(list_obj):
    ships = []
    not_ships = []

    for path in list_obj:
        label = int(str(path).split('/')[1][0])
        if label ==1:
            ships.append(str(path))
        else:
            not_ships.append(str(path))
    return(ships,not_ships)

def get_list_paths(path):
    path_to_images = Path(path)
    generator_paths = path_to_images.glob('*.png')
    print("This is a Path object",type(path_to_images))
    print("This is a Generator object", type(generator_paths))
    list_paths = list(generator_paths)

    return list_paths


def create_h5(path):

    list_paths = get_list_paths(path)
    total_images = len(list_paths)
    print("The total number of images is: ",total_images)

    ship,no_ship = extract_labels(list_paths)
    img_dim = cv2.imread(ship[0]).shape

    print("The dimensions of the images are: ",img_dim)

    dim1,dim2,dim3 = img_dim[0],img_dim[1],img_dim[2]
    h5_ship_shape = (len(ship),dim1,dim2,dim3)
    h5_no_ship_shape = (len(no_ship),dim1,dim2,dim3)
    print("The shape of the ship dataset is:", h5_ship_shape)
    print("The shape of the no_ship dataset is:", h5_no_ship_shape)
    with h5.File('data.hdf5', 'w') as f:
        g1 = f.create_group('ships')
        d1 = g1.create_dataset('data',h5_ship_shape)
        for i in range(0,len(ship)):
            d1[i] = cv2.imread(str(ship[i]))

        g2= f.create_group('no_ships')
        d2 = g2.create_dataset('data',h5_no_ship_shape)
        for i in range(0,len(no_ship)):
            d2[i] = cv2.imread(no_ship[i])

        print("The name of the ships dataset is ",d1.name)
        print("The name of the no_ships dataset is ",d2.name)
        f.close()

def load_dataset():
    with h5.File('data.hdf5', "r") as f:
        print("The hdf5 file acts as a big dictionary where the groups are the keys:",f.keys())

        print("Each group has a dataset associated with it:",f['ships'].keys(),f['no_ships'].keys())
        positive_cases = f['ships']['data'][:]
        negative_cases = f['no_ships']['data'][:]
        print("Images are stored as numpy arrays:",type(positive_cases))
        print("The shape of the positive cases is:",positive_cases.shape)
        print("The shape of the negative cases is:",negative_cases.shape)
        print("Checking one element of each dataset",d1[0].shape,d2[0].shape)
        f.close()

    return positive_cases, negative_cases

create_h5('./shipsnet/')
##load_dataset()
