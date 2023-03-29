from os.path import join, exists
import numpy as np
import os
# import imageio
from imageio.v2 import imread, get_writer

def ct(root):

    num_proj = 0
    # Iterate directory
    for path in os.listdir(root):
        # check if current path is a file
        if os.path.isfile(os.path.join(root, path)):
            num_proj += 1
    # num_proj = 360
   
    # DELETE THIS NEXT LINE AFTER CHECKING SOURCE
    # num_proj -= 1

    # This gets all the necessary angles needed for the projection
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)

    # Create an empty array
    all_c2w = []

    # Loop through the number of projections
    for i in range(0, num_proj):
        four_by_four = np.zeros((4,4))

        # Convert all the angles from degrees to redians
        # rad = math.radians(angles[i])
        rad = angles[i]
        # print(f'angle is {angles[i]}')

        # Set the rotation matrix for each angle
        four_by_four[0,0] = np.cos(rad)
        four_by_four[0,1] = -np.sin(rad)
        four_by_four[1,0] = np.sin(rad)
        four_by_four[1,1] = np.cos(rad)
        four_by_four[2,2] = 1

        # Giacomo's way
        # c = np.array([-4.0122, -0.0127, 1.6803])
        four_by_four[0, 3] = np.cos(rad)
        four_by_four[1, 3] = np.sin(rad)
        four_by_four[2, 3] = 0.0
        four_by_four[3, 3] = 1.0

        c2w = np.concatenate([four_by_four], axis=0)
        c2w[:3,3] += [np.cos(rad), np.sin(rad), 0] # This may be a reason why the numper of projections and the number of reconstruction images is 338 instead of 359
        # c2w[:3,3] += [np.sin(rad), 0, np.cos(rad)] #already tried, looks wierd
        # c2w[:3,3] += [np.cos(rad), 0, np.sin(rad)] 
        # c2w[:3,3] += [np.sin(rad), np.cos(rad), 0] # tmux 2
        # c2w[:3,3] += [0, np.cos(rad), np.sin(rad)] #tmux 3 orgsegs
        # c2w[:3,3] += [0, np.sin(rad), np.cos(rad)] # tmux 1 orgsegs1


        all_c2w.append(c2w)

    all_c2w = np.asarray(all_c2w)

    return(all_c2w, num_proj)
