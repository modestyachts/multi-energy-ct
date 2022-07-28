from os.path import join, exists
import numpy as np
from imageio.v2 import imread, get_writer

def ct(root):
    # Open the file
    if root == '/data/fabriziov/ct_scans/pepper/scans/':
        dir = "/data/fabriziov/ct_scans/pepper/"
        file = open(dir + "Pepper_CT_parameters.xtekct", "r")
    elif root == '/data/fabriziov/ct_scans/pomegranate/scans/':
        dir = "/data/fabriziov/ct_scans/pomegranate/"
        file = open(dir + "Pomegranate_CT_parameters.xtekct", "r")

    # Save a list of the lines read in a
    list = file.readlines()

    # Loop through the list of lines
    for i in list:
        # Loop through the characters in the line so see if it has an equal sign
        for j in i:
            if j == "=":
                # If there is an equal sign split it and save it to name
                split_str = i.split("=")
                name = split_str[0]

                # As long the part after the equal sign isn't empty
                if split_str[1] != "\n":
                    # Split the value from the newline character and save it to data
                    split_str2 = split_str[1].split("\n")
                    data = split_str2[0]

                    # Save the needed data into variables
                    if name == "Name":
                            data_name = data
                    elif name == "SrcToObject":
                        dist_source_obj = float(data)
                        # print(name + " is equal to " + data)
                    elif name == "SrcToDetector":
                        dist_source_detect = float(data)
                        # print(name + " is equal to " + data)
                    elif name == "DetectorPixelSizeX":
                        detect_pix_size = float(data)
                        # print(name + " is equal to " + data)
                    elif name == "Projections":
                        num_proj = int(data)
                        # print(name + " is equal to " + data)
                    # print()
                break

    # Calculates the distance from the object to the detector(camera)
    radius = dist_source_detect - dist_source_obj

    # This gets all the necessary angles needed for the projection
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)

    # Create an empty array
    # projections = []
    all_c2w = []

    # Loop through the number of projections
    for i in range(0, num_proj):
        # Create an array of 0.0 and reshape it to be a 3*3 matrix
        # my_arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # four_by_four = my_arr.reshape((3,4))
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
        c = np.array([-4.0122, -0.0127, 1.6803])
        four_by_four[0, 3] = 388 * np.cos(rad)
        four_by_four[1, 3] = 388 * np.sin(rad)
        four_by_four[2, 3] = 0.0
        four_by_four[3, 3] = 1.0

        # Add the matrix to the end of the array
        # projections.append(four_by_four)

        # Invert world -> camera to get camera -> world
        # projections[:,-1] = (projections[:,-1] - [400, 220, 200])
        # c2w = np.linalg.inv(np.concatenate([projections[i], [[0,0,0,1]]], axis=0))

        c2w = np.concatenate([four_by_four], axis=0)
        c2w[:3,3] += [radius*np.cos(rad), radius*np.sin(rad), 0]

        # c2w = np.concatenate([projections[i], [[0,0,0,1]]], axis=0)
        all_c2w.append(c2w)
        # print("Matrix #" + str(i+1) + ": ")
        # print(projections[i])
    # print(f'\nall_c2w at {0} is {all_c2w[0]}')
    # print(f'\nall_c2w at {30} is {all_c2w[30]}')
    # print(f'\nall_c2w at {60} is {all_c2w[60]}')
    # print(f'\nall_c2w at {90} is {all_c2w[90]}')
    # print(f'\nall_c2w at {120} is {all_c2w[120]}')
    # c2w[:3,3] += [radius*np.cos(rad), radius*np.sin(rad), 0]

    all_c2w = np.asarray(all_c2w)

    return(all_c2w, num_proj)