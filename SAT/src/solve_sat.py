# imports:

import glob
import json
import logging
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from z3 import *
import time
import os
import multiprocessing
import numpy as np
from sat_rotation import order_enc_rot
from sat_base import order_enc


# ******* User Defined Functions *******

# create output folders if not already created:
def create_folder_structure():
    # root folders:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    outputs_folder = os.path.join(project_folder, 'SAT', 'out')

    # check if output folder already exists:
    if not os.path.exists(outputs_folder):
        os.mkdir(outputs_folder)

        # outputs without considering rotations:
        os.mkdir(os.path.join(outputs_folder, 'base'))
        os.mkdir(os.path.join(outputs_folder, 'base', 'images'))  # cdmo_vlsi/SAT/out/base/images
        os.mkdir(os.path.join(outputs_folder, 'base', 'texts'))  # cdmo_vlsi/SAT/out/base/texts

        # outputs considering rotations:
        os.mkdir(os.path.join(outputs_folder, 'rotation'))
        os.mkdir(os.path.join(outputs_folder, 'rotation', 'images'))  # cdmo_vlsi/SAT/out/rotation/images
        os.mkdir(os.path.join(outputs_folder, 'rotation', 'texts'))  # cdmo_vlsi/SAT/out/rotation/texts

        print("Output folders have been created correctly!")

    # check if runtimes folder already exists:
    runtimes_folder = os.path.join(project_folder, 'runtimes')

    if not os.path.exists(runtimes_folder):
        os.mkdir(runtimes_folder)  # cdmo_vlsi/runtimes
        print("Runtimes folder has been created correctly!")

    # check if heights folder already exists:
    heights_folder = os.path.join(project_folder, 'heights')

    if not os.path.exists(heights_folder):
        os.mkdir(heights_folder)  # cdmo_vlsi/heights
        print("Heights folder has been created correctly!")

    # instances folder
    instances_folder = os.path.join(project_folder, 'instances')

    return project_folder, outputs_folder, runtimes_folder, heights_folder, instances_folder


def get_runtimes(args):
    # define path and name of runtime file:
    file_name = f'SAT' \
                f'{"-sb" if args.symmetry_breaking else ""}' \
                f'{"-rot" if args.rotation else ""}' \
                f'.json'

    file_path = os.path.join(runtimes_folder, file_name)

    # if file exists load it and extract dict values, otherwise return empty dict:
    if os.path.isfile(file_path):  # z3 I hate your timeout bug so much
        with open(file_path) as f:

            # load dictionary:
            dictionary = json.load(f)
            data = {}

            for k, v in dictionary.items():
                int_key = int(k)
                data[int_key] = v

    else:
        data = {}

    return data, file_path

#get upper bound
def get_upperbound(heights, widths, n, w):

    # initialize variables:
    previous_row_height = 0     #to store the highest height of the previous row
    h_res = 0                   #maximum found height
    acc_width = 0               #accumulated horizontal width

    # cycle all blocks
    for i in range(n):

        # if the new inserted block does not exceed the width of the plate, then keep inserting blocks horizontally:
        if acc_width + widths[i] < w:
            # add to the value of accumulated width the width of the i-th block:
            acc_width += widths[i]

            # if the height of current block is higher than the highest height found since now, then update it:
            if heights[i] + previous_row_height > h_res:
                h_res = heights[i] + previous_row_height

        #otherwise save current maximum height and start a new line:
        else:
            #re-initialize the value of accumulated width to the width of i-th block (the first of new row)
            acc_width = widths[i]

            #set maximum height of previous row:
            previous_row_height = h_res

            #update maximum height:
            h_res = h_res + heights[i]

    return h_res


# solve given instance:
def start_solving(instance, runtimes, index, args):
    print("-" * 20)
    print(f'Solving Instance {index}')

    # select model based on whether rotation is enabled or not:
    if args.rotation:
        model = order_enc_rot
    else:
        model = order_enc

    # start a new process:
    p = multiprocessing.Process(target=model, args=(instance, index, args))
    p.start()

    # start a timer to track the elapsed time:
    start_time = time.time()

    # stop process if timeout exceeded:
    p.join(args.timeout)

    # kill thread if still active:
    if p.is_alive():
        print("Timeout Reached without finding a solution")
        p.kill()
        p.join()

    # save elapsed time within runtimes and save file:
    elapsed_time = time.time() - start_time
    runtimes[index] = elapsed_time

    with open(runtimes_filename, 'w') as f:
        json.dump(runtimes, f)

    print(f"Time elapsed: {elapsed_time:.2f}")


# ******* Main Code *******
if __name__ == "__main__":

    # create output folders structure
    project_folder, outputs_folder, runtimes_folder, heights_folder, instances_folder = create_folder_structure()

    # define command line arguments:
    parser = ArgumentParser()
    parser.add_argument('-s', '--start', type=int, help='First instance to solve', default=1)
    parser.add_argument('-e', '--end', type=int, help='Last instance to solve', default=40)
    parser.add_argument('-t', '--timeout', type=int, help='Timeout (s)', default=300)
    parser.add_argument('-r', '--rotation', action="store_true", help="enables circuits rotation")
    parser.add_argument('-sb', '--symmetry_breaking', action="store_true", help="enables symmetry breaking")

    args = parser.parse_args()

    runtimes, runtimes_filename = get_runtimes(args)

    # solve Instances in range:
    print(f'Solving instances {args.start} - {args.end} using SAT model')

    for i in range(args.start, args.end + 1):
        print('=' * 20)
        print(f'Instance {i}')

        # open instance and extract instance data:
        with open(os.path.join(instances_folder, f'ins-{i}.txt')) as f:
            lines = f.readlines()

        # get list of lines:
        lines = [l.strip('\n') for l in lines]

        # get width of the map:
        w = int(lines[0].strip('\n'))

        # get number of blocks:
        n = int(lines[1].strip('\n'))

        # get list of the dimensions of each circuit:
        dim = [ln.split(' ') for ln in lines[2:]]

        # get x and y coordinates:
        x, y = list(zip(*map(lambda x_y: (int(x_y[0]), int(x_y[1])), dim)))

        # sort circuits by area:
        xy = np.array([x, y]).T
        areas = np.prod(xy, axis=1)
        sorted_idx = np.argsort(areas)[::-1]
        xy = xy[sorted_idx]
        x = list(map(int, xy[:, 0]))
        y = list(map(int, xy[:, 1]))

        # lower and upper bounds for height:
        min_area = np.prod(xy, axis=1).sum()
        minh = int(min_area / w)
        maxh = get_upperbound(y, x, n, w)

        # Pass instance parameters to the solver:
        instance = {"w": w, 'n': n, 'inputx': x, 'inputy': y, 'minh': minh, 'maxh': maxh}

        # begin to find solution:
        start_solving(instance, runtimes, i, args)
