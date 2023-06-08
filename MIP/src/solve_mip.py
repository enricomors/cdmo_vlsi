#imports
import json
import multiprocessing
import os
import time
from argparse import ArgumentParser

import gurobipy as gp
from gurobipy import GRB

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mip_base import mip_base
from mip_rotation import mip_rotation
#from mip_rot_2 import mip_rotation_2
#******* User Defined Functions *******

#create output folders if not already created:
def create_folder_structure():
    #root folders:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    outputs_folder = os.path.join(project_folder, 'MIP', 'out')

    #check if output folder already exists:
    if not os.path.exists(outputs_folder):
        os.mkdir(outputs_folder)

        # outputs without considering rotations:
        os.mkdir(os.path.join(outputs_folder, 'base'))
        os.mkdir(os.path.join(outputs_folder, 'base', 'images'))  # cdmo_vlsi/MIP/out/base/images
        os.mkdir(os.path.join(outputs_folder, 'base', 'texts'))  # cdmo_vlsi/MIP/out/base/texts

        # outputs considering rotations:
        os.mkdir(os.path.join(outputs_folder, 'rotation'))
        os.mkdir(os.path.join(outputs_folder, 'rotation', 'images'))  # cdmo_vlsi/MIP/out/rotation/images
        os.mkdir(os.path.join(outputs_folder, 'rotation', 'texts'))  # cdmo_vlsi/MIP/out/rotation/texts

        print("Output folders have been created correctly!")

    #check if runtimes folder already exists:
    runtimes_folder = os.path.join(project_folder, 'runtimes')

    if not os.path.exists(runtimes_folder):
        os.mkdir(runtimes_folder)  # cdmo_vlsi/runtimes
        print("Runtimes folder has been created correctly!")

    #instance folder:
    instances_folder = os.path.join(project_folder, 'instances')

    return project_folder, outputs_folder, runtimes_folder, instances_folder

def get_runtimes(args):
    #define path and name of runtime file:
    file_name = 'MIP'\
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


def Hmax(h, ws, n, w):

    res = []
    s = 0
    start = 0
    for i in range(n):
        if s + ws[i] > w:
            try:
                res.append(max(h[start:i]))
            except:
                None
            start = i
            s = h[i]
        else:
            s += h[i]
    res.append(max(h[start:]))
    return sum(res)


#solve given instance:
def start_solving(instance, runtimes, index, args):
    print("-" * 20)
    print(f'Solving Instance {index}')

    #select model based on whether rotation is enabled or not:
    if args.rotation:
        model = mip_rotation
        print("Rotation model chosen")
    else:
        model = mip_base
        print(model.__name__)
    #start a new process:
    p = multiprocessing.Process(target=model, args=(instance, index, args))
    p.start()

    #start a timer to track the elapsed time:
    start_time = time.time()

    #stop process if timeout exceeded:
    p.join(args.timeout)

    #kill thread if still active:
    if p.is_alive():
        print("Timeout Reached without finding a solution")
        p.kill()
        p.join()

    #save elapsed time within runtimes and save file:
    elapsed_time = time.time() - start_time
    runtimes[index] = elapsed_time

    with open(runtimes_filename, 'w') as f:
        json.dump(runtimes, f)

    print(f"Time elapsed: {elapsed_time:.2f}")

#******* Main Code *******
if __name__ == "__main__":

    #create folders structure:
    project_folder, outputs_folder, runtimes_folder, instances_folder = create_folder_structure()

    #define command line arguments:
    parser = ArgumentParser()

    parser.add_argument('-s', '--start', type=int, help='First instance to solve', default=1)
    parser.add_argument('-e', '--end', type=int, help='Last instance to solve', default=40)
    parser.add_argument('-t', '--timeout', type=int, help='Timeout (s)', default=300)
    parser.add_argument('-r', '--rotation', action="store_true", help="enables circuits rotation")
    parser.add_argument('-sb', '--symmetry_breaking', action="store_true", help="enables symmetry breaking")

    args = parser.parse_args()

    #QUI DOBBIAMO FARE L'ARGUMENT CHECK COME IN CP??

    #get runtimes:
    runtimes, runtimes_filename = get_runtimes(args)

    #solve Instances in range:
    print(f'Solving instances {args.start} - {args.end} using MIP model')

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
        maxh = Hmax(y, x, n, w)

        # pass instance parameters to the solver:
        instance = {"w": w, 'n': n, 'inputx': x, 'inputy': y, 'minh': minh, 'maxh': maxh}

        # begin to find solution:
        start_solving(instance, runtimes, i, args)
