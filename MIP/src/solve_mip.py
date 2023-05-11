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
        os.mkdir(os.path.join(outputs_folder, 'base', 'images'))  # cdmo_vlsi/CP/out/base/images
        os.mkdir(os.path.join(outputs_folder, 'base', 'texts'))  # cdmo_vlsi/CP/out/base/texts

        # outputs considering rotations:
        os.mkdir(os.path.join(outputs_folder, 'rotation'))
        os.mkdir(os.path.join(outputs_folder, 'rotation', 'images'))  # cdmo_vlsi/CP/out/rotation/images
        os.mkdir(os.path.join(outputs_folder, 'rotation', 'texts'))  # cdmo_vlsi/CP/out/rotation/texts

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

'''
# creates and plots the colour map with rectangles:
def plot_board(width, height, blocks, i, show_plot=False, show_axis=False, verbose=False):

    #define pyplot colour map of len(blocks) number of colours:
    cmap = plt.cm.get_cmap('jet', len(blocks))

    #define figure size:
    fig, ax = plt.subplots(figsize=(10, 10))

    # add each rectangle block in the colour map:
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'

        #PERCHE' E' COMMENTATO QUESTO CODICE??
        #if rotation is not None:
        #    label += f', R={1 if rotation[component] else 0}'
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))

    # set plot properties:
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('length', fontsize=15)
    # ax.legend() TOGLIERE?
    ax.set_title(f'Instance {i}, size (WxH): {width}x{height}', fontsize=22)

    # print axis if wanted
    if show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    # save colormap in .png format at given path:
    figure_path = os.path.join(outputs_folder, "base", "images", f"ins-{instance}.png")
    plt.savefig(figure_path)

    # check if file was saved at correct path:
    if verbose:
        if os.path.exists(figure_path):
            print(f"figure ins-{instance}.png has been correctly saved at path '{figure_path}'")

    # to show plot:
    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)
'''
#CHE SI FA CON QUESTOOO?
# def solve_mip(instance, index, args):
#
#     # initialize instance parameters
#     n = instance['n']
#     w = instance['w']
#
#     x = instance['inputx']
#     y = instance['inputy']
#
#     # height bounds
#     minh = instance['minh']
#     maxh = instance['maxh']
#
#     # Solver
#     with gp.Env(empty=True) as env:
#         env.setParam('OutputFlag', 0)
#         env.start()
#         m = gp.Model('vlsi', env=env)
#
#
#     # Variables definition
#     xhat = [m.addVar(vtype=GRB.INTEGER, name=(f'xhat_{i}'), lb=0) for i in range(n)]
#     yhat = [m.addVar(vtype=GRB.INTEGER, name=(f'yhat_{i}'), lb=0) for i in range(n)]
#     h = m.addVar(vtype=GRB.INTEGER, name='h')
#
#     # Constraints
#
#     # Height bounds
#     m.addConstr(minh <= h)
#     m.addConstr(h <= maxh)
#
#    # Circuit placement bound constraints
#     for i in range(n):
#         m.addConstr(xhat[i] + x[i] <= w)
#         m.addConstr(yhat[i] + y[i] <= h)
#
#         # TODO: var Hints
#
#         m.addConstr(yhat[i] <= maxh - y[i])
#
#     # Non overlappig constraints
#     for i in range(n):
#         for j in range(i + 1, n):
#
#             big_m = m.addVars(4, vtype=GRB.BINARY)
#
#             m.addGenConstrIndicator(big_m[0], True, xhat[i] + x[i] <= xhat[j])
#             m.addGenConstrIndicator(big_m[1], True, yhat[i] + y[i] <= yhat[j])
#             m.addGenConstrIndicator(big_m[2], True, xhat[j] + x[j] <= xhat[i])
#             m.addGenConstrIndicator(big_m[3], True, yhat[j] + y[j] <= yhat[i])
#
#             # Or on indicators
#             or_out = m.addVar(vtype=GRB.BINARY)
#             m.addGenConstrOr(or_out, big_m)
#             m.addConstr(or_out == 1)
#
#     # Symmetry breaking constraints
#     if args.symmetry_breaking:
#         pass
#
#
#     m.update()
#     m.setObjective(h, GRB.MINIMIZE)
#
#     m.optimize()
#
#     if m.Status == GRB.OPTIMAL:
#         print(f'Found optimal solution')
#
#         xhat = [int(var.X) for var in xhat]
#         yhat = [int(var.X) for var in yhat]
#         h = int(h.X)
#
#         instance['h'] = h

    #         instance['xhat'] = xhat
#         instance['yhat'] = yhat
#
#         print(f'x = {xhat}')
#         print(f'y = {yhat}')
#         print(f'h = {h}')
#
#         # instance['rotation'] = rotations
#
#         out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
#         out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
#                           for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'],
#                                                           instance['xhat'], instance['yhat'])])
#
#         with open(f'../../MIP/out/out-{index}.txt', 'w') as f:
#             f.write(out)
#
#         res = [(xi, yi, xhati, yhati)
#                for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'], instance['xhat'], instance['yhat'])]
#         plot_board(instance['w'], instance['h'], res, index)
#
#     else:
#         print('No optimal solution found')

#solve given instance:
def start_solving(instance, runtimes, index, args):
    print("-" * 20)
    print(f'Solving Instance {index}')

    #select model based on whether rotation is enabled or not:
    if args.rotation:
        model = mip_rotation
    else:
        model = mip_base

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

        #open instance and extract instance data:
        with open(os.path.join(instances_folder, f'ins-{i}.txt')) as f:
            lines = f.readlines()

        #get list of lines:
        lines = [l.strip('\n') for l in lines]

        #get width of the map:
        w = int(lines[0].strip('\n'))

        #get number of blocks:
        n = int(lines[1].strip('\n'))

        #get list of the dimensions of each circuit:
        dim = [ln.split(' ') for ln in lines[2:]]

        #get x and y coordinates:
        x, y = list(zip(*map(lambda x_y: (int(x_y[0]), int(x_y[1])), dim)))

        #sort circuits by area:
        xy = np.array([x, y]).T
        areas = np.prod(xy, axis=1)
        sorted_idx = np.argsort(areas)[::-1]
        xy = xy[sorted_idx]
        x = list(map(int, xy[:, 0]))
        y = list(map(int, xy[:, 1]))

        #lower and upper bounds for height:
        min_area = np.prod(xy, axis=1).sum()
        minh = int(min_area / w)
        maxh = np.sum(y)

        #pass instance parameters to the solver:
        instance = {"w": w, 'n': n, 'inputx': x, 'inputy': y, 'minh': minh, 'maxh': maxh}

        #begin to find solution:
        start_solving(instance, runtimes, i, args)
