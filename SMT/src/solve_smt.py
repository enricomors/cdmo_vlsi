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
from model import solve_instance
from model_rot import solve_instance_rot

def create_folder_structure():
    # root folders:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    outputs_folder = os.path.join(project_folder, 'SMT', 'out')

    # check if output folder already exists:
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

    # check if runtimes folder already exists:
    runtimes_folder = os.path.join(project_folder, 'runtimes')

    if not os.path.exists(runtimes_folder):
        os.mkdir(runtimes_folder)  # cdmo_vlsi/runtimes
        print("Runtimes folder has been created correctly!")

    return project_folder, outputs_folder, runtimes_folder


#get runtimes
def get_runtimes(args):

    # create runtime folder if doesn't exists:
    runtimes_folder = os.path.join(project_folder, 'runtimes')

    if not os.path.exists(runtimes_folder):
        os.mkdir(runtimes_folder)
        print("Runtimes folder has been created correctly")

    s = f"SMT"\
        f'{"-sb" if args.symmetry_breaking else ""}'\
        f'{"-rot" if args.rotation else ""}'\
        '.json'

    file_name = os.path.join(runtimes_folder, s)

    if os.path.isfile(file_name):  # z3 I hate your timeout bug so much
        with open(file_name) as f:
            data = {int(k): v for k, v in json.load(f).items()}
    else:
        data = {}

    return data, file_name

'''
def plot_board(width, height, blocks, index, show_plot=False, show_axis=False):
    cmap = plt.cm.get_cmap('jet', len(blocks))
    fig, ax = plt.subplots(figsize=(10, 10))
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'
       
        if rotation is not None:
           label += f', R={1 if rotation[component] else 0}'

        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('length', fontsize=15)
    # ax.legend()
    ax.set_title(f'Instance {index}, size (WxH): {width}x{height}', fontsize=22)
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(project_folder, "out", "images", f"fig-ins-{index}.png"))

    figure_folder = os.path.join(project_folder, "SMT", "out", "images", f"ins-{instance}.png")
    plt.savefig(figure_folder)
    print(f"figure ins-{instance}.png has been correctly saved at path '{figure_folder}'")

    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)
'''

# Try to solve instance before timeout
def start_solving(instance, runtimes, index, args):
    print("-" * 20)
    print(f'Solving Instance {index}')

    if args.rotation:
        model = solve_instance_rot  # use rotation model
    else:
        model = solve_instance     # use standard model

    p = multiprocessing.Process(target=model, args=(instance, index, args))

    p.start()
    start_time = time.time()
    p.join(args.timeout)

    # If thread is still active kill it
    if p.is_alive():
        print("Timeout Reached without finding a solution")
        p.kill()
        p.join()

    # Save runtime to json
    elapsed_time = time.time() - start_time
    runtimes[index] = elapsed_time
    with open(runtimes_filename, 'w') as f:
        json.dump(runtimes, f)
    print(f"Time elapsed: {elapsed_time:.2f}")


if __name__ == "__main__":

    # create output folders structure
    project_folder, outputs_folder, runtimes_folder = create_folder_structure()

    parser = ArgumentParser()
    parser.add_argument('-s', '--start', type=int, help='First instance to solve', default=1)
    parser.add_argument('-e', '--end', type=int, help='Last instance to solve', default=40)
    parser.add_argument('-t', '--timeout', type=int, help='Timeout (s)', default=300)
    parser.add_argument('-r', '--rotation', action="store_true", help="enables circuits rotation")
    parser.add_argument('-sb', '--symmetry_breaking', action="store_true", help="enables symmetry breaking")

    args = parser.parse_args()

    runtimes, runtimes_filename = get_runtimes(args)
    '''
    print(runtimes_filename)
    # Solve Instances in range
    file_names = os.listdir(os.path.join(project_folder, 'instances'))
    file_names.sort(key=mySort)
    print(file_names)
    i = 1
    '''
    instances_folder = os.path.join(project_folder, 'instances')

    for i in range(args.start, args.end + 1):
        print('=' * 20)
        print(f'Instance {i}')

        with open(os.path.join(instances_folder, f'ins-{i}.txt')) as f:
            lines = f.readlines()

        lines = [l.strip('\n') for l in lines]
        w = int(lines[0].strip('\n'))
        n = int(lines[1].strip('\n'))
        dim = [ln.split(' ') for ln in lines[2:]]
        x, y = list(zip(*map(lambda x_y: (int(x_y[0]), int(x_y[1])), dim)))

        # Sort circuits by area
        xy = np.array([x, y]).T
        areas = np.prod(xy, axis=1)
        sorted_idx = np.argsort(areas)[::-1]
        xy = xy[sorted_idx]
        x = list(map(int, xy[:, 0]))
        y = list(map(int, xy[:, 1]))

        # lower and upper bounds for height
        min_area = np.prod(xy, axis=1).sum()
        minh = max(max(y),int(min_area / w))
        maxh = np.sum(y)

        # Pass instance parameters to the solver
        instance = {"w": w, 'n': n, 'inputx': x, 'inputy': y, 'minh': minh, 'maxh': maxh}
        start_solving(instance, runtimes, i, args)

        i += 1