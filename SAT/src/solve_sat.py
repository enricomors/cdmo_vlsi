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

from order_enc_rot import order_enc_rot
from order_enc import order_enc

def get_runtimes(args):
    if not os.path.exists('../../runtimes'):
        os.mkdir('../../runtimes')
    name = f'../../runtimes/SAT' \
           f'{"-sb" if args.symmetry_breaking else ""}' \
           f'{"-rot" if args.rotation else ""}' \
           f'.json'
    if os.path.isfile(name):  # z3 I hate your timeout bug so much
        with open(name) as f:
            data = {int(k): v for k, v in json.load(f).items()}
    else:
        data = {}
    return data, name


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
    plt.savefig(f'../../SAT/out/fig-ins-{index}.png')
    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)


# Try to solve instance before timeout
def start_solving(instance, runtimes, index, args):
    print("-" * 20)
    print(f'Solving Instance {index}')

    if args.rotation:
        model = order_enc_rot  # use rotation model
    else:
        model = order_enc  # use standard model

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
    parser = ArgumentParser()
    parser.add_argument('-s', '--start', type=int, help='First instance to solve', default=1)
    parser.add_argument('-e', '--end', type=int, help='Last instance to solve', default=40)
    parser.add_argument('-t', '--timeout', type=int, help='Timeout (s)', default=300)
    parser.add_argument('-r', '--rotation', action="store_true", help="enables circuits rotation")
    parser.add_argument('-sb', '--symmetry_breaking', action="store_true", help="enables symmetry breaking")

    args = parser.parse_args()

    # Creates output folder
    if not os.path.exists(f'../../SAT/out'):
        os.mkdir(f'../../SAT/out')

    runtimes, runtimes_filename = get_runtimes(args)

    # Solve Instances in range
    for i in range(args.start, args.end + 1):
        print('=' * 20)
        print(f'Instance {i}')
        with open(f'../../instances/ins-{i}.txt') as f:
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
        minh = int(min_area / w)
        maxh = np.sum(y)

        # Pass instance parameters to the solver
        instance = {"w": w, 'n': n, 'inputx': x, 'inputy': y, 'minh': minh, 'maxh': maxh}
        start_solving(instance, runtimes, i, args)