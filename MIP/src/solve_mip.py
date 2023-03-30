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



def get_runtimes(arguments):
    if not os.path.exists('../../runtimes'):
        os.mkdir('../../runtimes')
    name = f'../../runtimes/MIP' \
           f'{"-sb" if arguments.symmetry_breaking else ""}' \
           f'{"-rot" if arguments.rotation else ""}' \
           f'.json'
    if os.path.isfile(name):  # z3 I hate your timeout bug so much
        with open(name) as f:
            data = {int(k): v for k, v in json.load(f).items()}
    else:
        data = {}
    return data, name


def plot_board(width, height, blocks, i, show_plot=False, show_axis=False):
    cmap = plt.cm.get_cmap('jet', len(blocks))
    fig, ax = plt.subplots(figsize=(10, 10))
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'
        #if rotation is not None:
        #    label += f', R={1 if rotation[component] else 0}'
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('length', fontsize=15)
    # ax.legend()
    ax.set_title(f'Instance {i}, size (WxH): {width}x{height}', fontsize=22)
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'../../MIP/out/fig-ins-{i}.png')
    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)


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


# Try to solve instance before timeout
def start_solving(instance, runtimes, index, args):
    print("-" * 20)
    print(f'Solving Instance {index}')

    if args.rotation:
        model = mip_rotation
    else:
        model = mip_base

    p = multiprocessing.Process(target=model, args=(instance, index, args))

    p.start()
    start_time = time.time()
    p.join(args.timeout)

    if p.is_alive():
        print("Reached timeout without finding a solution")
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
    if not os.path.exists(f'../../MIP/out'):
        os.mkdir(f'../../MIP/out')

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
