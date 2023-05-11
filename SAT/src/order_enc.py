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

project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

def plot_board(width, height, blocks, index, show_plot=False, show_axis=False):
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
    ax.set_title(f'Instance {index}, size (WxH): {width}x{height}', fontsize=22)
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])


    figure_folder = os.path.join(project_folder, "SAT", "out", "base", "images", f"ins-{index}.png")
    plt.savefig(figure_folder)
    print(f"figure ins-{index}.png has been correctly saved at path '{figure_folder}'")

    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)


def order_enc(instance, index, args):

    # initialize instance parameters
    n = instance['n']
    w = instance['w']
    x = instance['inputx']
    y = instance['inputy']
    minh = instance['minh']
    maxh = instance['maxh']

    s = Solver()

    # Variables definition
    px = [[Bool(f'px_{i + 1}_{e}') for e in range(w)] for i in range(n)]
    py = [[Bool(f"py_{j + 1}_{f}") for f in range(maxh)] for j in range(n)]

    lr = [[Bool(f"lr_{i + 1}_{j + 1}") if i != j else 0 for j in range(n)] for i in range(n)]
    ud = [[Bool(f"ud_{i + 1}_{j + 1}") if i != j else 0 for j in range(n)] for i in range(n)]

    # ph_o is true if all rectangles are packed under height o
    ph = [Bool(f"ph_{o}") for o in range(maxh + 1)]

    # Constraints

    # Order encoding constraints
    for i in range(n):
        for e in range(w - x[i]):
            s.add(Or(Not(px[i][e]), px[i][e + 1]))
        for f in range(maxh - y[i]):
            s.add(Or(Not(py[i][f]), py[i][f + 1]))

    for o in range(maxh - 1):
        s.add(Or(Not(ph[o]), ph[o + 1]))

    # Under-height packing constraint
    for o in range(maxh):
        for i in range(n):
            s.add(Or(Not(ph[o]), py[i][o - y[i]]))

    # Non-Overlapping constraints
    for i in range(n):
        for j in range(i + 1, n):
            if i < j:
                s.add(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

    def no_overlap_x(i, j):
        res = []
        res.append([Not(px[j][x[i] - 1])])
        for e in range(w - x[i] - 1):
            res.append([px[i][e], Not(px[j][e + x[i]])])
        res.append([px[i][w - x[i] - 1]])
        return res

    def no_overlap_y(i, j):
        res = []
        res.append([Not(py[j][y[i] - 1])])
        for f in range(maxh - y[i] - 1):
            res.append([py[i][f], Not(py[j][f + y[i]])])
        res.append([py[i][maxh - y[i] - 1]])
        return res

    # Add the 3-literal clauses for non-overlapping constraints
    # i.e. ¬lr[i][j] \/ ¬px[j][e + w_i] \/ px[i][e]
    for i in range(n):
        for j in range(i + 1, n):

            for pr in no_overlap_x(i, j):
                prop = [Not(lr[i][j])] + pr
                s.add(Or(prop))

            for pr in no_overlap_x(j, i):
                prop = [Not(lr[j][i])] + pr
                s.add(Or(prop))

            for pr in no_overlap_y(i, j):
                prop = [Not(ud[i][j])] + pr
                s.add(Or(prop))

            for pr in no_overlap_y(j, i):
                prop = [Not(ud[j][i])] + pr
                s.add(Or(prop))

    # domain reducing constraints of px and py
    for i in range(n):
        for e in range(w - x[i], w):
            figure_folder = os.path.join(project_folder, "SAT", "out","base", "images", f"ins-{i}.png")

    plt.savefig(figure_folder)

    print(f"figure ins-{instance}.png has been correctly saved at path '{figure_folder}'")

    s.add(px[i][e])
    for f in range(maxh - y[i], maxh):
        s.add(py[i][f])

    # Symmetry breaking constraints
    if args.symmetry_breaking:
        # Ordering for same size rectangles
        for i in range(n):
            for j in range(i + 1, n):
                if x[i] == x[j] and y[i] == y[j]:
                    s.add(Not(lr[j][i]))
                    s.add(Or(lr[i][j], Not(ud[j][i])))

        # large rectangles constraint
        for i in range(n):
            for j in range(i + 1, n):
                if x[i] + x[j] > w:
                    s.add(And(Not(lr[i][j]), Not(lr[j][i])))
                if y[i] + y[j] > maxh:
                    s.add(And(Not(ud[i][j]), Not(ud[j][i])))

        # Domain reduction for largest rectangle
        for e in range((w - x[0]) // 2):
            s.add(px[0][e])
        for f in range((maxh - y[0]) // 2):
            s.add(py[0][f])

    # Converter: SAT boolean variables are translated in cartesian coordinates
    def bool_to_coords(model, px, py):
        xhat = []
        yhat = []
        for i in range(n):
            j = 0
            while j < w:
                if model[px[i][j]]:
                    xhat.append(j)
                    break
                j += 1
            j = 0
            while j < maxh:
                if model[py[i][j]]:
                    yhat.append(j)
                    break
                j += 1
        return xhat, yhat

    # First test for satisfiability

    s.push()
    s.add(ph[minh])

    tries = 0

    while s.check() == unsat:
        tries += 1
        s.pop()
        s.push()

        s.add(ph[minh + tries])

    m = s.model()
    xhat, yhat = bool_to_coords(m, px, py)
    h_sol = np.max([yhat[i] + y[i] for i in range(len(yhat))])  # Compute height

    print(f'x = {xhat}')
    print(f'y = {yhat}')
    print(f'h = {h_sol}')

    print('Found optimal solution')
    instance['h'] = h_sol
    instance['xhat'] = xhat
    instance['yhat'] = yhat

    # output file
    out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
    out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                      for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'],
                                                      instance['xhat'], instance['yhat'])])

    with open(f'../../SAT/out/base/texts/out-{index}.txt', 'w') as f:
        f.write(out)

    # plot solution
    res = [(xi, yi, xhati, yhati)
           for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'], instance['xhat'], instance['yhat'])]
    plot_board(instance['w'], instance['h'], res, index)