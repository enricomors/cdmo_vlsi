import json

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from z3 import *
import time
import os
import numpy as np


def get_heights(heights_folder, args):
    # define path and name of runtime file:
    file_name = f'SAT' \
                f'{"-sb" if args.symmetry_breaking else ""}' \
                f'{"-rot" if args.rotation else ""}' \
                f'.json'

    file_path = os.path.join(heights_folder, file_name)

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


# creates and plots the colour map with rectangles:
def plot_board(width, height, blocks, instance, show_plot=False, show_axis=False, verbose=False):
    # define pyplot colour map of len(blocks) number of colours:
    cmap = plt.cm.get_cmap('jet', len(blocks))

    # define figure size:
    fig, ax = plt.subplots(figsize=(10, 10))

    # add each rectangle block in the colour map:
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'
        # if rotation is not None:
        #    label += f', R={1 if rotation[component] else 0}'
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))

    # set plot properties:
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('length', fontsize=15)
    # ax.legend()
    ax.set_title(f'Instance {instance}, size (WxH): {width}x{height}', fontsize=22)

    # print axis if wanted:
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    # save colormap in .png format at given path:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    figure_path = os.path.join(project_folder, "SAT", "out", "rotation", "images", f"ins-{instance}.png")
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


def order_enc_rot(instance, index, args):

    # initialize instance parameters:
    n = instance['n']
    w = instance['w']
    x = instance['inputx']
    y = instance['inputy']
    minh = instance['minh']
    maxh = instance['maxh']

    # initialize solver:
    s = Solver()

    # VARIABLES

    px = [[Bool(f'px_{i + 1}_{e}') for e in range(w)] for i in range(n)]
    py = [[Bool(f"py_{j + 1}_{f}") for f in range(maxh)] for j in range(n)]

    lr = [[Bool(f"lr_{i + 1}_{j + 1}") if i != j else 0 for j in range(n)] for i in range(n)]
    ud = [[Bool(f"ud_{i + 1}_{j + 1}") if i != j else 0 for j in range(n)] for i in range(n)]

    # ph_o is true if all rectangles are packed under height o:
    ph = [Bool(f"ph_{o}") for o in range(maxh + 1)]

    # Rotation variable:
    R = [Bool(f"R_{i + 1}") for i in range(n)]

    # CONSTRAINTS

    def add_3l_clause(direction, i, j):
        if direction == 'x':
            rectangle_measure = x[i]
            other_measure = y[i]
            strip_measure = w
            lrud = lr
            pxy = px
        elif direction == 'y':
            rectangle_measure = y[i]
            other_measure = x[i]
            strip_measure = maxh
            lrud = ud
            pxy = py
        else:
            print("The direction must be either 'x' or 'y'")
            return

        # do not allow rotation if rectangle height (width) is larger than strip width (height)
        if other_measure > strip_measure:
            s.add(Not(R[i]))

        # force rotation if rectangle width (height) is larger than strip width (height)
        if rectangle_measure > strip_measure:
            s.add(R[i])

        # if rectangle 1 is left of rectangle 2, rectangle 2 cannot be at the left of the right edge of rectangle 1.
        # no rotation
        for k in range(min(rectangle_measure, strip_measure)):
            s.add(Implies(Not(R[i]),
                          Or(Not(lrud[i][j]), Not(pxy[j][k]))))
        for k in range(strip_measure - rectangle_measure):
            k1 = k + rectangle_measure
            s.add(Implies(Not(R[i]),
                          Or(Not(lrud[i][j]), pxy[i][k], Not(pxy[j][k1]))))

        # rotation
        for k in range(min(other_measure, strip_measure)):
            s.add(Implies(R[i],
                          Or(Not(lrud[i][j]), Not(pxy[j][k]))))
        for k in range(strip_measure - other_measure):
            k1 = k + other_measure
            s.add(Implies(R[i],
                          Or(Not(lrud[i][j]), pxy[i][k], Not(pxy[j][k1]))))

    def domain_reducing_constraints():
        for i in range(n):
            # No rotation
            for e in range(w - x[i], w):
                s.add(Implies(Not(R[i]), px[i][e]))
            for f in range(maxh - y[i], maxh):
                s.add(Implies(Not(R[i]), py[i][f]))
            # rotation
            for e in range(w - y[i], w):
                s.add(Implies(R[i], px[i][e]))
            for f in range(maxh - x[i], maxh):
                s.add(Implies(R[i], py[i][f]))

        # largest rectangle:
        if args.symmetry_breaking:
            # no rotation
            for e in range((w - x[1]) // 2, w - x[1]):
                s.add(Implies(Not(R[1]), px[1][e]))
            for f in range((maxh - y[1]) // 2, maxh - y[1]):
                s.add(Implies(Not(R[1]), py[1][f]))
            # rotation
            for e in range((w - y[1]) // 2, w - y[1]):
                s.add(Implies(R[1], px[1][e]))
            for f in range((maxh - x[1]) // 2, maxh - x[1]):
                s.add(Implies(R[1], py[1][f]))

    def ordering_constraints():
        for i in range(n):
            for e in range(w - 1):
                s.add(Implies(px[i][e], px[i][e + 1]))

            for f in range(maxh - 1):
                s.add(Implies(py[i][f], py[i][f + 1]))

        for o in range(maxh - 1):
            s.add(Or(Not(ph[o]), ph[o + 1]))

    def under_height_packing_constraints():
        for o in range(maxh):
            for i in range(n):
                s.add(Implies(Not(R[i]), Or(Not(ph[o]), py[i][o - y[i]])))
                s.add(Implies(R[i], Or(Not(ph[o]), py[i][o - x[i]])))

    def add_non_overlapping_constraints(i, j, to_add=[True, True, True, True]):
        literals_4l = []
        if to_add[0]:
            literals_4l.append(lr[i][j])
            add_3l_clause('x', i, j)
        if to_add[1]:
            literals_4l.append(lr[j][i])
            add_3l_clause('x', j, i)
        if to_add[2]:
            literals_4l.append(ud[i][j])
            add_3l_clause('y', i, j)
        if to_add[3]:
            literals_4l.append(ud[j][i])
            add_3l_clause('y', j, i)

        s.add(Or(literals_4l))

    def non_overlapping_constraints():
        for j in range(n):
            for i in range(j):
                add_non_overlapping_constraints(i, j)

    def non_overlapping_constraints_sb():
        for j in range(n):
            for i in range(j):
                # LS: Reducing the domain for the largest rectangle
                if j == 1:
                    large_width = x[i] > (w - x[1]) // 2
                    large_height = y[i] > (maxh - y[1]) // 2
                    if large_width and large_height:
                        add_non_overlapping_constraints(i, j, [False, True, False, True])
                    elif large_width:
                        add_non_overlapping_constraints(i, j, [False, True, True, True])
                    elif large_height:
                        add_non_overlapping_constraints(i, j, [True, True, False, True])
                    else:
                        add_non_overlapping_constraints(i, j)
                # SR: Breaking symmetries for same-sized rectangles
                elif x[i] == x[j] and y[i] == y[j]:
                    add_non_overlapping_constraints(i, j, [True, False, True, True])
                    s.add(Or(Not(ud[i][j], lr[j][i])))
                # LR (horizontal)
                elif x[i] + x[j] > w:
                    add_non_overlapping_constraints(i, j, [False, False, True, True])
                # LR (vertical)
                elif y[i] + y[j] > maxh:
                    add_non_overlapping_constraints(i, j, [True, True, False, False])
                else:
                    add_non_overlapping_constraints(i, j)

    def add_constraints():
        domain_reducing_constraints()
        ordering_constraints()
        under_height_packing_constraints()

        if args.symmetry_breaking:
            non_overlapping_constraints_sb()
        else:
            non_overlapping_constraints()

    #  convert SAT boolean variables into cartesian coordinates:
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

    def check_rotation(model, R):
        rotations = [False for _ in range(n)]
        # for all variables R_i in R
        for i, rot in zip(range(n + 1), R):
            # if model for R_i evaluates to true
            if model.evaluate(rot, model_completion=True):
                # invert x and y dimensions for rectangle i + 1
                # print(f'Rectangle {i} is rotated')
                true_x, true_y = y[i], x[i]
                x[i], y[i] = true_x, true_y
                # print(f'Now rectangle {i} is w = {x[i]}, h = {y[i]}')
        return x, y

    # FIRST TEST FOR SATISFIABILITY

    add_constraints()

    s.push()
    s.add(ph[minh])
    tries = 0

    # Check satisfiability for given h
    while s.check() == unsat:
        tries += 1
        s.pop()
        s.push()

        s.add(ph[minh + tries])

    # Get model
    m = s.model()

    # extract values of optimal solution:
    xhat, yhat = bool_to_coords(m, px, py)
    # check for rotated circuits
    x, y = check_rotation(m, R)
    h_sol = np.max([yhat[i] + y[i] for i in range(len(yhat))])  # Compute height

    # prints:
    print(f'x = {xhat}')
    print(f'y = {yhat}')
    print(f'h = {h_sol}')
    print('Found optimal solution')

    # updating the instance dictionary:
    instance['h'] = h_sol
    instance['xhat'] = xhat
    instance['yhat'] = yhat
    # takes rotations into account
    instance['inputx'] = x
    instance['inputy'] = y


    # generate output string:
    out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
    out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                      for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'],
                                                      instance['xhat'], instance['yhat'])])

    # save output string in .txt format at given path:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    text_path = os.path.join(project_folder, 'SAT', 'out', 'rotation', 'texts', f'out-{index}.txt')
    with open(text_path, 'w') as f:
        f.write(out)

    # save height
    heights_folder = os.path.join(project_folder, 'heights')
    heights, heights_filepath = get_heights(heights_folder, args)
    heights[index] = int(instance['h'])
    with open(heights_filepath, 'w') as f:
        json.dump(heights, f)

    # creating a visualization of the solution and saving it to a file:
    res = [(xi, yi, xhati, yhati)
           for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'], instance['xhat'], instance['yhat'])]
    plot_board(instance['w'], instance['h'], res, index)

    return instance
