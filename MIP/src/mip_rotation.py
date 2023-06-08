# imports:
import os
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# creates and plots the colour map with rectangles:
def plot_board(width, height, rotation, blocks, instance, show_plot=False, show_axis=False, verbose=False):
    # define pyplot colour map of len(blocks) number of colours:
    cmap = plt.cm.get_cmap('jet', len(blocks))

    # define figure size:
    fig, ax = plt.subplots(figsize=(10, 10))

    # add each rectangle block in the colour map:
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'

        if rotation is not None:
            label += f', R={1 if rotation[component] else 0}'

        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))

    # set plot properties:
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('length', fontsize=15)
    ax.legend()
    ax.set_title(f'Instance {instance}, size (WxH): {width}x{height}', fontsize=22)

    # print axis if wanted:
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    # save colormap in .png format at given path:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    figure_path = os.path.join(project_folder, 'MIP', 'out', 'rotation', 'images', f'fig-ins-{instance}.png')
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


def mip_rotation(instance, index, args):
    print("Solving MIP-Rotation")
    # initialize instance parameters:
    n = instance['n']
    w = instance['w']
    inputx = instance['inputx']
    inputy = instance['inputy']

    # height bounds:
    minh = instance['minh']
    maxh = instance['maxh']
    # minimum dimension between all circuits
    min_Dim = min(inputx + inputy)

    # solver:
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("vlsi", env=env)
        m.setParam("Symmetry", -1)

    # variables definition:
    xhat = m.addVars(n, lb=0, ub=w - min_Dim, vtype=GRB.INTEGER, name="xhat")
    yhat = m.addVars(n, lb=0, ub=maxh - min_Dim, vtype=GRB.INTEGER, name="yhat")
    h = m.addVar(lb=minh, ub=maxh, vtype=GRB.INTEGER, name='h')
    # rotation vector
    rot = m.addVars(n, vtype=GRB.BINARY, name="rot")

    x = m.addVars(n, vtype=GRB.INTEGER)
    y = m.addVars(n, vtype=GRB.INTEGER)
    m.update()
    # constraints:

    # Circuit placement bound constraints
    for i in range(n):
        # set widths and heights according to rotation
        m.addConstr(x[i] == inputx[i] if rot[i] else inputy[i])
        m.addConstr(y[i] == inputy[i] if rot[i] else inputx[i])
        m.update()
        # main constraint
        m.addConstr(yhat[i] + y[i] <= h)
        m.addConstr(xhat[i] + x[i] <= w)

    # pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Non overlapping constraints

            indicators = m.addVars(4, vtype=gp.GRB.BINARY)
            or_out = m.addVar(vtype=gp.GRB.BINARY)
            m.update()
            m.addGenConstrIndicator(indicators[0], True, xhat[i] + x[i] <= xhat[j])
            m.addGenConstrIndicator(indicators[1], True, yhat[i] + y[i] <= yhat[j])
            m.addGenConstrIndicator(indicators[2], True, xhat[j] + x[j] <= xhat[i])
            m.addGenConstrIndicator(indicators[3], True, yhat[j] + y[j] <= yhat[i])

            # OR

            m.addGenConstrOr(or_out, indicators)
            m.addConstr(or_out == 1)

    # symmetry breaking constraints:
    if args.symmetry_breaking:
        print('sym_break')
        circuits_area = [inputx[i] * inputy[i] for i in range(n)]
        first_max = np.argsort(circuits_area)[-1]
        second_max = np.argsort(circuits_area)[-2]

        # biggest in the first quadrant
        m.addConstr(xhat[first_max] <= w / 2)
        m.addConstr(yhat[first_max] <= minh / 2)

        # biggest before second biggest
        temp = m.addVar(vtype=GRB.BINARY, name="temp")
        dif = m.addVar(name="dif")  # difference y[first_max] - y[second_max]
        difabs = m.addVar(lb=0, name="difabs")  # absolute difference
        eps = 1.0e-3

        m.addConstr(xhat[first_max] <= xhat[second_max])
        m.update()

        m.addConstr(dif == xhat[first_max] - xhat[second_max])
        m.addConstr(difabs == abs_(dif))
        m.addConstr((temp == 1) >> (difabs == 0))
        m.addConstr((temp == 0) >> (difabs >= eps))
        m.update()
        m.addConstr((temp == 1) >> (yhat[first_max] + eps <= yhat[second_max]))

    # object function: minimize h
    m.setObjective(h, gp.GRB.MINIMIZE)

    m.optimize()

    if m.Status == gp.GRB.OPTIMAL:
        print(f'Found optimal solution')

        x_sol = []
        y_sol = []
        rotation_c_sol = []
        # extracting values of optimal solution:
        for i in range(n):
            x_sol.append(int(m.getVarByName(f"xhat[{i}]").X))
            y_sol.append(int(m.getVarByName(f"yhat[{i}]").X))
            rotation_c_sol.append(int(m.getVarByName(f"rot[{i}]").X))
        h = int(h.X)

        # updating the instance dictionary:
        instance['h'] = h
        instance['xhat'] = x_sol
        instance['yhat'] = y_sol
        instance['rotation'] = rotation_c_sol

        x = [int(x[e].X) for e in range(len(x))]
        y = [int(y[e].X) for e in range(len(y))]

        # prints:
        print(f'x = {x_sol}')
        print(f'y = {y_sol}')
        print(f'h = {h}')
        print(rotation_c_sol)


        # generate output string:
        out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
        out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                          for xi, yi, xhati, yhati in zip(x, y, instance['xhat'], instance['yhat'])])

        # save output string in .txt format at given path:
        project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
        text_path = os.path.join(project_folder, 'MIP', 'out', 'rotation', 'texts', f'out-{index}.txt')

        with open(text_path, 'w') as f:
            f.write(out)

        # creating a visualization of the solution and saving it to a file
        res = [(xi, yi, xhati, yhati)
               for xi, yi, xhati, yhati in zip(x, y, instance['xhat'], instance['yhat'])]

        plot_board(instance['w'], instance['h'], rotation_c_sol, res, index)

    else:
        print('No optimal solution found')
