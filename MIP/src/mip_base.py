import os
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


# creates and plots the colour map with rectangles:
def plot_board(width, height, blocks, instance, show_plot=False, show_axis=False, verbose=False):
    # define pyplot colour map of len(blocks) number of colours:
    cmap = plt.cm.get_cmap('jet', len(blocks))

    # define figure size:
    fig, ax = plt.subplots(figsize=(10, 10))

    # add each rectangle block in the colour map:
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'

        # if the rectangle is rotated denote it in its label:
        # if rotation is not None:
        #    label += f', R={1 if rotation[component] else 0}'

        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))

    # set plot properties:
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('length', fontsize=15)
    ax.legend()
    ax.set_title(f'Instance {instance}, size (WxH): {width}x{height}', fontsize=22)

    # print axis if wanted:
    if show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    # save colormap in .png format at given path:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    figure_path = os.path.join(project_folder, 'MIP', 'out', 'base', 'images', f'fig-ins-{instance}.png')
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


def mip_base(instance, index, args):
    # initialize instance parameters:
    n = instance['n']  # number of components
    w = instance['w']  # board width
    x = instance['inputx']  # component width
    y = instance['inputy']  # component height

    # height bounds:
    minh = instance['minh']
    maxh = instance['maxh']

    # solver:
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model('vlsi', env=env)



    # variables definition:
    xhat = [m.addVar(vtype=GRB.INTEGER, name=f'xhat_{i}', lb=0, ub=w - min(x)) for i in range(n)]
    yhat = [m.addVar(vtype=GRB.INTEGER, name=f'yhat_{i}', lb=0, ub=maxh - min(y)) for i in range(n)]
    h = m.addVar(vtype=GRB.INTEGER, name='h', lb=minh, ub=maxh)
    m.update()
    # constraints:
    # Circuit placement bound constraints:
    for i in range(n):
        m.addConstr(xhat[i] + x[i] <= w)
        m.addConstr(yhat[i] + y[i] <= h)

    # non overlapping constraints:
    for i in range(n):
        for j in range(i + 1, n):
            big_m = m.addVars(4, vtype=GRB.BINARY)
            or_out = m.addVar(vtype=GRB.BINARY)
            m.update()
            m.addGenConstrIndicator(big_m[0], True, xhat[i] + x[i] <= xhat[j])
            m.addGenConstrIndicator(big_m[1], True, yhat[i] + y[i] <= yhat[j])
            m.addGenConstrIndicator(big_m[2], True, xhat[j] + x[j] <= xhat[i])
            m.addGenConstrIndicator(big_m[3], True, yhat[j] + y[j] <= yhat[i])

            m.addGenConstrOr(or_out, big_m)
            m.addConstr(or_out == 1)

    m.addConstr(xhat[0] == 0)
    m.addConstr(yhat[0] == 0)
    # symmetry breaking constraints:
    if args.symmetry_breaking:
        print('sym_break')
        circuits_area = [x[i] * y[i] for i in range(n)]
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

        m.addConstr(or_(xhat[first_max] <= xhat[second_max],xhat[first_max] + eps <= xhat[second_max]))
        m.update()

        m.addConstr(dif == xhat[first_max] - xhat[second_max])
        m.addConstr(difabs == abs_(dif))
        m.addConstr((temp == 1) >> (difabs == 0))
        m.addConstr((temp == 0) >> (difabs >= eps))
        m.update()
        m.addConstr((temp == 1) >> (yhat[first_max] <= yhat[second_max]))

    m.setObjective(h, GRB.MINIMIZE)

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        print(f'Found optimal solution')

        # extracting values of optimal solution:
        xhat = [int(var.X) for var in xhat]
        yhat = [int(var.X) for var in yhat]
        h = int(h.X)

        # updating the instance dictionary:
        instance['h'] = h
        instance['xhat'] = xhat
        instance['yhat'] = yhat

        # prints:
        print(f'x = {xhat}')
        print(f'y = {yhat}')
        print(f'h = {h}')
        print(f'w = {w}')

        # generate output string:
        out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
        out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                          for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'],
                                                          instance['xhat'], instance['yhat'])])

        # save output string in .txt format at given path:
        project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
        text_path = os.path.join(project_folder, 'MIP', 'out', 'base', 'texts', f'out-{index}.txt')

        with open(text_path, 'w') as f:
            f.write(out)

        # creating a visualization of the solution and saving it to a file
        res = [(xi, yi, xhati, yhati)
               for xi, yi, xhati, yhati in
               zip(instance['inputx'], instance['inputy'], instance['xhat'], instance['yhat'])]

        plot_board(instance['w'], instance['h'], res, index)

    else:
        print('No optimal solution found')
