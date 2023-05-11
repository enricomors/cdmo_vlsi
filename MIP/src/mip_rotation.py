import gurobipy as gp
from gurobipy import GRB

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


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
    plt.savefig(f'../../MIP/out/rotation/fig-ins-{i}.png')
    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)


def mip_rotation(instance, index, args):

    # initialize instance parameters
    n = instance['n']
    w = instance['w']

    inputx = instance['inputx']
    inputy = instance['inputy']

    # height bounds
    minh = instance['minh']
    maxh = instance['maxh']

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("vlsi", env=env)

    # Variables definition
    xhat = [m.addVar(vtype=GRB.INTEGER, name=(f'xhat_{i}'), lb=0) for i in range(n)]
    yhat = [m.addVar(vtype=GRB.INTEGER, name=(f'yhat_{i}'), lb=0) for i in range(n)]
    h = m.addVar(vtype=GRB.INTEGER, name='h')

    rot = m.addVars(instance.n_circuits, vtype=GRB.BINARY)

    x = m.addVars(instance.n_circuits, vtype=GRB.INTEGER)
    y = m.addVars(instance.n_circuits, vtype=GRB.INTEGER)

    # Constraints

    m.addConstr(minh <= h)
    m.addConstr(h <= maxh)

    # Circuit placement bound constraints
    for i in range(instance.n_circuits):
        m.addConstr(yhat[i] + y[i] <= h)
        m.addConstr(xhat[i] + x[i] <= w)

        # Make use of bounds
        m.addConstr(yhat[i] <= maxh - y[i])

        # Set widths and heights according to rotation
        m.addConstr(x[i] == rot[i] * inputy[i] + (1 - rot[i]) * inputx[i])
        m.addConstr(y[i] == rot[i] * inputx[i] + (1 - rot[i]) * inputy[i])

    # Pairwise constraints
    for i in range(instance.n_circuits):
        for j in range(i + 1, instance.n_circuits):
            # Non overlapping constraints

            indicators = m.addVars(4, vtype=gp.GRB.BINARY)

            m.addGenConstrIndicator(indicators[0], True, xhat[i] + x[i] <= xhat[j])
            m.addGenConstrIndicator(indicators[1], True, yhat[i] + y[i] <= yhat[j])
            m.addGenConstrIndicator(indicators[2], True, xhat[j] + x[j] <= xhat[i])
            m.addGenConstrIndicator(indicators[3], True, yhat[j] + y[j] <= yhat[i])

            # OR
            or_out = m.addVar(vtype=gp.GRB.BINARY)
            m.addGenConstrOr(or_out, indicators)
            m.addConstr(or_out == 1)

            # Symmetry breaking constraints
            if args.symmetry_breaking:
                pass

    m.update()
    m.setObjective(h, gp.GRB.MINIMIZE)

    m.optimize()

    if m.Status == gp.GRB.OPTIMAL:
        print(f'Found optimal solution')

        xhat = [int(var.X) for var in xhat]
        yhat = [int(var.X) for var in yhat]
        h = int(h.X)

        instance['h'] = h
        instance['xhat'] = xhat
        instance['yhat'] = yhat

        x = [int(var.X) for var in x]
        y = [int(var.X) for var in y]

        print(f'x = {xhat}')
        print(f'y = {yhat}')
        print(f'h = {h}')

        instance['rotation'] = rot
        out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
        out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                          for xi, yi, xhati, yhati in zip(x, y, instance['xhat'], instance['yhat'])])

        with open(f'../../MIP/out/rotation/texts/out-{index}.txt', 'w') as f:
            f.write(out)

        res = [(xi, yi, xhati, yhati)
               for xi, yi, xhati, yhati in
               zip(x, y, instance['xhat'], instance['yhat'])]
        plot_board(instance['w'], instance['h'], res, index)

    else:
        print('No optimal solution found')
