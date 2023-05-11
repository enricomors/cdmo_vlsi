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
    plt.savefig(f'../../MIP/out/base/images/fig-ins-{i}.png')
    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)


def mip_base(instance, index, args):

    # initialize instance parameters
    n = instance['n']
    w = instance['w']

    x = instance['inputx']
    y = instance['inputy']

    # height bounds
    minh = instance['minh']
    maxh = instance['maxh']

    # Solver
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        m = gp.Model('vlsi', env=env)

    # Variables definition
    xhat = [m.addVar(vtype=GRB.INTEGER, name=(f'xhat_{i}'), lb=0) for i in range(n)]
    yhat = [m.addVar(vtype=GRB.INTEGER, name=(f'yhat_{i}'), lb=0) for i in range(n)]
    h = m.addVar(vtype=GRB.INTEGER, name='h')

    # Constraints

    # Height bounds
    m.addConstr(minh <= h)
    m.addConstr(h <= maxh)

   # Circuit placement bound constraints
    for i in range(n):
        m.addConstr(xhat[i] + x[i] <= w)
        m.addConstr(yhat[i] + y[i] <= h)

        # TODO: var Hints

        m.addConstr(yhat[i] <= maxh - y[i])

    # Non overlappig constraints
    for i in range(n):
        for j in range(i + 1, n):

            big_m = m.addVars(4, vtype=GRB.BINARY)

            m.addGenConstrIndicator(big_m[0], True, xhat[i] + x[i] <= xhat[j])
            m.addGenConstrIndicator(big_m[1], True, yhat[i] + y[i] <= yhat[j])
            m.addGenConstrIndicator(big_m[2], True, xhat[j] + x[j] <= xhat[i])
            m.addGenConstrIndicator(big_m[3], True, yhat[j] + y[j] <= yhat[i])

            # Or on indicators
            or_out = m.addVar(vtype=GRB.BINARY)
            m.addGenConstrOr(or_out, big_m)
            m.addConstr(or_out == 1)

    # Symmetry breaking constraints
    if args.symmetry_breaking:
        pass

    m.update()
    m.setObjective(h, GRB.MINIMIZE)

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        print(f'Found optimal solution')

        xhat = [int(var.X) for var in xhat]
        yhat = [int(var.X) for var in yhat]
        h = int(h.X)

        instance['h'] = h
        instance['xhat'] = xhat
        instance['yhat'] = yhat

        print(f'x = {xhat}')
        print(f'y = {yhat}')
        print(f'h = {h}')

        # instance['rotation'] = rotations

        out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
        out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                          for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'],
                                                          instance['xhat'], instance['yhat'])])

        with open(f'../../MIP/out/base/texts/out-{index}.txt', 'w') as f:
            f.write(out)

        res = [(xi, yi, xhati, yhati)
               for xi, yi, xhati, yhati in zip(instance['inputx'], instance['inputy'], instance['xhat'], instance['yhat'])]
        plot_board(instance['w'], instance['h'], res, index)

    else:
        print('No optimal solution found')