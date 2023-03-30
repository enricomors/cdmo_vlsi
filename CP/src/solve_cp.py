import os
import json
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from minizinc import Instance, Model, Solver
from minizinc.result import Status


def plot_board(width, height, blocks, i, rotated, show_plot=False, show_axis=False):
    cmap = plt.cm.get_cmap('jet', len(blocks))
    fig, ax = plt.subplots(figsize=(10, 10))
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'
        if rotated is not None:
            label += f', R={1 if rotated[component] else 0}'
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('height', fontsize=15)
    ax.legend()
    ax.set_title(f'Instance {i}, size (WxH): {width}x{height}', fontsize=22)
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'../../CP/out/fig-ins-{i}.png')
    if show_plot:
        plt.show(block=False)
        plt.pause(1)
    plt.close(fig)


def get_runtimes():
    if not os.path.exists('../../runtimes'):
        os.mkdir('../../runtimes')
    name = f'../../runtimes/CP-{args.solver}' \
           f'{"-sb" if args.symmetry_breaking else ""}' \
           f'{"-rot" if args.rotation else ""}' \
           f'-heu{args.heu}-restart{args.restart}.json'
    if os.path.isfile(name):
        with open(name) as f:
            data = {int(k): v for k, v in json.load(f).items()}
    else:
        data = {}
    return data, name


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('technology', type=str, help='The technology to use (CP, SAT or SMT)')
    parser.add_argument('-s', '--start', type=int, help='First instance to solve', default=1)
    parser.add_argument('-e', '--end', type=int, help='Last instance to solve', default=40)
    parser.add_argument('-t', '--timeout', type=int, help='Timeout (ms)', default=300000)
    parser.add_argument('-r', '--rotation', action="store_true", help="enables circuits rotation")
    parser.add_argument('-sb', '--symmetry_breaking', action="store_true", help="enables symmetry breaking")

    parser.add_argument('--solver', type=str, help='CP solver (default: chuffed)', default='chuffed')
    parser.add_argument('--heu', type=str, help='CP search heuristic (default: input_order, min)',
                        default='input_order')
    parser.add_argument('--restart', type=str, help='CP restart strategy (default: none)', default='none')

    args = parser.parse_args()

    # Checks for errors
    if args.solver not in ('gecode', 'chuffed'):
        raise ValueError(f'wrong solver {args.solver}; supported ones are gecode and chuffed')
    if args.heu not in ('input_order', 'first_fail', 'dom_w_deg'):
        raise ValueError(f'wrong search heuristic {args.heu}; supported ones are '
                         f'input_order, first_fail and dom_w_deg')
    if args.restart not in ('none', 'geom', 'luby'):
        raise ValueError(f'wrong restart {args.restart}; supported ones are geom and luby')

    # Instantiate Model, Solver
    mod = ''
    if args.symmetry_breaking:
        mod += '-sb'
    if args.rotation:
        mod += '-rotation'
    model = Model(f"vlsi{mod}.mzn")
    solver = Solver.lookup(f'{args.solver}')

    # Creates output folder
    if not os.path.exists(f'../out'):
        os.mkdir(f'../out')

    runtimes, runtimes_filename = get_runtimes()
    print(f'SOLVING INSTANCES {args.start} - {args.end} USING CP MODEL')

    # Solve Instances in range
    for i in range(args.start, args.end + 1):
        print('=' * 20)
        print(f'INSTANCE {i}')
        # create instance
        instance = Instance(solver, model)
        with open(f'../../instances/ins-{i}.txt') as f:
            lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        w = int(lines[0].strip('\n'))
        n = int(lines[1].strip('\n'))
        dim = [ln.split(' ') for ln in lines[2:]]
        x, y = list(zip(*map(lambda x_y: (int(x_y[0]), int(x_y[1])), dim)))
        xy = np.array([x, y]).T

        # Sort circuits by area
        areas = np.prod(xy, axis=1)
        sorted_idx = np.argsort(areas)[::-1]
        xy = xy[sorted_idx]
        x = list(map(int, xy[:, 0]))
        y = list(map(int, xy[:, 1]))

        # lower and upper bounds for height
        min_area = np.prod(xy, axis=1).sum()
        minh = int(min_area / w)
        maxh = np.sum(y)

        instance['w'] = w
        instance['n'] = n

        if args.rotation:
            instance['inx'] = x
            instance['iny'] = y
        else:
            instance['x'] = x
            instance['y'] = y

        instance['minh'] = minh
        instance['maxh'] = maxh

        instance['search'] = args.heu
        instance['restart'] = args.restart

        # Solve instance with timeout
        res = instance.solve(timeout=timedelta(milliseconds=args.timeout))

        if res.status == Status.OPTIMAL_SOLUTION:
            out = f"{instance['w']} {res.objective}\n{instance['n']}\n"
            out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                              for xi, yi, xhati, yhati in zip(res['x'] if args.rotation else instance['x'],
                                                              res['y'] if args.rotation else instance['y'],
                                                              res['xhat'], res['yhat'])])

            with open(f'../../CP/out/out-{i}.txt', 'w') as f:
                f.write(out)
            print('Instance solved')
            print(f'Solution status: {res.status}')
            print(f'h = {res.objective}')
            print(f'time: {res.statistics["time"]}')

            # Plot the resulting board
            solution = [(xi, yi, xhati, yhati)
                        for xi, yi, xhati, yhati in zip(res['x'] if args.rotation else instance['x'],
                                                        res['y'] if args.rotation else instance['y'],
                                                        res['xhat'], res['yhat'])]
            plot_board(instance['w'], res.objective, solution, i, res['rotated'] if args.rotation else None)

            # total runtime in seconds
            runtimes[i] = (res.statistics['time'].microseconds / (10 ** 6)) + res.statistics['time'].seconds

        else:
            print('Not solved within timeout')
            runtimes[i] = 300

        with open(runtimes_filename, 'w') as f:
            json.dump(runtimes, f)
