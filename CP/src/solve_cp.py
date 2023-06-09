# imports:
import os
import json
import warnings
from argparse import ArgumentParser
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from minizinc import Instance, Model, Solver
from minizinc.result import Status


# ******* User Defined Functions *******

# create output folders if not already created:
def create_folder_structure():
    # root folders:
    project_folder = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    outputs_folder = os.path.join(project_folder, 'CP', 'out')

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

    # check if heights folder already exists:
    heights_folder = os.path.join(project_folder, 'heights')

    if not os.path.exists(heights_folder):
        os.mkdir(heights_folder)  # cdmo_vlsi/heights
        print("Heights folder has been created correctly!")

    # check if runtimes folder already exists:
    runtimes_folder = os.path.join(project_folder, 'runtimes')

    if not os.path.exists(heights_folder):
        os.mkdir(heights_folder)  # cdmo_vlsi/heights
        print("Heights folder has been created correctly!")

    # instance folder:
    instances_folder = os.path.join(project_folder, 'instances')

    return project_folder, outputs_folder, heights_folder, runtimes_folder, instances_folder


#get upper bound
def get_upperbound(heights, widths, n, w):

    # initialize variables:
    previous_row_height = 0     #to store the highest height of the previous row
    h_res = 0                   #maximum found height
    acc_width = 0               #accumulated horizontal width

    # cycle all blocks
    for i in range(n):

        # if the new inserted block does not exceed the width of the plate, then keep inserting blocks horizontally:
        if acc_width + widths[i] < w:
            # add to the value of accumulated width the width of the i-th block:
            acc_width += widths[i]

            # if the height of current block is higher than the highest height found since now, then update it:
            if heights[i] + previous_row_height > h_res:
                h_res = heights[i] + previous_row_height

        #otherwise save current maximum height and start a new line:
        else:
            #re-initialize the value of accumulated width to the width of i-th block (the first of new row)
            acc_width = widths[i]

            #set maximum height of previous row:
            previous_row_height = h_res

            #update maximum height:
            h_res = h_res + heights[i]

    return h_res


#get optimal heights of each instance:
def get_heights(args):
    # define path and name of heights file:
    file_name = f'CP-{args.solver}' \
                f'{"-sb" if args.symmetry_breaking else ""}' \
                f'{"-rot" if args.rotation else ""}' \
                f'-heu{args.heu}-restart{args.restart}.json'

    file_path = os.path.join(heights_folder, file_name)

    # if file exists load it and extract dict values, otherwise return empty dict:
    if os.path.isfile(file_path):
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


#get runtimes of each instance
def get_runtimes(args):
    # define path and name of heights file:
    file_name = f'CP-{args.solver}' \
                f'{"-sb" if args.symmetry_breaking else ""}' \
                f'{"-rot" if args.rotation else ""}' \
                f'-heu{args.heu}-restart{args.restart}.json'

    file_path = os.path.join(runtimes_folder, file_name)

    # if file exists load it and extract dict values, otherwise return empty dict:
    if os.path.isfile(file_path):
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
def plot_board(width, height, blocks, instance, rotated, show_plot=False, show_axis=False, verbose=False):

    # suppress get_cmap warning:
    warnings.filterwarnings('ignore', message="The get_cmap function was deprecated in Matplotlib 3.7")

    # define pyplot colour map of len(blocks) number of colours:
    cmap = plt.cm.get_cmap('jet', len(blocks))

    # define figure size:
    fig, ax = plt.subplots(figsize=(10, 10))

    # add each rectangle block in the colour map:
    for component, (w, h, x, y) in enumerate(blocks):
        label = f'{w}x{h}, ({x},{y})'

        # if the rectangle is rotated denote it in its label:
        if rotated is not None:
            label += f', R={1 if rotated[component] else 0}'

        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))

    # set plot properties:
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
    ax.set_xlabel('width', fontsize=15)
    ax.set_ylabel('height', fontsize=15)
    ax.legend()
    ax.set_title(f'Instance {instance}, size (WxH): {width}x{height}', fontsize=22)

    # print axis if wanted:
    if show_axis:
        ax.set_xticks([])
        ax.set_yticks([])

    # save colormap in .png format at given path (checking if using rotation or not):
    figure_path = os.path.join(outputs_folder, 'rotation' if args.rotation else 'base', "images", f"ins-{instance}.png")
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


# ******* Main Code *******
if __name__ == "__main__":

    # create folders structure:
    project_folder, outputs_folder, heights_folder, runtimes_folder, instances_folder = create_folder_structure()

    # define command line arguments:
    parser = ArgumentParser()

    parser.add_argument('-s', '--start', type=int, help='First instance to solve', default=1)
    parser.add_argument('-e', '--end', type=int, help='Last instance to solve', default=40)
    parser.add_argument('-t', '--timeout', type=int, help='Timeout (ms)', default=300000)
    parser.add_argument('-r',    '--rotation', action="store_true", help="enables circuits rotation")
    parser.add_argument('-sb', '--symmetry_breaking', action="store_true", help="enables symmetry breaking")
    parser.add_argument('--solver', type=str, help='CP solver (default: chuffed)', default='chuffed')
    parser.add_argument('--heu', type=str, help='CP search heuristic (default: input_order, min)', default='input_order')
    parser.add_argument('--restart', type=str, help='CP restart strategy (default: none)', default='none')

    args = parser.parse_args()

    # check for argument errors:
    # solver:
    if args.solver not in ('gecode', 'chuffed'):
        raise ValueError(f'wrong solver {args.solver};\nsupported ones are gecode and chuffed')

    # heuristics:
    if args.heu not in ('input_order', 'first_fail', 'dom_w_deg'):
        raise ValueError(
            f'wrong search heuristic {args.heu};\nsupported ones are input_order, first_fail and dom_w_deg')

    # restart strategies:
    if args.restart not in ('none', 'geom', 'luby'):
        raise ValueError(f'wrong restart {args.restart};\nsupported ones are geom and luby')

    # instantiate Model and Solver:
    mod = ''

    if args.symmetry_breaking:
        mod += '-sb'

    if args.rotation:
        mod += '-rotation'

    # select model based on arguments of command line prompt:
    model = Model(f"cp{mod}.mzn")

    # select solver based on arguments of command line prompt:
    solver = Solver.lookup(f'{args.solver}')

    # get heights:
    heights, heights_filepath = get_heights(args)

    # get runtimes:
    runtimes, runtimes_filepath = get_runtimes(args)

    # solve Instances in range:
    print(f'Solving instances {args.start} - {args.end} using CP model')

    # solve from start instance to end instance (DEFAULT: start = 1, end = 41):
    for i in range(args.start, args.end + 1):
        print('=' * 20)
        print(f'Instance {i}')

        # create instance of the model and define solver to use:
        instance = Instance(solver, model)

        # open instance and extract instance data:
        with open(os.path.join(instances_folder, f'ins-{i}.txt')) as f:
            lines = f.readlines()

        # get a list of lines:
        lines = [l.strip('\n') for l in lines]

        # get width of the map:
        w = int(lines[0].strip('\n'))

        # get number of blocks:
        n = int(lines[1].strip('\n'))

        # get list of the dimensions of each circuit:
        dim = [ln.split(' ') for ln in lines[2:]]

        # get x and y dimensions:
        x, y = list(zip(*map(lambda x_y: (int(x_y[0]), int(x_y[1])), dim)))


        # *** sort circuits by area ***:

        # create a matrix with horizontal and vertical dimensions of circuits:
        xy = np.array([x, y]).T

        # compute area of each block
        areas = np.prod(xy, axis=1)

        # indices to sort areas array in descending order:
        sorted_idx = np.argsort(areas)[::-1]

        # re-order the xy array using the new indeces
        xy = xy[sorted_idx]

        # re-order horizontal and vertical dimensions based on the sorted indeces:
        x = list(map(int, xy[:, 0]))
        y = list(map(int, xy[:, 1]))

        # *** define lower and upper bounds for height ***:

        # height's lower bound:
        min_area = np.prod(xy, axis=1).sum()  #sum of the areas of all blocks
        minh = int(min_area / w)

        # height's upper bound
        maxh = get_upperbound(y, x, n, w)



        # *** initialize variables based on instance *** :

        # width:
        instance['w'] = w

        # number of blocks:
        instance['n'] = n

        # if rotation is enabled, save original width and heigth
        if args.rotation:

            # original input dimensions before rotating block:
            instance['xinput'] = x
            instance['yinput'] = y

        else:
            instance['x'] = x
            instance['y'] = y

        # height's lower bound:
        instance['minh'] = minh

        # height's upper bound:
        instance['maxh'] = maxh

        # search heuristic:
        instance['search'] = args.heu

        # restart strategy:
        instance['restart'] = args.restart



        # *** solve each instance before timeout exceeds ***:

        # solve instance:
        res = instance.solve(timeout=timedelta(milliseconds=args.timeout)) #5 minutes timeout

        # check whether the instance was solved within timeout:
        if res.status == Status.OPTIMAL_SOLUTION:

            # generate output string that contains information about the execution of the instance:
            out = f"{instance['w']} {res.objective}\n{instance['n']}\n"
            out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                              for xi, yi, xhati, yhati in zip(res['x'] if args.rotation else instance['x'],
                                                              res['y'] if args.rotation else instance['y'],
                                                              res['xhat'], res['yhat'])])

            # save output string on a text file:
            with open(os.path.join(outputs_folder, 'rotation' if args.rotation else 'base', 'texts', f'out-{i}.txt'),
                      'w') as f:
                f.write(out)

            # prints:
            print('Instance solved')
            print(f'Solution status: {res.status}')
            print(f'h = {res.objective}')
            print(f'time: {res.statistics["time"]}')

            # plot the resulting board:
            solution = [(xi, yi, xhati, yhati)
                        for xi, yi, xhati, yhati in zip(res['x'] if args.rotation else instance['x'],
                                                        res['y'] if args.rotation else instance['y'],
                                                        res['xhat'], res['yhat'])]

            plot_board(instance['w'], res.objective, solution, i, res['rotated'] if args.rotation else None)

            # runtime of i-th instance converted to seconds:
            runtimes[i] = (res.statistics['time'].microseconds / (10 ** 6)) + res.statistics['time'].seconds

            # optimal height value found:
            heights[i] = res.objective

        # if the timeout:
        else:
            print('Not solved within timeout')

            # set height and runtime to default values of unsatisfied instances:
            heights[i] = 'UNSAT'
            runtimes[i] = 300

        # *** save results of each instance in a json file for both heights and runtimes: ***

        # save heights:
        with open(heights_filepath, 'w') as f:
            json.dump(heights, f)

        # save runtimes:
        with open(runtimes_filepath, 'w') as f:
            json.dump(runtimes, f)
