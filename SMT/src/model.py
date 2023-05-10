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

def solve_instance(instance, index, args):
    def plot_board(width, height, blocks, index, show_plot=False, show_axis=False):
        cmap = plt.cm.get_cmap('jet', len(blocks))
        fig, ax = plt.subplots(figsize=(10, 10))
        for component, (w, h, x, y) in enumerate(blocks):
            label = f'{w}x{h}, ({x},{y})'
            '''
            if rotation is not None:
               label += f', R={1 if rotation[component] else 0}'
               '''
            ax.add_patch(
                Rectangle((x.as_long(), y.as_long()), w, h, facecolor=cmap(component), edgecolor='k', label=label, lw=2, alpha=0.8))
        ax.set_ylim(0, height)
        ax.set_xlim(0, width)
        ax.set_xlabel('width', fontsize=15)
        ax.set_ylabel('length', fontsize=15)
        # ax.legend()
        ax.set_title(f'Instance {index}, size (WxH): {width}x{height}', fontsize=22)
        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(os.path.join(project_folder, "out", "images", f"fig-ins-{index}.png"))
        if show_plot:
            plt.show(block=False)
            plt.pause(1)
        plt.close(fig)
    
    def max_z3(vars):
        maximum = vars[0]
        for v in vars[1:]:
            maximum = If(v > maximum, v, maximum)
        return maximum 

    def min_z3(vars):
        min = vars[0]
        for v in vars[1:]:
            min = If(v < min, v, min)
        return min

    def cumulative_z3(start, duration, resources, total):
        c = []
        for u in resources:
            c.append(sum([If(And(start[i] <= u, u < start[i] + duration[i]), resources[i], 0)
                          for i in range(len(start))]) <= total)
        return c

    # initialize instance parameters
    n = instance['n']
    w = instance['w']
    widths = instance['inputx']
    heights = instance['inputy']
    minh = instance['minh']
    # maxh = instance['maxh']

  
    # variables
    x = IntVector('x', n)
    y = IntVector('y', n)


    # plate_height = max_z3([y[i] + heights[i] for i in range(n)])
    plate_height = Int('plate_height' ) #z3
    
    min_height = minh
    max_height = sum(heights)

    big_circuit_idx = np.argmax([widths[i]*heights[i]
                                for i in range(n)])

    opt = Optimize()

    # Handle rotation
   # if args.rotation:
    #    rotations = BoolVector("rotations", n)
     #   copy_widths = widths
      ##  widths = [If(rotations[i], heights[i], widths[i])
       #           for i in range(n)]
   #     heights = [If(rotations[i], copy_widths[i], heights[i])
    #               for i in range(n)]
     #   big_circuit_idx = 0







    # Bounds on variables
    # Plate height bounds
    opt.add(And(plate_height >= min_height, plate_height <= max_height))

    for i in range(n):

        # X and Y bounds
        opt.add(And(x[i] >= 0, x[i] <= w - min_z3(widths)))
        opt.add(And(y[i] >= 0, y[i] <= max_height - min_z3(heights)))

        # Main constraints
        opt.add(x[i] <= w-widths[i])
        opt.add(y[i] <= plate_height-heights[i])

        #if rotation:
            # Constraint to avoid rotation of square circuits
         #   opt.add(Implies(widths[i] == heights[i], Not(rotations[i])))

        for j in range(i+1,n):

            # Non overlaping constraints
            opt.add(Or(x[i] + widths[i] <= x[j],
                       x[j] + widths[j] <= x[i],
                       y[i] + heights[i] <= y[j],
                       y[j] + heights[j] <= y[i]))
                    

            # Ordering constraint between equal circuits
            opt.add(Implies(And(heights[i] == heights[j], widths[i] == widths[j]),
                           And(x[i] <= x[j],Implies(x[i]==x[j], y[i] < y[j]))))

    # Adding comulative constraints
    #opt.add(cumulative_z3(x, widths, heights, plate_height))
    opt.add(cumulative_z3(y, heights, widths, w))

    # Largest circuit in the origin
    opt.add(And(x[big_circuit_idx] == 0, y[big_circuit_idx] == 0))

    #SYM BREAK
    if True:#args.symmetry_breaking:
        # find the indexes of the 2 largest pieces
        circuits_area = [widths[i] * heights[i] for i in range(n)]

        first_max = np.argsort(circuits_area)[-1]
        second_max = np.argsort(circuits_area)[-2]
                
        # the biggest circuit is always placed for first w.r.t. the second biggest one
        sb_biggest_lex_less = Or(x[first_max] < x[second_max],
                                And(x[first_max] == x[second_max], y[first_max] <= y[second_max])
                                )
                
        # width maggiore -> coord y < h/2
        # height maggiore -> coord x < w/2
        sb_biggest_in_first_quadrande = And(x[first_max] < w/2, y[first_max] < plate_height/2)

        # add constraint
        opt.add(sb_biggest_in_first_quadrande)
        opt.add(sb_biggest_lex_less)





    opt.minimize(plate_height)

    opt.set("timeout", 300*1000)  # 5 minutes timeout
    start_time = time.time()

    if opt.check() == sat:
        model = opt.model()
        print("Solving time (s): ", round(time.time()-start_time, 3))
        height = model.eval(plate_height).as_long()
        print("Height:", height)
        print("\n")
        xs, ys = [], []
        for i in range(n):
            xs.append(model.evaluate(x[i]))
            ys.append(model.evaluate(y[i]))
        print("x:",xs)
        print("y:",ys)

        instance['h'] = height
        instance['xsol'] = xs
        instance['ysol'] = ys
        # output file
        out = f"{instance['w']} {instance['h']}\n{instance['n']}\n"
        out += '\n'.join([f"{xi} {yi} {xhati} {yhati}"
                          for xi, yi, xhati, yhati in zip(instance['inputx'],
                                                          instance['inputy'],
                                                          instance['xsol'], instance['ysol'])])

        project_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))

        s = 'out'\
            f'{index}'\
            '.txt'
        pathname = os.path.join(project_folder, 'out', 'texts', s)

        with open(pathname, 'w') as f:
            f.write(out)

        res = [(xi, yi, xhati, yhati)
               for xi, yi, xhati, yhati in
               zip(instance['inputx'], instance['inputy'], instance['xsol'], instance['ysol'])]
        plot_board(instance['w'], instance['h'], res, index)
        return xs, ys, height, time.time()-start_time
    else:
        print("Time limit excedeed\n")
        return [], [], None, 0