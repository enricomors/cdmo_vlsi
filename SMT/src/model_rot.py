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





def solve_instance_rot(instance, index, args):
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

    def real_dim(dim, dim_swap, rot, n):
        res = [If(rot[i], dim_swap[i], dim[i]) for i in range(n)]
        return res

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
    min_height = minh
    max_height = sum(heights)
    # max_height = sum([If(rotation_c[i], widths[i], heights[i]) for i in range(n)])

    # variables
    x = IntVector('x', n)
    y = IntVector('y', n)

    # array of booleans, each one telling if the corresponding chip is rotated or not
    rotation_c = BoolVector('r', n)

    # plate_height = max_z3([y[i] + heights[i] for i in range(n)])
    # plate_height = Int('plate_height' ) #z3

    # height that is going to be minimized from the optimizer

    plate_height = max_z3([If(rotation_c[i], widths[i], heights[i]) + y[i] for i in range(n)])

    big_circuit_idx = np.argmax([widths[i] * heights[i]
                                 for i in range(n)])

    # array of booleans, each one telling if the corresponding chip is rotated or not
    rotation_c = BoolVector('r', n)
    opt = Optimize()

    # Handle rotation
    '''
    if args.rotation:
        rotations = BoolVector("rotations", n)
        copy_widths = widths
        widths = [If(rotations[i], heights[i], widths[i])
                  for i in range(n)]
        heights = [If(rotations[i], copy_widths[i], heights[i])
                   for i in range(n)]
      '''

    # Bounds on variables
    # Plate height bounds
    opt.add(And(plate_height >= min_height, plate_height <= max_height))

    for i in range(n):

        # X and Y bounds
        opt.add(And(x[i] >= 0, x[i] <= w - min_z3(real_dim(widths, heights, rotation_c, n))))
        opt.add(And(y[i] >= 0, y[i] <= max_height - min_z3(real_dim(heights, widths, rotation_c, n))))

        # Main constraints
        opt.add(x[i] + If(rotation_c[i], heights[i], widths[i]) <= w)
        opt.add(y[i] + If(rotation_c[i], widths[i], heights[i]) <= plate_height)

        # Constraint to avoid rotation of square circuits
        opt.add(Implies(widths[i] == heights[i], Not(rotation_c[i])))

        # Constraint to avoid rotation of out of bound circuits
        opt.add(Implies(Or(widths[i] > max_height, heights[i] > w), Not(rotation_c[i])))

        for j in range(i + 1, n):
            # Non overlapping constraints
            opt.add(Or(x[i] + If(rotation_c[i], heights[i], widths[i]) <= x[j],
                       x[j] + If(rotation_c[j], heights[j], widths[j]) <= x[i],
                       y[i] + If(rotation_c[i], widths[i], heights[i]) <= y[j],
                       y[j] + If(rotation_c[j], widths[j], heights[j]) <= y[i]))

            # Ordering constraint between equal circuits
            opt.add(Implies(And(heights[i] == heights[j], widths[i] == widths[j]),
                            And(x[i] <= x[j], Implies(x[i] == x[j], y[i] < y[j]))))

    # Adding cumulative constraints
    # opt.add(cumulative_z3(x, widths, heights, plate_height))
    opt.add(cumulative_z3(y,
                          [If(rotation_c[i], widths[i], heights[i]) for i in range(n)],
                          [If(rotation_c[i], heights[i], widths[i]) for i in range(n)],
                          w))

    # Largest circuit in the origin
    opt.add(And(x[big_circuit_idx] == 0, y[big_circuit_idx] == 0))

    # SYM BREAK

    if True:  # args.symmetry_breaking:
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
        sb_biggest_in_first_quadrande = And(x[first_max] < w / 2, y[first_max] < plate_height / 2)

        # add constraint
        opt.add(sb_biggest_in_first_quadrande)
        opt.add(sb_biggest_lex_less)

    opt.minimize(plate_height)

    opt.set("timeout", 300 * 1000)  # 5 minutes timeout
    start_time = time.time()

    if opt.check() == sat:
        model = opt.model()
        print("Solving time (s): ", round(time.time() - start_time, 3))
        height = model.eval(plate_height).as_long()
        print("Height:", height)
        print("\n")
        xs, ys = [], []
        for i in range(n):
            xs.append(model.evaluate(x[i]))
            ys.append(model.evaluate(y[i]))
        print("x:", xs)
        print("y:", ys)
        # print("rotations vector: ", rotation_c)
        return xs, ys, height, time.time() - start_time
    else:
        print("Time limit excedeed\n")
        return [], [], None, 0
