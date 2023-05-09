import os
import numpy as np
from z3 import *
import time
from glob import glob





def z3_max(vector):
    maximum = vector[0]
    for value in vector[1:]:
        maximum = If(value > maximum, value, maximum)
    return maximum 

def z3_min(vars):
    min = vars[0]
    for v in vars[1:]:
        min = If(v < min, v, min)
    return min


def z3_cumulative(start, duration, resources, total):

    cumulative = []
    for u in resources:
        cumulative.append(
            sum([If(And(start[i] <= u, u < start[i] + duration[i]), resources[i], 0)
                 for i in range(len(start))]) <= total
        )
    return cumulative

def solve_instance(instance, index, args):

   # instance_name = in_file.split('/')[-1]
   # instance_name = instance_name[:len(instance_name) - 4]
   # out_file = os.path.join(out_dir, instance_name + '-out.txt')
    n_circuits = instance['n']
    width = instance['w']
    dx = instance['inputx']
    dy = instance['inputy']
   
    minh = instance['minh']
    maxh = sum(dy)

 

    # Coordinates of the circuits
    x = IntVector('x',n_circuits)  
    y = IntVector('y',n_circuits)

    # Maximum plate height to minimize
    height = z3_max([y[i] + dy[i] for i in range(n_circuits)])

    # Setting the optimizer
    opt = Optimize()
    

    # Setting domain and no overlap constraints
    domain_x = []
    domain_y = []
    no_overlap = []

    domain_height =[]
    domain_height.append(height<= maxh)
    domain_height.append(height>= minh)

    for i in range(n_circuits):
        domain_x.append(x[i] >= 0)
        domain_x.append(x[i] + dx[i] <= width)
        domain_x.append(x[i]<=width - z3_min(dx))
        


        domain_y.append(y[i]>=0)
        domain_y.append(y[i] + dy[i] <= height)
        domain_y.append(y[i]<= maxh - z3_min(dy))

        
        
        for j in range(i+1, n_circuits):
            no_overlap.append(Or(x[i]+dx[i] <= x[j], x[j]+dx[j] <= x[i], y[i]+dy[i] <= y[j], y[j]+dy[j] <= y[i]))

    opt.add(domain_height + domain_x + domain_y + no_overlap)

    # Cumulative constraints
    cumulative_y = z3_cumulative(y, dy, dx, width)
    #cumulative_x = z3_cumulative(x, dx, dy, sum(dy))

    opt.add( cumulative_y)

    # Boundaries constraints
    #max_width = [z3_max([x[i] + dx[i] for i in range(n_circuits)]) <= width]
    max_height = [z3_max([y[i] + dy[i] for i in range(n_circuits)]) <= maxh]
    

    opt.add(max_height)
    # Obj function
    opt.minimize(height)

    # Maximum time of execution
    opt.set("timeout", 300000)

    x_sol = []
    y_sol = []

    # Solve

    #print(f'{out_file}:', end='\t', flush=True)
    start_time = time.time()

    if opt.check() == sat:
        model = opt.model()
        elapsed_time = time.time() - start_time
        print(f'{elapsed_time * 1000:.1f} ms')
        # Getting values of variables
        for i in range(n_circuits):
            x_sol.append(model.evaluate(x[i]).as_string())
            y_sol.append(model.evaluate(y[i]).as_string())
        height_sol = model.evaluate(height).as_string()

        # Storing the result
        print("result:    ", x_sol,"\n" ,y_sol,"\n", height_sol, elapsed_time)
    
    else:
        elapsed_time = time.time() - start_time
        print(f'{elapsed_time * 1000:.1f} ms')
        print("Solution not found")