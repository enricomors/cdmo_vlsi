# Constraint Programming

Solves the VLSI design problem by means of constraint programming

## Usage 
In order to solve the instances by means of CP you should launch the `solve_cp.py`. You can specify the following parameters

```
python main.py [-h] [-s START] [-e END] [-t TIMEOUT] [-r]
               [--solver SOLVER] [--heu HEU] [--restart RESTART]

```

Command line arguments:

| Argument                                         | Description                                                                  |
| ------------------------------------------------ |------------------------------------------------------------------------------|
| `-h, --help`                                     | Shows help message                                                           |
| `-s START, --start START`                        | First instance to solve (default: 1)                                         |
| `-e END, --end END`                              | Last instance to solve (default: 40)                                         |
| `-t TIMEOUT, --timeout TIMEOUT`                  | Sets the timeout (ms, default: 300000)                                       |
| `-r, --rotation`                                 | Enables circuits rotation (default: false)                                   |
| `--solver SOLVER`                                | CP solver to use (gecode/chuffed, default: chuffed)                          |
| `--heu HEU`                                      | CP search heuristic (input_order/first_fail/dom_w_deg, default: input_order) |
| `--restart RESTART`                              | CP restart strategy (none/lub/geom), default: none)                          |

### Results

Here's the link to a spreadsheet with results of various experiments, made up combining different hyperparameters 
(solver, use of symmetry breaking, rotations, restart, search heuristic, VOHs)

https://docs.google.com/spreadsheets/d/1oityTs147Fbq10D1cxfhFs7Oad5r_t9sa16O9LzCKjY/edit#gid=0