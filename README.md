# VLSI Design

VLSI (Very Large Scale Integration) refers to the trend of integrating circuits into silicon chips.
A typical example is the smartphone. The modern trend of shrinking transistor sizes, allowing engineers to 
fit more and more transistors into the same area of silicon, has pushed the integration of more and more functions 
of cellphone circuitry into a single silicon die (i.e. plate). This enabled the modern cellphone to mature into a 
powerful tool that shrank from the size of a large brick-sized unit to a device small enough to comfortably carry 
in a pocket or purse, with a video camera, touchscreen, and other advanced features.

# Usage

## Setup the virtual environment

This project is executed in a virtual environment created using `pipenv`, so the first thing to do will be to install
it:

```commandline
pip install pipenv
```

To activate the virtual environment, go inside the `/cdmo_vlsi` folder and run:

```commandline
pipenv shell
```

Once the virtual environment is activated, you must install all the required packages for the project, which can be done
by running:

```commandline
pipenv install -r requirements.txt
```

Now, all the packages should be installed. Ready to solve some instances

## CP
In order to solve the instances by means of CP you should first move to the `CP/src` folder
and then launch the `solve_cp.py` script, for which you can specify the following parameters:

```
python solve_cp.py [-h] [-s START] [-e END] [-t TIMEOUT] [-r]
               [--solver SOLVER] [--heu HEU] [--restart RESTART]

```

Command line arguments:

| Argument                        | Description                                                                  |
|---------------------------------|------------------------------------------------------------------------------|
| `-h, --help`                    | Shows help message                                                           |
| `-s START, --start START`       | First instance to solve (default: 1)                                         |
| `-e END, --end END`             | Last instance to solve (default: 40)                                         |
| `-t TIMEOUT, --timeout TIMEOUT` | Sets the timeout (ms, default: 300000)                                       |
| `-r, --rotation`                | Enables circuits rotation (default: false)                                   |
| `--solver SOLVER`               | CP solver to use (gecode/chuffed, default: chuffed)                          |
| `--heu HEU`                     | CP search heuristic (input_order/first_fail/dom_w_deg, default: input_order) |
| `--restart RESTART`             | CP restart strategy (none/lub/geom), default: none)                          |

## SAT
In order to solve the instances using SAT, you should move to the `SAT/src` folder and launch the `solve_sat.py` file. 
You can specify the following parameters:

```
python solve_sat.py [-h] [-s START] [-e END] [-t TIMEOUT] [-r] [-sb]

```

Command line arguments:

| Argument                        | Description                                                                  |
|---------------------------------|------------------------------------------------------------------------------|
| `-h, --help`                    | Shows help message                                                           |
| `-s START, --start START`       | First instance to solve (default: 1)                                         |
| `-e END, --end END`             | Last instance to solve (default: 40)                                         |
| `-t TIMEOUT, --timeout TIMEOUT` | Sets the timeout (ms, default: 300000)                                       |
| `-r, --rotation`                | Enables circuits rotation (default: false)                                   |
| `--sb SIMMETRY_BREAKING`        | Enables symmetry breaking (default: false)                                   |

## SMT

To solve the instances using SMT, you first have to move into the `SMT/src` folder, and launch the `solve_smt.py` script:

```
python solve_sm.py [-h] [-s START] [-e END] [-t TIMEOUT] [-r] [-sb]

```

The arguments are the same as for the SAT script

## MIP

To solve the instances using SMT, you first have to move into the `MIP/src` folder, and launch the `solve_mip.py` script:

```
python solve_sm.py [-h] [-s START] [-e END] [-t TIMEOUT] [-r] [-sb]

```

The arguments are the same as for the SAT and SMT scripts