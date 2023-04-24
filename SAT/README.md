# SAT

Solves the VLSI design problem using SAT

## Usage

In order to solve the instances you should launch the `solve_sat.py` file . You can specify the following parameters

```
python main.py [-h] [-s START] [-e END] [-t TIMEOUT] [-r] [-sb]

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
