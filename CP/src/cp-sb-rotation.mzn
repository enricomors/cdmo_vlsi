include "globals.mzn";

% width of the plate
int: w;

% number of circuits
int: n;

% height's upper and lower bounds
int: maxh;
int: minh;

% indices of circuits
set of int: circuits = 1..n;

% arrays of heights and widths of each circuit before rotating
array [circuits] of 1..w: xinput;
array [circuits] of 1..maxh: yinput;

% arrays of heights and widths of each circuit
array [circuits] of var 1..w: x;
array [circuits] of var 1..maxh: y;

% array of booleans where if a piece is rotated the array has value True, otherwise false
array [circuits] of var bool: rotated;


%%%%% Variables

% height of the plate
var minh..maxh: h;

% horizontal and vertical coordinates
array [circuits] of var 0..w-min(xinput): xhat;
array [circuits] of var 0..maxh-min(yinput): yhat;




%%%%% constraints
constraint forall(i in circuits)(yhat[i] < h);
constraint forall(i in circuits)(xhat[i] + x[i] <= w);
constraint forall(i in circuits)(yhat[i] + y[i] <= h);

% No-overlapping constraint
constraint diffn(xhat, yhat, x, y) :: domain;

% implied constraints
constraint cumulative(xhat, x, y, h) :: domain;
constraint cumulative(yhat, y, x, w) :: domain;

% rotation constraint
constraint forall (i in circuits)(if rotated[i]
then y[i] == xinput [i] /\ x[i] == yinput [i]
else x[i] == xinput [i] /\ y[i] == yinput [i] endif);

% symmetry breaking constraints

% rows and columns symmetry
constraint symmetry_breaking_constraint(
    forall (i,j in circuits where i < j) ((xhat[i] == xhat[j] /\ x[i] == x[j]) -> yhat[i] <= yhat[j] ));
constraint symmetry_breaking_constraint(
    forall (i,j in circuits where i < j) ((yhat[i] == yhat[j] /\ y[i] == y[j]) -> xhat[i] <= xhat[j] ));

% three blocks symmetry
constraint symmetry_breaking_constraint(
    forall (i,j,k in circuits where i > j /\ j > k)
        ((xhat[i] == xhat[j] /\ x[i] == x[j] /\ yhat[i] == yhat[k] /\ y[i] + y[j] == y[k]) -> xhat[k] <= xhat[i] ));
constraint symmetry_breaking_constraint(
    forall (i,j,k in circuits where i > j /\ j > k)
        ((yhat[i] == yhat[j] /\ y[i] == y[j] /\ xhat[i] == xhat[k] /\ x[i] + x[j] == x[k]) -> yhat[k] <= yhat[i] ));

% force the biggest block to be always to the bottom left of the second biggest
constraint symmetry_breaking_constraint(lex_lesseq([xhat[1], yhat[1]], [xhat[2],yhat[2]]));

% constraint for square circuits
constraint forall (i in circuits) (xinput [i] == yinput [i] -> rotated[i] = false);



%%%%% Search Strategies
string: search;
string: restart;

ann: search_xhat = 
if search == "dom" then
  int_search(xhat, dom_w_deg, indomain_min)
elseif search == "ff" then
  int_search(xhat, first_fail, indomain_min)
else
  int_search(xhat, input_order, indomain_min)
endif;

ann: search_yhat = 
if search == "dom" then 
  int_search(yhat, dom_w_deg, indomain_min)
elseif search == "ff" then
  int_search(yhat, first_fail, indomain_min)
else
  int_search(yhat, input_order, indomain_min)
endif;

ann: restart_ann = 
if restart == "lub" then
  restart_luby(150)
elseif restart == "geom" then
  restart_geometric(2, 50)
else
  restart_none
endif;


%%%%% Solve parameters
solve :: search_xhat
      :: search_yhat
      :: restart_ann
         minimize h;