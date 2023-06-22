include "globals.mzn";



%%%%% Parameters

% width of the plate
int: w;

% number of circuits to be placed
int: n;

% height's upper and lower bounds
int: maxh; 
int: minh;

% indices of circuits
set of int: circuits = 1..n;

% arrays of heights and widths of each circuit
array [circuits] of int: y;
array [circuits] of int: x;



%%%%% Variables

% horizontal and vertical coordinates
array [circuits] of var 0..w-min(x): xhat;
array [circuits] of var 0..maxh-min(y): yhat;

% height of the plate
var minh..maxh: h;


% index of the circuit with the biggest area
% int : biggest_circuit_idx = arg_max([y[c] * x[c] | c in circuits]);



%%%%% Constraints

constraint forall (i in circuits) (xhat[i] <= w - x[i]);
constraint forall (i in circuits) (yhat[i] <= h - y[i]);


% Implied constraints
constraint cumulative(xhat, x, y, h)::domain;
constraint cumulative(yhat, y, x, w)::domain;

% No-overlapping constrait
constraint diffn(xhat, yhat, x, y)::domain;


%%%%% Search strategies

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
