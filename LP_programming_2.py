import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Defining the model
model = pyo.ConcreteModel()

# Sets
model.i = pyo.Set(initialize = ['Desk','Table', 'Chair'])

# Parameters
model.L = pyo.Param(model.i, initialize = {'Desk':8,'Table':6,'Chair':1})
L = model.L
model.F = pyo.Param(model.i, initialize = {'Desk':4,'Table':2,'Chair':1.5})
F = model.F
model.C = pyo.Param(model.i, initialize = {'Desk':2,'Table':1.5,'Chair':0.5})
C = model.C
model.P = pyo.Param(model.i, initialize = {'Desk':60,'Table':30,'Chair':20})
P = model.P

# Decision variable
model.x = pyo.Var(model.i, within = pyo.NonNegativeReals)
x = model.x

# Objective function
def objective_rule(model):
    return sum(P[i]*x[i] for i in model.i)

model.Obj = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

# Constrains
def contraint1(model,i):
    return sum(L[i]*x[i] for i in model.i)<=48
model.Const1 = pyo.Constraint(model.i,rule=contraint1)

def contraint2(model,i):
    return sum(F[i]*x[i] for i in model.i)<=20
model.Const2 = pyo.Constraint(model.i,rule=contraint2)

def contraint3(model,i):
    return sum(C[i]*x[i] for i in model.i)<=8
model.Const3 = pyo.Constraint(model.i,rule=contraint3)

def contraint4(model,i):
    return x['Table']<=5

# Solve
solver = SolverFactory('glpk')
result = solver.solve(model)

print(result)
print(f"Objective function = {model.Obj()}")
for i in model.i:
    print(f"Number of {i} produced = {x[i]()}")