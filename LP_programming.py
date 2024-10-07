import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Defining the model
model = pyo.ConcreteModel()

# Decisison variables
model.x1 = pyo.Var(within = pyo.NonNegativeReals)
x1 = model.x1
model.x2 = pyo.Var(within = pyo.NonNegativeReals)
x2 = model.x2

# Objective function
model.Obj = pyo.Objective(expr = 4*x1 + 3*x2, sense = pyo.maximize)

# Constrains
model.Const1 = pyo.Constraint(expr = x1 + x2<=40)
model.Const2 = pyo.Constraint(expr = 2*x1 + x2<=60)

optm = SolverFactory('glpk')
result = optm.solve(model)

print(result)
print('objective function = ', model.Obj())
print(f"x1 = {x1()}, x2 = {x2()}")

