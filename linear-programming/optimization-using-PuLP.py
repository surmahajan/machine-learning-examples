# Use Linear Programming Optimization Package PuLP to solve a production scheduling problem


from pulp import *

# Create a problem object. 
# problem constructor receives a problem name and LpMaximize, which means we want to maximize our objective function. 
# In our case, this is the profit from selling a certain number of computers.
problem = LpProblem("problemName", LpMaximize)

# Define the constants that we received from the problem statement
# factory cost per day
cf0 = 450
cf1 = 420
cf2 = 400

# factory throughput per day
f0 = 2000
f1 = 1500
f2 = 1000

# production goal
goal = 80000

# time limit
max_num_days = 30

# number of factories
num_factories = 3

# Decision Variables
# Define variables in PuLP is using the dicts function. This can be very useful in cases where we need to define a 
# large number of variables of the same type and bounds, variableNames would be a list of keys for the dictionary:
# varDict = LpVariable.dicts("boundedVariableName",lowerBound,upperBound)

factory_days = LpVariable.dicts("factoryDays", list(range(num_factories)), 0, 30, cat="Continuous")

# Constraints
# The constraints that we care about are that the number of units assembled should be above or equal to 
# the goal amount and the production constraint that no factory should produce more than double as the other factory:

# Goal Constraint
c1 = factory_days[0]*f0 + factory_days[1]*f1 +factory_days[2]*f2 >= goal

# Production Constraint
c2 = factory_days[0]*f0 <= 2*factory_days[1]*f1
c3 = factory_days[0]*f0 <= 2*factory_days[2]*f2
c4 = factory_days[1]*f1 <= 2*factory_days[2]*f2
c5 = factory_days[1]*f1 <= 2*factory_days[0]*f0
c6 = factory_days[2]*f2 <= 2*factory_days[1]*f1
c7 = factory_days[2]*f2 <= 2*factory_days[0]*f0

# Adding constraints to the problem
problem += c1
problem += c2
problem += c3
problem += c4
problem += c5
problem += c6
problem += c7

# The objective function for the computer assembly problem is basically minimizing the cost of assembling 
# all of those computers. This can be written simply as maximizing the negative cost:
problem += -factory_days[0]*cf0*f0 - factory_days[1]*cf1*f1 - factory_days[2]*cf2*f2


print(problem)

# Solving 
problem.solve()

for i in range(3):
    print(f"Factory {i}: {factory_days[i].varValue}")

