from solver.solver import Solver
from model.baseline import Baseline

if __name__ == '__main__':
    print('C2JD Start.')
    model = Baseline()
    solver = Solver(model)
    solver.solve()
    solver.evaluate()
    print('C2JD End.')
