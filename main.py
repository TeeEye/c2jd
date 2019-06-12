from solver.solver import Solver
from model.baseline import Baseline

if __name__ == '__main__':
    print('Baseline Start.')
    model = Baseline()
    solver = Solver(model)
    solver.solve()
    solver.evaluate()
    print('Baseline End.')
