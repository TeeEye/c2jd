from solver.solver import Solver
from model.esim import ESIM

if __name__ == '__main__':
    print('Baseline Start.')
    model = ESIM()
    solver = Solver(model)
    solver.solve()
    solver.evaluate()
    print('Baseline End.')
