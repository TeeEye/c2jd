from solver.solver import Solver
from model.baseline import Baseline

"""
    创建一个模型,
    训练这个模型,
    测试这个模型
"""

if __name__ == '__main__':
    print('C2JD Start.')
    model = Baseline()
    solver = Solver(model)
    solver.solve()
    solver.evaluate()
    print('C2JD End.')
