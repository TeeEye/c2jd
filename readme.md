# Overview
本项目的主要流程是:
1. 通过 offline/* 模块生成原始数据
2. 通过 utils/preprocess 将原始数据处理为训练数据, 目前训练数据格式为 sentence1 | sentence2 | label
    表示的是 sentence1 和 sentence2 所对应的职位是否一致
3. 选取 model/ 下的一个模型作为 model
4. 创建 solver, 来求解 model 的最优参数