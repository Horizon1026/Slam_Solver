# Slam_Solver
General solver for slam problem, such as graph optimization problem solver.

# Components
- [x] General graph optimization solver.
    - [x] Vertex template.
    - [x] Edge template.
    - [x] Graph problem template.
    - [x] Non-linear optimization solver template.
        - [x] Dog-leg solver.
        - [x] Levenberg-Marquardt solver.
    - [x] Marginalization.
    - [x] Support tbb parallel.
- [x] Kalman filter problem solver.
    - [x] Basic kalman filter.
    - [x] Error state kalman filter.
    - [x] Square root error state kalman filter.
    - [ ] Iteration kalman filter.
- [x] Polynomial solver. (3rd lib)
- [x] Linear pose graph solver.

# Dependence
- Slam_Utility
- Visualizor2D (only for test)
- Visualizor3D (only for test)
- oneTBB (没有也可以用)

# Tips
- 欢迎一起交流学习，不同意商用；
- 使用方法参考 ./test/test_xxx.cpp，暂时没时间写详细文档；
