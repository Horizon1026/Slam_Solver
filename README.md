# Slam_Solver
General solver for slam problem, such as graph optimization problem solver.

# Components
- [x] General graph optimization solver.
    - [x] Vertex template.
    - [x] Edge template.
    - [x] Graph problem template.
    - [x] Non-linear optimization solver template.
    - [x] Marginalization.
    - [x] Support tbb parallel.
- [ ] Square root BA problem solver.
    - [ ] Landmark block template.
    - [ ] Non-linear optimization solver template.
    - [ ] Marginalization.
    - [ ] Support tbb parallel.
- [x] Kalman filter problem solver.
    - [x] Basic kalman filter.
    - [x] Error state kalman filter.
    - [x] Square root error state kalman filter.
    - [ ] Iteration kalman filter.

# Dependence
- Slam_Utility
- oneTBB

# Tips
- 欢迎一起交流学习，不同意商用；
- 使用方法参考 /test，暂时没时间写详细文档；
