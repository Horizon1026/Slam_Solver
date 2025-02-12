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
    - [x] Audo compute jacobians.(Only work on 'double')
- [x] State filter problem solver.
    - [x] Basic kalman filter.
    - [x] Error state kalman filter.
    - [x] Square root error state kalman filter.
    - [ ] Iteration kalman filter.
    - [x] Basic information/inverse filter.
    - [x] Error state information/inverse filter.
    - [x] Square root error state information/inverse filter.
- [x] Polynomial solver. (3rd lib)
- [x] Linear pose graph solver.

# Dependence
- Slam_Utility
- Visualizor2D (only for test)
- Visualizor3D (only for test)
- oneTBB (没有也可以用)

# Compile and Run
- 第三方仓库的话需要自行 apt-get install 安装
- 拉取 Dependence 中的源码，在当前 repo 中创建 build 文件夹，执行标准 cmake 过程即可
```bash
mkdir build
cmake ..
make -j
```
- 编译成功的可执行文件就在 build 中，具体有哪些可执行文件可参考 run.sh 中的列举。可以直接运行 run.sh 来依次执行所有可执行文件

```bash
sh run.sh
```

# Tips
- 欢迎一起交流学习，不同意商用；
- 使用方法参考 ./test/test_xxx.cpp，暂时没时间写详细文档；
- Square root information filter参考了美团的 SR-ISWF，但从结果上来看似乎并不等价于标准 information filter;
