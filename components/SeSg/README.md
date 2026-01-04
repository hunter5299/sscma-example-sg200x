# SeSg 组件扩展说明

## 第一章：如何移植 SSCMA 之外的新库（并放到 components/SeSg）

本仓库的构建体系把“第三方库/新增库”当作一个 **component** 来管理。
你把头文件、库文件、（可选）源代码放进 `components/SeSg/<name>/`，再用 `component_register()` 注册，就能在任意 solution 里通过 `REQUIREDS` 引用。

### 1.1 目标目录结构（推荐）

以 `<name>=foo` 为例：

- `components/SeSg/foo/CMakeLists.txt`
- `components/SeSg/foo/include/`（放头文件）
- `components/SeSg/foo/lib/`（放交叉编译出的 `.so/.a`）
- （可选）`components/SeSg/foo/src/`（放你需要一起编译的源码）

说明：
- “只有头文件 + 预编译库”的场景非常常见（例如你这里的 OpenCV objdetect 相关库）。
- “带源码一起编”的场景适用于你想把第三方库直接编进静态库，或需要打补丁的情况。

### 1.2 只引入预编译库（最常用）

在 `components/SeSg/foo/CMakeLists.txt`：

- 用 `component_register()` 只填写 `INCLUDE_DIRS` / `LIBRARY_DIRS`
- 不需要 `SRCS`（不会生成新的库目标，只是把 include/lib 路径注入到工程）

最小模板：

```cmake
set(_DIR ${CMAKE_CURRENT_LIST_DIR})

component_register(
    COMPONENT_NAME foo
    INCLUDE_DIRS
        ${_DIR}/include
        # 如果你的头文件是 OpenCV 风格（include/opencv4/opencv2/...），把 opencv4 也加上
        ${_DIR}/include/opencv4
    LIBRARY_DIRS ${_DIR}/lib
)
```

然后在某个 solution 的 `main/CMakeLists.txt` 里使用：

- `REQUIREDS foo`（需要“对外传递”的依赖）
- `PRIVATE_REQUIREDS foo`（只在本组件内部使用，不对外传播）

### 1.3 带源码一起编（你需要改源码/打补丁时）

如果你把第三方库源码放进 `components/SeSg/foo/src`：

```cmake
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_LIST_DIR}/src/*.c ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

component_register(
    COMPONENT_NAME foo
    SRCS ${srcs}
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
)
```

这会生成一个名为 `foo` 的静态库（默认）并参与链接。

### 1.4 交叉编译与 ABI 一致性（非常关键）

把库“放进 components/SeSg”只是组织方式，**能否正常链接/运行**取决于 ABI 是否一致：

- 交叉工具链：必须与当前工程一致（你这里是 `riscv64-unknown-linux-musl-*`）
- libc：必须匹配 musl
- C++ ABI：编译器版本、`-fno-exceptions/-fno-rtti` 等差异都可能踩雷
- 依赖闭包：一个 `.so` 的 `DT_NEEDED` 里依赖的其它 `.so` 也必须同时提供

实践建议：
- 像 OpenCV 这类“模块化大库”，不要只拷一个 `.so` 就结束；要把它依赖的模块（例如 `calib3d/features2d/flann`）一并带上。
- 如果出现“链接时找不到 transitive 依赖”（例如缺 `libz.so.1`），可以在可执行文件链接阶段显式加上 `z`（也就是 `-lz`）。

### 1.5 运行时部署（设备上也要找得到 .so）

编译通过 ≠ 设备运行一定正常。
你需要保证设备运行时能找到这些 `.so`：

- 把 `.so*` 拷到设备的库目录（例如 `/mnt/system/lib`），或
- 在启动程序前设置 `LD_LIBRARY_PATH` 指向你部署的目录

建议：把 `components/SeSg/foo/lib` 中的 `.so*` 在打包阶段同步到设备 rootfs 的 lib 目录，避免每次手动设置环境变量。

### 1.6 以 opencv_objdetect 为例：只编 objdetect（riscv64 musl）并落到 components/SeSg（+ 最小清单 + 设备端部署）

本节给一套“可照抄执行”的流程：

- 交叉编译 OpenCV 4.5.x 的 `objdetect`（最小化构建）
- 安装（落盘）到 `components/SeSg/opencv_objdetect`
- 列出你必须一起带上的 `libopencv_*.so*` 最小清单
- 写清楚设备端如何部署与验证

#### 1.6.1 准备变量（确保和本工程一致）

```bash
export SG200X_SDK_PATH=/home/steven/sg2002_recamera_emmc
export SYSROOT=$SG200X_SDK_PATH/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot

# 注意：仓库 toolchain 文件里用的变量名是 ARM_SYSROOT_PATH（虽然目标是 riscv）
export ARM_SYSROOT_PATH=$SYSROOT
```

仓库 toolchain 文件路径：

- `toolchain-riscv64-linux-musl-x86_64.cmake`

#### 1.6.2 获取 OpenCV 源码（必须匹配 4.5.x，建议 4.5.0）

有网络：

```bash
git clone -b 4.5.0 --depth 1 https://github.com/opencv/opencv.git opencv-4.5.0
```

没网络：把 OpenCV 源码目录准备好即可（例如你解压到 `opencv-4.5.0/`），下面命令里的路径对应替换。

#### 1.6.3 CMake 配置：只构建 objdetect（最小化）

在 OpenCV 源码目录同级执行（示例）：

```bash
rm -rf build-opencv-objdetect

cmake -S . -B build-opencv-objdetect \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/home/steven/sscma-example-sg200x/cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
  -DCMAKE_SYSROOT=$SYSROOT \
  -DARM_SYSROOT_PATH=$SYSROOT \
  -DCMAKE_INSTALL_PREFIX=/home/steven/sscma-example-sg200x/components/SeSg/opencv_objdetect \
  -DBUILD_LIST=core,imgproc,objdetect \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python3=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_OPENMP=OFF

```

说明：

- `BUILD_LIST=objdetect` 会让 OpenCV 只构建 objdetect 以及它需要的最小依赖（通常至少会带上 `core/imgproc`）。
- 如果 OpenCV 构建过程中发现 objdetect 还硬依赖其它模块（常见是 `calib3d/features2d/flann`），它们也会被一起构建出来；这种情况你也必须把对应 `libopencv_*.so*` 一并带到 `components/SeSg/opencv_objdetect/lib`，否则链接/运行会缺库。

#### 1.6.4 编译 + 安装到 components/SeSg

```bash
cmake --build build-opencv-objdetect -j --target opencv_objdetect
cmake --install build-opencv-objdetect
```

#### 1.6.5 验证落盘结果

你最终希望看到至少这些东西之一（取决于 OpenCV 安装布局）：

- `components/SeSg/opencv_objdetect/lib/libopencv_objdetect.so*`
- `components/SeSg/opencv_objdetect/include/opencv2/objdetect.hpp`
- 或 `components/SeSg/opencv_objdetect/include/opencv4/opencv2/objdetect.hpp`

说明：我加的 shim 组件同时加了 `include/` 和 `include/opencv4/` 两条 include 路径，所以两种布局都能让 `#include <opencv2/objdetect.hpp>` 正常找到头文件。

#### 1.6.6 最小可用组件清单（你必须一起带上的 OpenCV 子库）

`opencv_objdetect` 是典型的“只引入预编译库”的组件。为了让 `cv::QRCodeDetector` 能正常链接/运行，建议在同一个 `components/SeSg/opencv_objdetect/lib/` 下至少具备：

- `libopencv_objdetect.so*`
- `libopencv_calib3d.so*`
- `libopencv_features2d.so*`
- `libopencv_flann.so*`
- `libopencv_imgproc.so*`
- `libopencv_core.so*`

说明：`cv::QRCodeDetector` 属于 objdetect 模块，但内部会走到几何/特征相关能力，所以“只带 objdetect 一个 .so”经常会在链接或运行期缺符号。

#### 1.6.7 工程侧怎么用（只动新工程/新项目）

在你要用 `cv::QRCodeDetector` 的那个 solution 的 `main` 组件里，把 `REQUIREDS` 增加：

- `opencv_objdetect`

额外提醒：如果你发现链接器把 SDK 自带的 `-lopencv_core` 等库抢在前面（导致符号缺失/ABI 不一致），最稳妥的做法是在可执行文件链接阶段显式链接 `components/SeSg/opencv_objdetect/lib` 下同一套构建出来的 OpenCV 子库（core/imgproc/...）来保证一致性。

#### 1.6.8 设备运行时部署

你把库放到仓库 SeSg 只是“编译期/打包期”的组织方式；板子上运行时仍然需要能找到 `.so`。

推荐做法（择一）：

1) 拷到设备的系统库搜索路径（例如 `/mnt/system/lib`）

2) 随应用一起带一个私有目录（例如应用同级 `./lib`），并在启动前设置：

```bash
export LD_LIBRARY_PATH=/path/to/your/lib:$LD_LIBRARY_PATH
```

你至少要部署：

- `components/SeSg/opencv_objdetect/lib/` 下的所有 `libopencv_*.so*`

另外 OpenCV core 常见还会依赖 `libz.so.1`（zlib）以及 C/C++ 运行库：

- 如果运行期提示缺 `libz.so.1`：确认系统已带 zlib，或把对应 `libz.so*` 一并部署。

#### 1.6.9 如何快速核对依赖闭包（不靠猜）

在主机侧对交叉产物可以用 `readelf` 看 `DT_NEEDED`：

```bash
readelf -d components/SeSg/opencv_objdetect/lib/libopencv_objdetect.so.4.5.0 | grep NEEDED
readelf -d components/SeSg/opencv_objdetect/lib/libopencv_core.so.4.5.0 | grep NEEDED
```

你看到的每一个 `NEEDED` 条目，都要么：

- 在设备的系统库目录里存在，要么
- 由你随应用一起带上并确保运行时能搜到。
