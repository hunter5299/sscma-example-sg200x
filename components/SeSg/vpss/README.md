# VPSS 初始化助手 (SeSg)
`components/SeSg/vpss` 提供了一个非常轻量的封装，统一调用 Sophgo 视频模块里的 VPSS 初始化/反初始化接口。这个库可以让外部模块在不直接依赖 `components/sophgo/video` 的头文件的前提下，方便地完成 VPSS 资源申请与释放。

## 接口
```c
#include <sesg/vpss.h>

if (sesg_vpss_init() != 0) {
    // 初始化失败
}

// 运行你的 VPSS/编码逻辑

sesg_vpss_deinit();
```
- `sesg_vpss_init()` 直接映射到 `app_ipcam_Vpss_Init()`，会初始化视频处理子系统；`sesg_vpss_deinit()` 对应 `app_ipcam_Vpss_DeInit()`。
- 这个封装避免了 SOP/SDK 头文件污染，只导出一个纯 C ABI，方便混用 C++/C 项目。

## 构建与集成
使用 `component_register`，生成组件名 `sesg_vpss`。在你的 CMakeLists.txt 里做：
```cmake
find_package(ssg REQUIRED)  # 视你的项目结构而定
add_executable(my_vpss_tool ...)
target_link_libraries(my_vpss_tool PRIVATE sesg_vpss)
```
`component_register` 已把头文件 `sesg/vpss.h` 安装到 SDK include 目录，直接 `#include <sesg/vpss.h>` 即可。

## 建议的使用场景
1. **独立的摄像头服务**：在自定义的 `camera` 控制线程里调用 `sesg_vpss_init()` 后再启动 `Camera::startStream()`，结束时 `sesg_vpss_deinit()`。
2. **业务模块测试**：可以写个最小可运行示例：
   ```c
   #include <stdio.h>
   #include <sesg/vpss.h>

   int main(void) {
       if (sesg_vpss_init() != 0) {
           fprintf(stderr, "VPSS init fail\n");
           return 1;
       }
       printf("VPSS ready\n");
       sesg_vpss_deinit();
       return 0;
   }
   ```
3. **系统启动/关闭钩子**：如果你的服务需要在整个设备启动时提前配置 VPSS，也可以在 daemon 主线程中 `sesg_vpss_init()`，结束时再反初始化。

## 依赖说明
- 依赖 `components/sophgo/video`（已在 `component_register` 的 `PRIVATE_REQUIREDS sophgo` 中声明）。
- 被封装为 C 接口后，可在 C++ 项目中直接链接 `sesg_vpss` 组件而无需引入 Sophgo 内部头。

是否还需要我给你写一个简单的 `CMakeLists.txt` 示例或 mini test 用来验证 `sesg_vpss_init()`/`deinit()`？