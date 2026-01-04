# cls_test - 分类模型测试程序

用于测试 SSCMA SDK 支持的分类模型（如 MobileNet、人脸属性分类等）。

## 功能

- 加载并验证 `.cvimodel` 格式的分类模型
- 自动识别模型类型和输入输出格式
- 支持图片输入测试
- 显示分类结果和性能统计

## 编译

```bash
cd solutions/cls_test
mkdir -p build && cd build
cmake ..
cmake --build .
```

## 使用方法

```bash
# 基本用法：只加载模型（使用测试图像）
./cls_test <模型路径>

# 指定测试图片
./cls_test <模型路径> <图片路径>
```

## 示例

```bash
# 测试人脸属性分类模型
./cls_test cls_attribute_gender_age_glass_emotion_112_112_INT8_cv181x.cvimodel

# 使用真实人脸图片测试
./cls_test cls_attribute_gender_age_glass_emotion_112_112_INT8_cv181x.cvimodel face.jpg
```

## 输出说明

程序会输出以下信息：

1. **模型信息**
   - 模型类型 (分类/检测等)
   - 输入尺寸 (如 112x112)
   - 输出类型

2. **分类结果**
   - 类别 ID
   - 置信度 (百分比)

3. **性能统计**
   - 预处理时间
   - 推理时间
   - 后处理时间

## 人脸属性模型说明

`cls_attribute_gender_age_glass_emotion_112_112_INT8_cv181x.cvimodel` 是一个多属性分类模型，可识别：

- 性别 (Gender)
- 年龄 (Age)
- 眼镜 (Glass)
- 表情 (Emotion)

模型输入尺寸为 112x112 像素，使用 INT8 量化。

## 注意事项

1. 确保模型文件是为 cv181x 平台编译的
2. 图片会自动缩放到模型需要的尺寸
3. 如果模型类型不被 SDK 支持，程序会提示错误
