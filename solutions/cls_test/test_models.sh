#!/usr/bin/env bash

# 批量测试脚本：遍历 /home/recamera/tdl_models/cv181x/ 下的模型
# 对每个 .cvimodel 调用 cls_test 并记录是否支持

MODELS_DIR="/home/recamera/tdl_models/cv181x"
CLS_BIN="$(pwd)/cls_test"  # assume running from build dir
RESULT_FILE="results.txt"

if [ ! -x "$CLS_BIN" ]; then
  echo "Error: cls_test 可执行程序不存在或不可执行: $CLS_BIN"
  echo "请在 build 目录运行本脚本，或修改脚本中的 CLS_BIN 变量。"
  exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
  echo "Error: 模型目录不存在: $MODELS_DIR"
  exit 1
fi

echo "测试开始: $(date)" > "$RESULT_FILE"

for m in "$MODELS_DIR"/*.cvimodel; do
  [ -e "$m" ] || continue
  echo "\n=== 测试模型: $m ===" | tee -a "$RESULT_FILE"

  # 运行 cls_test，捕获输出和返回码
  OUTPUT=$("$CLS_BIN" "$m" 2>&1)
  RET=$?

  echo "$OUTPUT" | tee -a "$RESULT_FILE"
  if [ $RET -eq 0 ]; then
    echo "RESULT: SUPPORTED" | tee -a "$RESULT_FILE"
  else
    # 如果 ModelFactory::create 失败，我们会在输出中看到提示
    # 仅作保守判断：如果输出包含 "模型实例创建成功" 则认为支持
    if echo "$OUTPUT" | grep -q "模型实例创建成功"; then
      echo "RESULT: SUPPORTED (by string)" | tee -a "$RESULT_FILE"
    else
      echo "RESULT: NOT SUPPORTED" | tee -a "$RESULT_FILE"
    fi
  fi

done

echo "\n测试结束: $(date)" | tee -a "$RESULT_FILE"

echo "结果已写入: $RESULT_FILE"
