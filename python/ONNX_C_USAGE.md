# BERT模型ONNX转换及C语言调用指南

## 概述

本指南介绍如何将`bert-base-chinese`模型转换为ONNX格式，并在C语言中调用。

## 步骤1: 转换模型为ONNX格式

### 基本转换

```bash
python convert_to_onnx.py
```

这将：
- 下载并加载`bert-base-chinese`模型
- 转换为ONNX格式
- 保存到`onnx_models/`目录
- 同时保存分词器配置文件

### 高级选项

```bash
# 指定模型名称
python convert_to_onnx.py --model bert-base-chinese

# 指定输出目录
python convert_to_onnx.py --output my_onnx_models

# 只转换编码器部分（更轻量，适合只需要句子嵌入的场景）
python convert_to_onnx.py --encoder-only

# 固定输入长度（不使用动态轴）
python convert_to_onnx.py --no-dynamic

# 指定ONNX opset版本
python convert_to_onnx.py --opset 13
```

### 转换后的文件

转换完成后，`onnx_models/`目录将包含：
- `bert-base-chinese.onnx` - ONNX模型文件
- `tokenizer_config.json` - 分词器配置
- `vocab.txt` - 词汇表文件

## 步骤2: 在C语言中调用ONNX模型

### 安装ONNX Runtime C库

#### Windows

1. 下载ONNX Runtime预编译库：
   - 访问：https://github.com/microsoft/onnxruntime/releases
   - 下载 `onnxruntime-win-x64-<version>.zip`

2. 解压并设置环境变量：
   ```powershell
   # 设置库路径
   $env:ONNXRUNTIME_LIB_PATH = "C:\path\to\onnxruntime\lib"
   $env:ONNXRUNTIME_INCLUDE_PATH = "C:\path\to\onnxruntime\include"
   ```

#### Linux

```bash
# 下载并解压
wget https://github.com/microsoft/onnxruntime/releases/download/v<version>/onnxruntime-linux-x64-<version>.tgz
tar -xzf onnxruntime-linux-x64-<version>.tgz

# 设置环境变量
export ONNXRUNTIME_LIB_PATH=/path/to/onnxruntime/lib
export ONNXRUNTIME_INCLUDE_PATH=/path/to/onnxruntime/include
```

### C语言示例代码

创建 `bert_onnx_inference.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

// 简化的BERT推理示例
int main() {
    // 初始化ONNX Runtime环境
    OrtEnv* env;
    OrtStatus* status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "BERT", &env);
    if (status != NULL) {
        printf("创建环境失败\n");
        return 1;
    }

    // 创建会话选项
    OrtSessionOptions* session_options;
    OrtCreateSessionOptions(&session_options);
    
    // 设置执行提供者（可选，使用CPU）
    // OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1);
    
    // 创建会话
    OrtSession* session;
    const char* model_path = "onnx_models/bert-base-chinese.onnx";
    status = OrtCreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        printf("加载模型失败\n");
        return 1;
    }

    // 获取输入输出信息
    size_t num_input_nodes;
    OrtStatus* status2 = OrtSessionGetInputCount(session, &num_input_nodes);
    printf("输入节点数量: %zu\n", num_input_nodes);

    // 准备输入数据
    // 注意：实际使用时需要先进行分词（tokenization）
    // 这里只是示例，实际需要：
    // 1. 使用vocab.txt进行分词
    // 2. 将文本转换为input_ids和attention_mask
    
    // 示例：假设已经分词后的数据
    int64_t input_ids[] = {101, 872, 1962, 6821, 4518, 1355, 4638, 102}; // [CLS] 这是一个示例 [SEP]
    int64_t attention_mask[] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t shape[] = {1, 8}; // batch_size=1, sequence_length=8

    // 创建输入tensor
    OrtMemoryInfo* memory_info;
    OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    
    OrtValue* input_tensor_ids;
    OrtValue* input_tensor_mask;
    
    // 创建input_ids tensor
    OrtCreateTensorWithDataAsOrtValue(
        memory_info,
        input_ids,
        sizeof(input_ids),
        shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensor_ids
    );
    
    // 创建attention_mask tensor
    OrtCreateTensorWithDataAsOrtValue(
        memory_info,
        attention_mask,
        sizeof(attention_mask),
        shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensor_mask
    );

    // 准备输入名称
    const char* input_names[] = {"input_ids", "attention_mask"};
    OrtValue* inputs[] = {input_tensor_ids, input_tensor_mask};

    // 准备输出
    const char* output_names[] = {"logits"};
    OrtValue* outputs[1];

    // 运行推理
    status = OrtRun(
        session,
        NULL,
        input_names,
        inputs,
        2,
        output_names,
        1,
        outputs
    );

    if (status != NULL) {
        printf("推理失败\n");
        return 1;
    }

    // 获取输出数据
    OrtTensorTypeAndShapeInfo* output_info;
    OrtGetTensorTypeAndShape(outputs[0], &output_info);
    
    size_t num_dims;
    OrtGetDimensionsCount(output_info, &num_dims);
    
    int64_t* output_shape = (int64_t*)malloc(num_dims * sizeof(int64_t));
    OrtGetDimensions(output_info, output_shape, num_dims);
    
    printf("输出形状: [%lld, %lld, %lld]\n", 
           output_shape[0], output_shape[1], output_shape[2]);
    
    // 获取输出数据指针
    float* output_data;
    OrtGetTensorMutableData(outputs[0], (void**)&output_data);
    
    // 使用输出数据...
    // output_data包含logits，形状为 [batch_size, sequence_length, vocab_size]
    
    // 清理资源
    OrtReleaseValue(outputs[0]);
    OrtReleaseValue(input_tensor_ids);
    OrtReleaseValue(input_tensor_mask);
    OrtReleaseSession(session);
    OrtReleaseSessionOptions(session_options);
    OrtReleaseEnv(env);
    
    printf("推理完成！\n");
    return 0;
}
```

### 编译C程序

#### Windows (使用MinGW或Visual Studio)

```bash
# 使用gcc
gcc -o bert_inference bert_onnx_inference.c \
    -I"$ONNXRUNTIME_INCLUDE_PATH" \
    -L"$ONNXRUNTIME_LIB_PATH" \
    -lonnxruntime \
    -std=c99

# 使用cl (Visual Studio)
cl /EHsc bert_onnx_inference.c /I"%ONNXRUNTIME_INCLUDE_PATH%" \
    /link /LIBPATH:"%ONNXRUNTIME_LIB_PATH%" onnxruntime.lib
```

#### Linux

```bash
gcc -o bert_inference bert_onnx_inference.c \
    -I"$ONNXRUNTIME_INCLUDE_PATH" \
    -L"$ONNXRUNTIME_LIB_PATH" \
    -lonnxruntime \
    -std=c99 \
    -Wl,-rpath,"$ONNXRUNTIME_LIB_PATH"
```

## 重要注意事项

### 1. 分词（Tokenization）

BERT模型需要先进行分词。在C语言中，你需要：
- 读取`vocab.txt`词汇表
- 实现分词逻辑（或使用第三方库）
- 将文本转换为`input_ids`和`attention_mask`

### 2. 输入格式

- `input_ids`: `[batch_size, sequence_length]`，类型为`int64`
- `attention_mask`: `[batch_size, sequence_length]`，类型为`int64`
- 最大序列长度：512（bert-base-chinese）

### 3. 输出格式

- `logits`: `[batch_size, sequence_length, vocab_size]`，类型为`float32`
- vocab_size = 21128（bert-base-chinese）

### 4. 性能优化

- 使用GPU执行提供者（如果可用）：
  ```c
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
  ```
- 使用动态形状时注意性能影响
- 考虑批处理多个输入以提高吞吐量

## 参考资源

- [ONNX Runtime C API文档](https://onnxruntime.ai/docs/api/c/)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [BERT模型说明](https://huggingface.co/bert-base-chinese)

## 常见问题

### Q: 如何实现中文分词？
A: 你需要实现BERT的分词逻辑，或者使用现有的C语言分词库。可以参考Hugging Face的tokenizers库的C绑定。

### Q: 模型文件太大怎么办？
A: 可以考虑：
1. 使用量化版本的ONNX模型
2. 使用更小的模型（如TinyBERT）
3. 只转换编码器部分（使用`--encoder-only`选项）

### Q: 如何在嵌入式设备上使用？
A: ONNX Runtime支持多种执行提供者，包括针对嵌入式设备的优化版本。可以查看ONNX Runtime的移动端支持。

