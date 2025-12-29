# BERT Grammar 编译指南

## 前置要求

1. **librime** 开发环境已配置
2. **ONNX Runtime** 库（推荐版本 1.12+）

## 安装 ONNX Runtime

### Windows

1. 从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载预编译包
2. 解压到某个目录，例如 `C:\onnxruntime`
3. 设置环境变量或 CMake 变量：
   ```cmake
   set(ONNXRUNTIME_ROOT_DIR "C:/onnxruntime")
   ```

### Linux

**方法 1: 使用预编译包**
```bash
# 下载并解压
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export ONNXRUNTIME_ROOT_DIR=$(pwd)/onnxruntime-linux-x64-1.16.0
```

**方法 2: 使用包管理器（如果可用）**
```bash
# Ubuntu/Debian (如果仓库有)
sudo apt-get install libonnxruntime-dev
```

### macOS

```bash
# 使用 Homebrew
brew install onnxruntime

# 或下载预编译包
# 从 GitHub releases 下载 macOS 版本
```

## 编译步骤

### 1. 配置 CMake

在 librime 的构建目录中：

```bash
cd librime
mkdir build && cd build

# 设置 ONNX Runtime 路径（如果未通过 find_package 找到）
cmake .. \
  -DENABLE_EXTERNAL_PLUGINS=ON \
  -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime \
  -DENABLE_ONNXRUNTIME=ON
```

### 2. 编译

```bash
make -j$(nproc)  # Linux/macOS
# 或
cmake --build . --config Release  # Windows
```

### 3. 验证

检查编译输出中是否有：
```
ONNX Runtime: ENABLED
  Include dirs: ...
  Libraries: ...
```

如果看到警告 "ONNX Runtime not found"，请检查：
- `ONNXRUNTIME_ROOT_DIR` 是否正确设置
- ONNX Runtime 库文件是否存在
- 路径是否正确（注意 Windows 使用反斜杠或正斜杠）

## 常见问题

### Q: 找不到 ONNX Runtime

**A:** 确保：
1. `ONNXRUNTIME_ROOT_DIR` 指向正确的目录
2. 目录结构正确：
   ```
   onnxruntime/
   ├── include/
   │   └── onnxruntime_cxx_api.h
   └── lib/
       └── libonnxruntime.so (或 .dll/.dylib)
   ```

### Q: 链接错误

**A:** 
- Windows: 确保链接了正确的库（Debug/Release 版本匹配）
- Linux: 确保库文件在链接器搜索路径中
- 检查库文件架构（x64/x86）是否匹配

### Q: 运行时错误 "ONNX Runtime not available"

**A:** 
- 确保编译时定义了 `RIME_USE_ONNXRUNTIME`
- 检查 ONNX Runtime 动态库是否在运行时路径中
- Windows: 将 DLL 复制到可执行文件目录或添加到 PATH

## 测试

编译成功后，在配置文件中启用：

```yaml
# default.yaml 或 schema.yaml
bert_grammar:
  model_path: "bert_grammar/model.onnx"
  vocab_path: "bert_grammar/vocab.txt"
```

重新部署 Rime 并测试输入。


