# RimeDllWrapper 使用指南

`RimeDllWrapper` 类提供了通过 ctypes 直接调用 `rime.dll` 的接口，比 `RimeConsoleWrapper` 更灵活，但实现复杂度更高。

## 快速开始

### 基本使用

```python
from call_librime import RimeDllWrapper

# 1. 创建包装器（自动查找 rime.dll）
rime = RimeDllWrapper()

# 2. 初始化
rime.initialize(app_name="rime.python")

# 3. 创建会话
session_id = rime.create_session()

# 4. 输入拼音
rime.simulate_key_sequence(session_id, "congmingdeRime shurufa")

# 5. 获取上下文（候选词）
context = rime.get_context(session_id)
if context:
    print(f"输入: {context.get('input', '')}")
    for i, cand in enumerate(context.get('candidates', [])):
        print(f"{i+1}. {cand.get('text', '')}")

# 6. 清理
rime.destroy_session(session_id)
rime.finalize()
```

### 使用自定义 DLL 路径

```python
# 手动指定 DLL 路径
dll_path = r"D:\vscode\rime_projs\librime\build\bin\Release\rime.dll"
rime = RimeDllWrapper(dll_path=dll_path)
```

## 完整示例

运行示例文件：

```bash
# 基本使用示例
python example_use_dll_wrapper.py --example basic

# 使用自定义路径
python example_use_dll_wrapper.py --example custom

# 对比两种包装器
python example_use_dll_wrapper.py --example compare
```

## API 说明

### 初始化

```python
rime = RimeDllWrapper(dll_path=None)
```
- `dll_path`: rime.dll 的路径，如果为 `None` 会自动查找

### 方法

#### `initialize(app_name="rime.python")`
初始化 Rime 库
- `app_name`: 应用程序名称，格式应为 "rime.xxx"

#### `create_session() -> int`
创建输入会话
- 返回: 会话 ID（如果失败返回 0）

#### `destroy_session(session_id: int) -> bool`
销毁会话
- `session_id`: 要销毁的会话 ID
- 返回: 是否成功

#### `simulate_key_sequence(session_id: int, key_sequence: str) -> bool`
模拟按键序列
- `session_id`: 会话 ID
- `key_sequence`: 按键序列（例如："congmingdeRime shurufa"）
- 返回: 是否成功

#### `get_context(session_id: int) -> Optional[Dict]`
获取输入上下文（候选词等）
- `session_id`: 会话 ID
- 返回: 包含输入和候选词的字典，格式：
  ```python
  {
      'input': '输入的文本',
      'candidates': [
          {'text': '候选词1', 'comment': '注释1'},
          {'text': '候选词2', 'comment': '注释2'},
          ...
      ]
  }
  ```

#### `finalize()`
清理 Rime 资源

## 注意事项

### 当前实现状态

⚠️ **重要**: `RimeDllWrapper` 当前为**简化实现版本**，部分功能可能不完整：

1. **已实现**:
   - DLL 加载和路径查找
   - 基本初始化
   - 会话创建/销毁（如果 DLL 导出相关函数）
   - 按键序列模拟（如果 DLL 导出相关函数）

2. **部分实现**:
   - 上下文获取（需要完整定义 RimeContext 结构体）
   - 候选词列表（需要完整实现）

3. **需要完善**:
   - 完整的 API 结构体定义
   - 内存管理（释放 RimeContext 等）
   - 错误处理
   - 更多 API 函数

### 推荐使用场景

- ✅ **快速测试和开发**: 使用 `RimeConsoleWrapper`（更简单、功能完整）
- ✅ **生产环境或需要高性能**: 完善 `RimeDllWrapper` 实现
- ✅ **学习 C API 绑定**: 参考 `RimeDllWrapper` 的实现

### 常见问题

#### 1. 找不到 rime.dll

**错误信息**:
```
FileNotFoundError: 找不到 rime.dll
```

**解决方法**:
- 确保已构建 librime
- 检查 DLL 是否在以下位置之一：
  - `librime/build/bin/Release/rime.dll`
  - `librime/build/bin/rime.dll`
- 或手动指定路径：
  ```python
  rime = RimeDllWrapper(dll_path="完整路径/rime.dll")
  ```

#### 2. 加载 DLL 失败

**错误信息**:
```
RuntimeError: 加载 rime.dll 失败
```

**解决方法**:
- 确保 rime.dll 及其依赖（如 onnxruntime.dll）在同一目录
- 检查系统架构是否匹配（32位/64位）
- 在 Windows 上，确保 Visual C++ 运行库已安装

#### 3. 会话创建失败

**可能原因**:
- DLL 未导出 `RimeCreateSession` 函数
- 需要通过 `rime_get_api()` 获取 API 结构体访问

**解决方法**:
- 使用 `RimeConsoleWrapper` 作为替代
- 或完善 `RimeDllWrapper` 实现，通过 API 结构体访问函数

#### 4. 无法获取候选词

**原因**: 需要完整定义 `RimeContext` 结构体

**解决方法**:
- 参考 `librime/src/rime_api.h` 中的结构体定义
- 使用 ctypes 定义对应的 Python 结构体
- 正确管理内存（调用 `free_context`）

## 实现完整版本的步骤

如果要实现完整的 `RimeDllWrapper`，需要：

1. **定义 C 结构体**:
   ```python
   class RimeTraits(ctypes.Structure): ...
   class RimeContext(ctypes.Structure): ...
   class RimeCommit(ctypes.Structure): ...
   class RimeStatus(ctypes.Structure): ...
   # ... 等等
   ```

2. **获取 API 结构体**:
   ```python
   api_ptr = dll.rime_get_api()
   # 通过偏移量或结构体定义访问函数指针
   ```

3. **实现函数调用**:
   ```python
   # 定义函数指针类型
   CreateSessionFunc = ctypes.CFUNCTYPE(ctypes.c_uint64)
   # 从 API 结构体中获取函数指针
   create_session = CreateSessionFunc(api_ptr.create_session)
   ```

4. **内存管理**:
   - 正确释放 RimeContext、RimeCommit 等结构体
   - 使用 `free_context`、`free_commit` 等函数

## 参考资源

- `librime/src/rime_api.h` - C API 定义
- `librime/tools/rime_api_console.cc` - 官方使用示例
- `call_librime.py` - 当前实现代码
- `example_use_dll_wrapper.py` - 使用示例

## 与 RimeConsoleWrapper 对比

| 特性 | RimeDllWrapper | RimeConsoleWrapper |
|------|---------------|-------------------|
| 实现复杂度 | 高 | 低 |
| 性能 | 高（无进程间通信） | 中（进程间通信） |
| 功能完整性 | 需要完善 | 完整 |
| 易用性 | 中 | 高 |
| 推荐场景 | 生产环境、高性能需求 | 快速开发、测试 |

## 总结

- **快速开始**: 使用 `RimeConsoleWrapper`
- **深入学习**: 参考 `RimeDllWrapper` 的实现
- **生产使用**: 完善 `RimeDllWrapper` 或使用官方 Python 绑定（如果有）

