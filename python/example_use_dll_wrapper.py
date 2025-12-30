"""
RimeDllWrapper 使用示例

这个示例演示如何使用 RimeDllWrapper 类来调用 rime.dll
"""

import os
import time
from call_librime import RimeDllWrapper


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("RimeDllWrapper 基本使用示例")
    print("=" * 60)
    
    try:
        # 1. 创建包装器实例（自动查找 rime.dll）
        print("\n1. 创建 RimeDllWrapper 实例...")
        rime = RimeDllWrapper()
        print(f"   ✓ 找到 rime.dll: {rime.dll_path}")
        
        # 2. 初始化 Rime
        print("\n2. 初始化 Rime...")
        # 数据文件在 build/bin/ 目录，而 DLL 在 build/bin/Release/ 目录
        # 需要将数据目录指向 build/bin/（DLL 的父目录）
        dll_dir = os.path.dirname(rime.dll_path)  # build/bin/Release
        data_dir = os.path.dirname(dll_dir)  # build/bin
        
        print(f"   DLL 目录: {dll_dir}")
        print(f"   数据目录: {data_dir}")
        
        # 检查数据文件
        schema_file = os.path.join(data_dir, 'luna_pinyin.schema.yaml')
        if os.path.exists(schema_file):
            print(f"   ✓ 找到数据文件: {schema_file}")
        else:
            print(f"   ⚠ 警告: 未找到数据文件: {schema_file}")
        
        # 指定数据目录
        rime.initialize(
            app_name="rime.python.example",
            shared_data_dir=data_dir,
            user_data_dir=data_dir
        )
        
        # 3. 创建会话
        print("\n3. 创建会话...")
        session_id = rime.create_session()
        if session_id:
            print(f"   ✓ 会话创建成功，ID: {session_id}")
            
            # 4. 检查/选择输入方案
            print("\n4. 检查输入方案...")
            current_schema = rime.get_current_schema(session_id)
            if current_schema:
                print(f"   当前方案: {current_schema}")
            else:
                print("   未选择方案，尝试选择 luna_pinyin...")
                if rime.select_schema(session_id, "luna_pinyin"):
                    print("   ✓ 已选择 luna_pinyin")
                else:
                    print("   ⚠ 选择方案失败，可能影响输入")
            
            # 5. 输入拼音
            print("\n5. 输入拼音...")
            # 注意：simulate_key_sequence 会将大写字母解析为特殊按键
            # 应该使用全小写拼音，参考命令行工具的成功示例
            # 命令行工具输入: "congmingdeshurufa" 可以正常工作
            test_inputs = [
                "congmingdeshurufa",  # 全小写，无空格（推荐）
                "congmingdeRimeshurufa",  # 有大写字母（可能有问题）
            ]
            
            input_success = False
            for input_text in test_inputs:
                print(f"   尝试输入: {input_text}")
                if rime.simulate_key_sequence(session_id, input_text):
                    print("   ✓ 输入成功")
                    input_success = True
                    break
                else:
                    print("   ✗ 输入失败，尝试下一个...")
            
            if not input_success:
                print("   ⚠ 所有输入尝试都失败")
            
            # 6. 获取上下文（候选词）
            print("\n6. 获取输入上下文...")
            # 等待一下让 Rime 处理输入
            # time.sleep(0.2)  # 等待 200ms，确保 Rime 处理完成
            
            # 可选：启用调试模式（取消注释以查看详细信息）
            # os.environ['RIME_DEBUG'] = '1'
            context = rime.get_context(session_id)
            if context:
                input_text = context.get('input', '')
                print(f"   输入文本: {input_text}")
                
                # 显示组合信息
                composition = context.get('composition', {})
                if composition:
                    print(f"   组合长度: {composition.get('length', 0)}")
                    print(f"   光标位置: {composition.get('cursor_pos', 0)}")
                
                candidates = context.get('candidates', [])
                if candidates:
                    print(f"   候选词数量: {len(candidates)}")
                    menu = context.get('menu', {})
                    if menu:
                        print(f"   当前页: {menu.get('page_no', 0) + 1} / 每页: {menu.get('page_size', 0)}")
                        print(f"   高亮索引: {menu.get('highlighted_index', -1)}")
                        print(f"   是否最后一页: {menu.get('is_last_page', False)}")
                    
                    for i, cand in enumerate(candidates[:10]):  # 显示前10个
                        text = cand.get('text', '')
                        comment = cand.get('comment', '')
                        marker = " <--" if menu and i == menu.get('highlighted_index', -1) else ""
                        print(f"   {i+1}. {text}" + (f" ({comment})" if comment else "") + marker)
                else:
                    print("   （无候选词）")
                    print("   提示: 如果输入正确但无候选词，可能是：")
                    print("     1. 数据文件未正确加载")
                    print("     2. 输入格式有问题（应使用全小写拼音）")
                    print("     3. 需要等待 Rime 处理输入")
            else:
                print("   （无法获取上下文）")
            
            # 7. 清理会话
            print("\n7. 清理会话...")
            rime.destroy_session(session_id)
        else:
            print("   ✗ 会话创建失败")
            print("   提示: 如果失败，可能需要使用 RimeConsoleWrapper 或完整实现 API")
        
        # 7. 清理资源
        print("\n7. 清理资源...")
        rime.finalize()
        
        print("\n" + "=" * 60)
        print("示例完成！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示:")
        print("  1. 确保已构建 librime")
        print("  2. 确保 rime.dll 存在于以下位置之一:")
        print("     - librime/build/bin/Release/rime.dll")
        print("     - librime/build/bin/rime.dll")
        print("  3. 或者手动指定 DLL 路径:")
        print("     rime = RimeDllWrapper(dll_path='path/to/rime.dll')")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def example_with_custom_path():
    """使用自定义路径的示例"""
    print("=" * 60)
    print("使用自定义 DLL 路径的示例")
    print("=" * 60)
    
    # 手动指定 DLL 路径
    dll_path = r"D:\vscode\rime_projs\librime\build\bin\Release\rime.dll"
    
    try:
        rime = RimeDllWrapper(dll_path=dll_path)
        print(f"✓ 成功加载: {rime.dll_path}")
        rime.initialize()
        # ... 其他操作
        rime.finalize()
    except Exception as e:
        print(f"错误: {e}")


def example_compare_with_console():
    """对比 DLL 包装器和控制台包装器"""
    print("=" * 60)
    print("DLL 包装器 vs 控制台包装器")
    print("=" * 60)
    
    print("\nRimeDllWrapper (DLL 包装器):")
    print("  优点:")
    print("    - 更灵活，可以直接访问所有 API")
    print("    - 性能更好（无进程间通信）")
    print("    - 可以精确控制 Rime 的行为")
    print("  缺点:")
    print("    - 需要完整实现 API 定义（当前为简化版本）")
    print("    - 实现复杂度较高")
    print("    - 需要处理 C 结构体和内存管理")
    
    print("\nRimeConsoleWrapper (控制台包装器):")
    print("  优点:")
    print("    - 实现简单，易于使用")
    print("    - 功能完整（使用官方控制台程序）")
    print("    - 不需要处理 C API 细节")
    print("  缺点:")
    print("    - 需要进程间通信")
    print("    - 性能稍差")
    print("    - 输出解析可能复杂")
    
    print("\n建议:")
    print("  - 快速测试和开发: 使用 RimeConsoleWrapper")
    print("  - 生产环境或需要高性能: 完善 RimeDllWrapper 实现")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RimeDllWrapper 使用示例')
    parser.add_argument('--example', type=str, 
                       choices=['basic', 'custom', 'compare'],
                       default='basic',
                       help='要运行的示例')
    
    args = parser.parse_args()
    
    if args.example == 'basic':
        example_basic_usage()
    elif args.example == 'custom':
        example_with_custom_path()
    elif args.example == 'compare':
        example_compare_with_console()

