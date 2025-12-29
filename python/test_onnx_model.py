"""
测试转换后的ONNX模型是否正常工作
"""

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer
import os


def test_onnx_model(model_path='onnx_models/bert-base-chinese.onnx', 
                   tokenizer_path='onnx_models'):
    """
    测试ONNX模型
    
    Args:
        model_path: ONNX模型文件路径
        tokenizer_path: 分词器配置路径
    """
    print("=" * 60)
    print("测试ONNX模型")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行: python convert_to_onnx.py")
        return False
    
    # 加载分词器
    print(f"\n加载分词器: {tokenizer_path}")
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("✓ 分词器加载成功")
    except Exception as e:
        print(f"⚠ 警告: 无法从指定路径加载分词器: {e}")
        print("尝试使用在线分词器...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建ONNX Runtime会话
    print(f"\n加载ONNX模型: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✓ ONNX模型加载成功")
    except Exception as e:
        print(f"✗ ONNX模型加载失败: {e}")
        return False
    
    # 显示模型信息
    print(f"\n模型信息:")
    print(f"  输入数量: {len(session.get_inputs())}")
    for i, input_info in enumerate(session.get_inputs()):
        print(f"    输入 {i+1}: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
    
    print(f"  输出数量: {len(session.get_outputs())}")
    for i, output_info in enumerate(session.get_outputs()):
        print(f"    输出 {i+1}: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
    
    # 测试句子
    test_sentences = [
        "这是一个测试",
        "各个国家有各个国家的国歌",
        "你好世界"
    ]
    
    print(f"\n" + "=" * 60)
    print("运行推理测试")
    print("=" * 60)
    
    for sentence in test_sentences:
        print(f"\n测试句子: {sentence}")
        
        # 分词
        inputs = tokenizer(
            sentence,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        
        # 运行推理
        try:
            outputs = session.run(
                None,
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            )
            
            logits = outputs[0]
            print(f"  ✓ 推理成功")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits范围: [{logits.min():.4f}, {logits.max():.4f}]")
            
            # 计算一些统计信息
            vocab_size = logits.shape[-1]
            print(f"  词汇表大小: {vocab_size}")
            
            # 获取每个位置最可能的token
            predicted_token_ids = np.argmax(logits, axis=-1)[0]
            print(f"  预测的token IDs (前10个): {predicted_token_ids[:10].tolist()}")
            
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")
            return False
    
    print(f"\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print(f"\n模型可以正常使用，可以在C语言中调用。")
    
    return True


def compare_with_pytorch(model_path='onnx_models/bert-base-chinese.onnx',
                         tokenizer_path='onnx_models',
                         test_sentence="这是一个测试"):
    """
    比较ONNX模型和PyTorch模型的输出是否一致
    """
    print("=" * 60)
    print("比较ONNX和PyTorch模型输出")
    print("=" * 60)
    
    from transformers import BertForMaskedLM
    import torch
    
    # 加载PyTorch模型
    print(f"\n加载PyTorch模型...")
    pytorch_model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    pytorch_model.eval()
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path if os.path.exists(tokenizer_path) else 'bert-base-chinese')
    
    # 准备输入
    inputs = tokenizer(
        test_sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # PyTorch推理
    print(f"\n运行PyTorch推理...")
    with torch.no_grad():
        pytorch_outputs = pytorch_model(**inputs)
        pytorch_logits = pytorch_outputs.logits.numpy()
    
    # ONNX推理
    print(f"运行ONNX推理...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    onnx_inputs = {
        'input_ids': inputs['input_ids'].numpy().astype(np.int64),
        'attention_mask': inputs['attention_mask'].numpy().astype(np.int64)
    }
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_logits = onnx_outputs[0]
    
    # 比较结果
    print(f"\n比较结果:")
    print(f"  PyTorch logits shape: {pytorch_logits.shape}")
    print(f"  ONNX logits shape: {onnx_logits.shape}")
    
    # 计算差异
    diff = np.abs(pytorch_logits - onnx_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    
    # 检查是否在可接受范围内（通常ONNX转换会有微小的数值差异）
    if max_diff < 1e-5:
        print(f"  ✓ 输出完全一致（差异 < 1e-5）")
    elif max_diff < 1e-3:
        print(f"  ✓ 输出基本一致（差异 < 1e-3，可接受）")
    else:
        print(f"  ⚠ 输出存在较大差异（差异 >= 1e-3）")
    
    return max_diff < 1e-3


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试ONNX模型')
    parser.add_argument('--model', type=str, default='onnx_models/bert-base-chinese.onnx',
                       help='ONNX模型路径')
    parser.add_argument('--tokenizer', type=str, default='onnx_models',
                       help='分词器路径')
    parser.add_argument('--compare', action='store_true',
                       help='与PyTorch模型比较输出')
    
    args = parser.parse_args()
    
    # 基本测试
    success = test_onnx_model(args.model, args.tokenizer)
    
    # 可选：与PyTorch模型比较
    if args.compare and success:
        compare_with_pytorch(args.model, args.tokenizer)

