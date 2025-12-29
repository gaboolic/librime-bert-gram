"""
将BERT模型转换为ONNX格式，以便C语言调用
支持bert-base-chinese模型的转换
"""

import torch
from transformers import BertForMaskedLM, BertTokenizer
import os


def convert_bert_to_onnx(
    model_name='bert-base-chinese',
    output_dir='onnx_models',
    opset_version=14,
    dynamic_axes=True
):
    """
    将BERT模型转换为ONNX格式
    
    Args:
        model_name: 模型名称，默认为'bert-base-chinese'
        output_dir: 输出目录
        opset_version: ONNX opset版本（建议11-14）
        dynamic_axes: 是否支持动态输入长度
    """
    print(f"正在加载模型: {model_name}")
    print("=" * 60)
    
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    print("模型加载完成！")
    print(f"模型配置:")
    print(f"  - 词汇表大小: {tokenizer.vocab_size}")
    print(f"  - 最大序列长度: {model.config.max_position_embeddings}")
    print(f"  - 隐藏层大小: {model.config.hidden_size}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备示例输入
    # 使用一个示例句子来创建输入
    example_text = "这是一个示例文本"
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    print(f"\n示例输入:")
    print(f"  - 文本: {example_text}")
    print(f"  - Input IDs shape: {input_ids.shape}")
    print(f"  - Attention mask shape: {attention_mask.shape}")
    
    # 定义输入和输出的动态轴
    if dynamic_axes:
        dynamic_axes_config = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    else:
        dynamic_axes_config = None
    
    # 导出ONNX模型
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}.onnx")
    
    print(f"\n开始转换为ONNX格式...")
    print(f"输出路径: {output_path}")
    print(f"ONNX Opset版本: {opset_version}")
    print(f"动态轴: {dynamic_axes}")
    
    with torch.no_grad():
        torch.onnx.export(
            model,                                    # 模型
            (input_ids, attention_mask),             # 模型输入（元组）
            output_path,                              # 输出路径
            input_names=['input_ids', 'attention_mask'],  # 输入名称
            output_names=['logits'],                  # 输出名称
            dynamic_axes=dynamic_axes_config,         # 动态轴配置
            opset_version=opset_version,             # ONNX opset版本
            do_constant_folding=True,                 # 常量折叠优化
            verbose=False
        )
    
    print(f"\n✓ ONNX模型转换完成！")
    print(f"  文件路径: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # 验证ONNX模型
    print(f"\n验证ONNX模型...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过！")
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  - IR版本: {onnx_model.ir_version}")
        print(f"  - 生产者: {onnx_model.producer_name} {onnx_model.producer_version}")
        print(f"  - 输入数量: {len(onnx_model.graph.input)}")
        print(f"  - 输出数量: {len(onnx_model.graph.output)}")
        
    except ImportError:
        print("⚠ 警告: 未安装onnx包，跳过验证")
        print("  可以运行: pip install onnx")
    except Exception as e:
        print(f"⚠ 警告: ONNX模型验证失败: {e}")
    
    # 保存分词器配置（C语言调用时需要）
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    vocab_path = os.path.join(output_dir, "vocab.txt")
    
    try:
        # 保存tokenizer配置
        tokenizer.save_pretrained(output_dir)
        print(f"\n✓ 分词器配置已保存到: {output_dir}")
        print(f"  - tokenizer_config.json")
        print(f"  - vocab.txt")
    except Exception as e:
        print(f"⚠ 警告: 保存分词器配置失败: {e}")
    
    print(f"\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"\n下一步：")
    print(f"1. 使用ONNX Runtime C API加载模型")
    print(f"2. 参考文档: https://onnxruntime.ai/docs/api/c/")
    print(f"3. 模型文件: {output_path}")
    print(f"4. 分词器文件: {output_dir}/")
    
    return output_path


def convert_bert_encoder_only(
    model_name='bert-base-chinese',
    output_dir='onnx_models',
    opset_version=14
):
    """
    只转换BERT的编码器部分（不包含MLM head）
    这个版本更轻量，适合只需要获取句子嵌入的场景
    
    Args:
        model_name: 模型名称
        output_dir: 输出目录
        opset_version: ONNX opset版本
    """
    from transformers import BertModel
    
    print(f"正在加载BERT编码器模型: {model_name}")
    print("=" * 60)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    print("模型加载完成！")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备示例输入
    example_text = "这是一个示例文本"
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # 动态轴配置
    dynamic_axes_config = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
        'pooler_output': {0: 'batch_size'}
    }
    
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_encoder.onnx")
    
    print(f"\n开始转换为ONNX格式...")
    print(f"输出路径: {output_path}")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state', 'pooler_output'],
            dynamic_axes=dynamic_axes_config,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"\n✓ ONNX编码器模型转换完成！")
    print(f"  文件路径: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将BERT模型转换为ONNX格式')
    parser.add_argument('--model', type=str, default='bert-base-chinese',
                       help='模型名称 (默认: bert-base-chinese)')
    parser.add_argument('--output', type=str, default='onnx_models',
                       help='输出目录 (默认: onnx_models)')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset版本 (默认: 14)')
    parser.add_argument('--encoder-only', action='store_true',
                       help='只转换编码器部分（不包含MLM head）')
    parser.add_argument('--no-dynamic', action='store_true',
                       help='不使用动态轴（固定输入长度）')
    
    args = parser.parse_args()
    
    if args.encoder_only:
        convert_bert_encoder_only(
            model_name=args.model,
            output_dir=args.output,
            opset_version=args.opset
        )
    else:
        convert_bert_to_onnx(
            model_name=args.model,
            output_dir=args.output,
            opset_version=args.opset,
            dynamic_axes=not args.no_dynamic
        )

