"""
TinyBERT 调用示例
TinyBERT是华为诺亚方舟实验室提出的BERT模型压缩方法
"""

from transformers import BertTokenizer, BertModel
import torch


def load_tinybert_model(model_name='huawei-noah/TinyBERT_General_4L_312D'):
    """
    加载TinyBERT模型和分词器
    
    Args:
        model_name: 模型名称，默认为通用4层312维的TinyBERT模型
        
    Returns:
        tokenizer: 分词器
        model: TinyBERT模型
    """
    print(f"正在加载模型: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # 设置为评估模式
    print("模型加载完成！")
    return tokenizer, model


def encode_text(tokenizer, model, text, return_cls=True):
    """
    对文本进行编码并获取BERT表示
    
    Args:
        tokenizer: 分词器
        model: BERT模型
        text: 输入文本
        return_cls: 是否返回[CLS]标记的表示（用于句子级任务）
        
    Returns:
        文本的向量表示
    """
    # 对文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # 推理（不计算梯度）
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state
    
    if return_cls:
        # 返回[CLS]标记的表示（第一个token）
        return last_hidden_states[:, 0, :]
    else:
        # 返回所有token的表示
        return last_hidden_states


def main():
    """主函数：演示如何使用TinyBERT"""
    
    # 1. 加载模型
    print("=" * 50)
    print("步骤1: 加载TinyBERT模型")
    print("=" * 50)
    tokenizer, model = load_tinybert_model()
    
    # 2. 准备输入文本
    print("\n" + "=" * 50)
    print("步骤2: 准备输入文本")
    print("=" * 50)
    texts = [
        "这是一个TinyBERT使用示例。",
        "TinyBERT是一个轻量级的BERT模型。",
        "它通过知识蒸馏技术压缩了BERT模型。"
    ]
    
    # 3. 对文本进行编码
    print("\n" + "=" * 50)
    print("步骤3: 对文本进行编码")
    print("=" * 50)
    for i, text in enumerate(texts, 1):
        print(f"\n文本 {i}: {text}")
        
        # 获取句子表示（使用[CLS]标记）
        sentence_embedding = encode_text(tokenizer, model, text, return_cls=True)
        print(f"句子向量维度: {sentence_embedding.shape}")
        print(f"句子向量（前10个值）: {sentence_embedding[0][:10].tolist()}")
        
        # 获取所有token的表示
        token_embeddings = encode_text(tokenizer, model, text, return_cls=False)
        print(f"Token向量维度: {token_embeddings.shape}")
    
    # 4. 计算句子相似度示例
    print("\n" + "=" * 50)
    print("步骤4: 计算句子相似度")
    print("=" * 50)
    from torch.nn.functional import cosine_similarity
    
    text1 = "TinyBERT是一个轻量级模型"
    text2 = "TinyBERT通过知识蒸馏压缩BERT"
    
    emb1 = encode_text(tokenizer, model, text1, return_cls=True)
    emb2 = encode_text(tokenizer, model, text2, return_cls=True)
    
    similarity = cosine_similarity(emb1, emb2).item()
    print(f"文本1: {text1}")
    print(f"文本2: {text2}")
    print(f"余弦相似度: {similarity:.4f}")


if __name__ == "__main__":
    main()

