#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成训练代码测试脚本

这个脚本用于测试统一生成训练框架的各个组件，包括：
1. 数据处理器测试
2. 训练器配置测试  
3. 模拟训练步骤测试
4. 评估指标测试
"""

import os
import sys
import json
import torch
import tempfile
from PIL import Image
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.unified_data_processor import UnifiedGenerationDataset, UnifiedTrainingExample
from training.unified_trainer import UnifiedTrainingConfig, UnifiedTrainer
from training.evaluation_metrics import UnifiedEvaluator


def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    # 创建测试图像
    test_images = []
    for i in range(3):
        img = Image.new('RGB', (256, 256), color=(i*80, 100, 150))
        img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
        img.save(img_path)
        test_images.append(img_path)
    
    # 创建测试数据
    test_data = [
        # 对话格式
        {
            "conversations": [
                {
                    "role": "user",
                    "type": "text",
                    "content": "请生成一张蓝色的图片"
                },
                {
                    "role": "assistant", 
                    "type": "image",
                    "image_path": test_images[0]
                }
            ],
            "metadata": {"task_type": "text_to_image"}
        },
        
        # 图像描述格式
        {
            "image_path": test_images[1],
            "caption": "这是一张测试图片，颜色偏绿色。",
            "metadata": {"task_type": "image_to_text"}
        },
        
        # 直接格式
        {
            "input_sequence": [
                {"type": "text", "content": "编辑这张图片：改变颜色"},
                {"type": "image", "image_path": test_images[2]}
            ],
            "target_sequence": [
                {"type": "image", "image_path": test_images[0]}
            ],
            "metadata": {"task_type": "image_editing"}
        }
    ]
    
    # 保存测试数据文件
    train_data_path = os.path.join(temp_dir, "train_data.jsonl")
    with open(train_data_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return temp_dir, train_data_path, test_images


def test_data_processor():
    """测试数据处理器"""
    print("\n" + "="*50)
    print("测试数据处理器")
    print("="*50)
    
    try:
        # 创建模拟的tokenizer和transforms
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4, 5]  # 简单的模拟编码
            
        class MockTransform:
            def __call__(self, image):
                # 返回模拟的tensor
                return torch.randn(3, 224, 224)
            
            def resize_transform(self, image):
                return torch.randn(3, 224, 224)
        
        tokenizer = MockTokenizer()
        vae_transform = MockTransform()
        vit_transform = MockTransform()
        new_token_ids = {
            'bos_token_id': 1,
            'eos_token_id': 2, 
            'start_of_image': 3,
            'end_of_image': 4
        }
        
        # 创建测试数据
        temp_dir, train_data_path, test_images = create_test_data()
        
        # 创建数据集
        dataset = UnifiedGenerationDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            max_sequence_length=512,
            max_image_tokens=256,
        )
        
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试获取样本
        for i in range(min(2, len(dataset))):
            sample = dataset[i]
            print(f"样本 {i}:")
            print(f"  序列长度: {sample['sequence_length']}")
            print(f"  文本tokens: {len(sample['packed_text_ids'])}")
            print(f"  输入长度: {sample['input_length']}")
            print(f"  目标长度: {sample['target_length']}")
            print(f"  元数据: {sample['metadata']}")
        
        print("✅ 数据处理器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_config():
    """测试训练配置"""
    print("\n" + "="*50)
    print("测试训练配置")
    print("="*50)
    
    try:
        config = UnifiedTrainingConfig(
            train_data_path="/fake/path/train.jsonl",
            val_data_path="/fake/path/val.jsonl",
            output_dir="./test_outputs",
            batch_size=1,
            gradient_accumulation_steps=4,
            num_epochs=2,
            learning_rate=1e-5,
            text_loss_weight=1.0,
            image_loss_weight=1.0,
        )
        
        print("✅ 训练配置创建成功")
        print("配置参数:")
        config_dict = config.to_dict()
        for key, value in list(config_dict.items())[:10]:  # 只显示前10个参数
            print(f"  {key}: {value}")
        print("  ...")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练配置测试失败: {e}")
        return False


def test_evaluation_metrics():
    """测试评估指标"""
    print("\n" + "="*50)
    print("测试评估指标")
    print("="*50)
    
    try:
        # 创建评估器
        evaluator = UnifiedEvaluator()
        
        # 创建模拟的评估数据
        test_results = [
            {
                "input_sequence": ["生成一张猫的图片"],
                "target_sequence": ["这是一张可爱的小猫图片"],
                "generated_sequence": ["这是一张猫的图片"],
                "input_types": ["text"],
                "target_types": ["text"],
            },
            {
                "input_sequence": [Image.new('RGB', (256, 256), 'blue')],
                "target_sequence": [Image.new('RGB', (256, 256), 'red')],
                "generated_sequence": [Image.new('RGB', (256, 256), 'green')],
                "input_types": ["image"],
                "target_types": ["image"],
            }
        ]
        
        # 运行评估
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = evaluator.evaluate_generation_results(
                results=test_results,
                output_dir=temp_dir
            )
        
        print("✅ 评估指标计算成功")
        print("评估结果:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_training_step():
    """测试模拟训练步骤"""
    print("\n" + "="*50)
    print("测试模拟训练步骤")
    print("="*50)
    
    try:
        # 这里只测试训练器的初始化，不进行真实的模型训练
        # 因为需要完整的BAGEL模型
        
        class MockModel:
            def __init__(self):
                self.device = torch.device('cpu')
                self.linear = torch.nn.Linear(10, 10)  # 添加一个实际的层
                
            def parameters(self):
                return self.linear.parameters()
                
            def named_parameters(self):
                return self.linear.named_parameters()
                
            def to(self, device):
                self.linear.to(device)
                return self
                
            def train(self):
                self.linear.train()
                
            def eval(self):
                self.linear.eval()
        
        class MockVAEModel:
            def to(self, device):
                return self
                
            def encode(self, x):
                return torch.randn(1, 16, 32, 32)
        
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4, 5]
        
        class MockTransform:
            def __call__(self, image):
                return torch.randn(3, 224, 224)
                
            def resize_transform(self, image):
                return torch.randn(3, 512, 512)
        
        # 创建模拟组件
        model = MockModel()
        vae_model = MockVAEModel()
        tokenizer = MockTokenizer()
        vae_transform = MockTransform()
        vit_transform = MockTransform()
        new_token_ids = {
            'bos_token_id': 1,
            'eos_token_id': 2, 
            'start_of_image': 3,
            'end_of_image': 4
        }
        
        config = UnifiedTrainingConfig(
            train_data_path="/fake/path",
            output_dir="./test_outputs",
            batch_size=1,
            num_epochs=1,
        )
        
        # 创建训练器（不进行实际训练）
        trainer = UnifiedTrainer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            config=config,
        )
        
        print("✅ 训练器初始化成功")
        print(f"优化器: {type(trainer.optimizer).__name__}")
        print(f"设备: {trainer.device}")
        print(f"全局步数: {trainer.global_step}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 统一生成训练代码测试")
    print("="*60)
    
    # 运行所有测试
    tests = [
        ("数据处理器", test_data_processor),
        ("训练配置", test_training_config),
        ("评估指标", test_evaluation_metrics),
        ("模拟训练步骤", test_mock_training_step),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results[test_name] = False
    
    # 总结结果
    print("\n" + "="*60)
    print("🏁 测试结果总结")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✅ 通过" if passed_test else "❌ 失败"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！训练代码基本功能正常。")
        print("\n接下来你可以:")
        print("1. 准备真实的训练数据")
        print("2. 运行 train_unified_generation.py 开始训练")
        print("3. 参考 training/README_training.md 了解更多细节")
    else:
        print("⚠️  部分测试失败，请检查代码或依赖库。")
    
    print("="*60)


if __name__ == "__main__":
    main()
