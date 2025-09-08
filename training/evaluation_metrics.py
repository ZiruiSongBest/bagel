#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一生成模型的评估指标

这个模块提供了评估统一生成模型（文本+图像）的各种指标，包括：
1. 文本生成质量指标 (BLEU, ROUGE, METEOR等)
2. 图像生成质量指标 (FID, IS, CLIP Score等)
3. 多模态一致性指标
4. 定性评估工具
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Union, Optional, Any, Tuple
from PIL import Image
import logging
from tqdm import tqdm
from pathlib import Path

# 文本评估指标
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from rouge_score import rouge_scorer
    import evaluate
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK或rouge_score未安装，某些文本评估指标将不可用")

# 图像评估指标
try:
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm
    import clip
    VISION_METRICS_AVAILABLE = True
except ImportError:
    VISION_METRICS_AVAILABLE = False
    logging.warning("图像评估依赖库未安装，某些图像评估指标将不可用")


class TextEvaluator:
    """文本生成评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            try:
                self.meteor_metric = evaluate.load("meteor")
                self.bertscore_metric = evaluate.load("bertscore")
            except Exception as e:
                self.logger.warning(f"加载Hugging Face评估指标失败: {e}")
                self.meteor_metric = None
                self.bertscore_metric = None
        else:
            self.rouge_scorer = None
            self.meteor_metric = None
            self.bertscore_metric = None
    
    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """计算BLEU分数"""
        if not NLTK_AVAILABLE:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        # 分词处理
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split() for ref in refs] for refs in references]
        
        # 计算不同n-gram的BLEU
        bleu_1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1.0, 0, 0, 0))
        bleu_2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数"""
        if not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        
        return {
            "rouge1": np.mean(rouge_scores["rouge1"]),
            "rouge2": np.mean(rouge_scores["rouge2"]),
            "rougeL": np.mean(rouge_scores["rougeL"]),
        }
    
    def compute_meteor(self, predictions: List[str], references: List[str]) -> float:
        """计算METEOR分数"""
        if not self.meteor_metric:
            return 0.0
        
        try:
            result = self.meteor_metric.compute(predictions=predictions, references=references)
            return result["meteor"]
        except Exception as e:
            self.logger.warning(f"计算METEOR时出错: {e}")
            return 0.0
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BERTScore"""
        if not self.bertscore_metric:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        try:
            results = self.bertscore_metric.compute(
                predictions=predictions, 
                references=references, 
                model_type="microsoft/deberta-xlarge-mnli"
            )
            return {
                "bertscore_precision": np.mean(results["precision"]),
                "bertscore_recall": np.mean(results["recall"]),
                "bertscore_f1": np.mean(results["f1"]),
            }
        except Exception as e:
            self.logger.warning(f"计算BERTScore时出错: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    def evaluate_all(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算所有文本评估指标"""
        metrics = {}
        
        # BLEU (需要转换references格式)
        ref_lists = [[ref] for ref in references]
        metrics.update(self.compute_bleu(predictions, ref_lists))
        
        # ROUGE
        metrics.update(self.compute_rouge(predictions, references))
        
        # METEOR
        metrics["meteor"] = self.compute_meteor(predictions, references)
        
        # BERTScore
        metrics.update(self.compute_bertscore(predictions, references))
        
        return metrics


class ImageEvaluator:
    """图像生成评估器"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        if VISION_METRICS_AVAILABLE:
            # 加载预训练模型
            self._load_models()
        else:
            self.inception_model = None
            self.clip_model = None
            self.clip_preprocess = None
    
    def _load_models(self):
        """加载评估所需的模型"""
        try:
            # Inception v3 for FID
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = torch.nn.Identity()  # 移除最后的分类层
            self.inception_model.eval()
            self.inception_model.to(self.device)
            
            # CLIP for semantic similarity
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            
            self.logger.info("图像评估模型加载完成")
        except Exception as e:
            self.logger.warning(f"加载图像评估模型失败: {e}")
            self.inception_model = None
            self.clip_model = None
            self.clip_preprocess = None
    
    def compute_fid(self, real_images: List[Image.Image], generated_images: List[Image.Image]) -> float:
        """计算FID (Fréchet Inception Distance)"""
        if not self.inception_model:
            return 0.0
        
        def get_features(images):
            features = []
            for img in tqdm(images, desc="提取特征"):
                # 预处理图像
                img_tensor = torch.tensor(np.array(img.resize((299, 299)))).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    feature = self.inception_model(img_tensor)
                features.append(feature.cpu().numpy())
            return np.concatenate(features, axis=0)
        
        try:
            # 提取特征
            real_features = get_features(real_images)
            gen_features = get_features(generated_images)
            
            # 计算均值和协方差
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
            
            # 计算FID
            diff = mu1 - mu2
            covmean = sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid)
            
        except Exception as e:
            self.logger.warning(f"计算FID时出错: {e}")
            return 0.0
    
    def compute_clip_score(self, images: List[Image.Image], texts: List[str]) -> float:
        """计算CLIP Score (图像和文本的相似度)"""
        if not self.clip_model:
            return 0.0
        
        try:
            scores = []
            for img, text in tqdm(zip(images, texts), desc="计算CLIP Score"):
                # 预处理图像和文本
                image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                text_input = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    # 编码
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text_input)
                    
                    # 归一化
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # 计算相似度
                    similarity = torch.cosine_similarity(image_features, text_features)
                    scores.append(similarity.item())
            
            return np.mean(scores)
            
        except Exception as e:
            self.logger.warning(f"计算CLIP Score时出错: {e}")
            return 0.0
    
    def evaluate_all(
        self, 
        generated_images: List[Image.Image], 
        reference_images: Optional[List[Image.Image]] = None,
        prompt_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """计算所有图像评估指标"""
        metrics = {}
        
        # FID (需要参考图像)
        if reference_images:
            metrics["fid"] = self.compute_fid(reference_images, generated_images)
        
        # CLIP Score (需要对应的文本)
        if prompt_texts:
            metrics["clip_score"] = self.compute_clip_score(generated_images, prompt_texts)
        
        return metrics


class UnifiedEvaluator:
    """统一生成模型的综合评估器"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.text_evaluator = TextEvaluator()
        self.image_evaluator = ImageEvaluator(device)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_generation_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        评估统一生成的结果
        
        Args:
            results: 生成结果列表，每个元素包含:
                - input_sequence: 输入序列
                - target_sequence: 目标序列
                - generated_sequence: 生成序列
                - input_types: 输入类型列表
                - target_types: 目标类型列表
                - generated_types: 生成类型列表
            output_dir: 结果保存目录
        
        Returns:
            评估指标字典
        """
        self.logger.info("开始评估统一生成结果...")
        
        # 分离文本和图像结果
        text_predictions = []
        text_references = []
        generated_images = []
        reference_images = []
        image_prompts = []
        
        for result in results:
            target_seq = result['target_sequence']
            generated_seq = result['generated_sequence']
            target_types = result['target_types']
            generated_types = result.get('generated_types', target_types)
            
            for i, (target_item, target_type) in enumerate(zip(target_seq, target_types)):
                if target_type == 'text':
                    text_references.append(target_item)
                    if i < len(generated_seq):
                        gen_item = generated_seq[i]
                        if isinstance(gen_item, str):
                            text_predictions.append(gen_item)
                        else:
                            text_predictions.append("")  # 生成失败的情况
                    else:
                        text_predictions.append("")
                
                elif target_type == 'image':
                    if isinstance(target_item, Image.Image):
                        reference_images.append(target_item)
                    
                    if i < len(generated_seq):
                        gen_item = generated_seq[i]
                        if isinstance(gen_item, Image.Image):
                            generated_images.append(gen_item)
                            
                            # 尝试获取对应的文本提示
                            input_seq = result['input_sequence']
                            input_types = result['input_types']
                            prompt_texts = [item for item, itype in zip(input_seq, input_types) if itype == 'text']
                            if prompt_texts:
                                image_prompts.append(prompt_texts[-1])  # 使用最后一个文本作为提示
                            else:
                                image_prompts.append("")
        
        # 计算评估指标
        evaluation_results = {}
        
        # 文本评估
        if text_predictions and text_references:
            self.logger.info(f"评估 {len(text_predictions)} 个文本生成结果...")
            text_metrics = self.text_evaluator.evaluate_all(text_predictions, text_references)
            evaluation_results.update({f"text_{k}": v for k, v in text_metrics.items()})
        
        # 图像评估
        if generated_images:
            self.logger.info(f"评估 {len(generated_images)} 个图像生成结果...")
            image_metrics = self.image_evaluator.evaluate_all(
                generated_images=generated_images,
                reference_images=reference_images if reference_images else None,
                prompt_texts=image_prompts if image_prompts else None
            )
            evaluation_results.update({f"image_{k}": v for k, v in image_metrics.items()})
        
        # 计算综合指标
        evaluation_results["total_samples"] = len(results)
        evaluation_results["text_samples"] = len(text_predictions)
        evaluation_results["image_samples"] = len(generated_images)
        
        # 保存评估结果
        if output_dir:
            self._save_evaluation_results(evaluation_results, output_dir)
            self._save_detailed_results(results, output_dir)
        
        self.logger.info("评估完成!")
        return evaluation_results
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON格式的结果
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存可读格式的报告
        with open(os.path.join(output_dir, "evaluation_report.txt"), 'w') as f:
            f.write("统一生成模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("总览:\n")
            f.write(f"  总样本数: {results.get('total_samples', 0)}\n")
            f.write(f"  文本样本数: {results.get('text_samples', 0)}\n")
            f.write(f"  图像样本数: {results.get('image_samples', 0)}\n\n")
            
            f.write("文本生成指标:\n")
            for key, value in results.items():
                if key.startswith('text_'):
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n图像生成指标:\n")
            for key, value in results.items():
                if key.startswith('image_'):
                    f.write(f"  {key}: {value:.4f}\n")
        
        self.logger.info(f"评估结果已保存到 {output_dir}")
    
    def _save_detailed_results(self, results: List[Dict[str, Any]], output_dir: str):
        """保存详细的生成结果"""
        detailed_dir = os.path.join(output_dir, "detailed_results")
        os.makedirs(detailed_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            sample_dir = os.path.join(detailed_dir, f"sample_{i:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存文本结果
            with open(os.path.join(sample_dir, "result.json"), 'w') as f:
                # 创建可序列化的结果副本
                serializable_result = {
                    "input_types": result['input_types'],
                    "target_types": result['target_types'],
                    "generated_types": result.get('generated_types', result['target_types']),
                }
                
                # 处理序列中的文本
                for seq_name in ['input_sequence', 'target_sequence', 'generated_sequence']:
                    if seq_name in result:
                        serializable_result[seq_name] = []
                        for item in result[seq_name]:
                            if isinstance(item, str):
                                serializable_result[seq_name].append(item)
                            else:
                                serializable_result[seq_name].append(f"[IMAGE: {type(item).__name__}]")
                
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            # 保存图像
            for seq_name in ['input_sequence', 'target_sequence', 'generated_sequence']:
                if seq_name in result:
                    for j, item in enumerate(result[seq_name]):
                        if isinstance(item, Image.Image):
                            item.save(os.path.join(sample_dir, f"{seq_name}_{j}.png"))


def create_sample_evaluation_data():
    """创建示例评估数据（用于测试）"""
    return [
        {
            "input_sequence": ["生成一张猫的图片"],
            "target_sequence": [Image.new('RGB', (512, 512), 'white')],  # 示例图像
            "generated_sequence": [Image.new('RGB', (512, 512), 'gray')],  # 示例生成图像
            "input_types": ["text"],
            "target_types": ["image"],
        },
        {
            "input_sequence": [Image.new('RGB', (512, 512), 'blue')],  # 示例输入图像
            "target_sequence": ["这是一张蓝色的图片"],
            "generated_sequence": ["这是一张图片"],
            "input_types": ["image"],
            "target_types": ["text"],
        }
    ]


if __name__ == "__main__":
    # 测试评估器
    logging.basicConfig(level=logging.INFO)
    
    # 创建评估器
    evaluator = UnifiedEvaluator()
    
    # 创建示例数据
    sample_data = create_sample_evaluation_data()
    
    # 运行评估
    results = evaluator.evaluate_generation_results(
        results=sample_data,
        output_dir="./test_evaluation_output"
    )
    
    print("评估结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
