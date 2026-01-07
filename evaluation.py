"""
audio2face测试集推理与评估脚本 - 更新版
根据补充信息调整数据格式处理
"""

import torch
import numpy as np
import librosa
import json
import os
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，无需图形界面
import matplotlib.pyplot as plt
import pandas as pd

# 导入模型相关模块
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from ModelDecoder import TransformerStackedDecoder
from AudioDataset import pack_exp
from PreProcess import ctrl_expressions as ctrl_expressions_list

# 设置日志
def setup_logging(log_dir: str = "./logs"):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class Audio2FaceTester:
    def __init__(
        self, 
        model_weights_path: str,
        wav2vec_path: str = "./wav2vec2-base-960h",
        device: str = None,
        output_dir: str = "./test_results",
        fps: int = 25  # 帧率
    ):
        """
        初始化测试器
        
        Args:
            model_weights_path: 解码器模型权重路径
            wav2vec_path: wav2vec2模型路径
            device: 设备 ('cuda' 或 'cpu')
            output_dir: 输出目录
            fps: 输出帧率
        """
        self.logger = setup_logging()
        self.logger.info("初始化Audio2Face测试器...")
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.logger.info(f"使用设备: {self.device}")
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.pred_json_dir = self.output_dir / "pred_json"
        self.visualization_dir = self.output_dir / "visualization"
        self.metrics_dir = self.output_dir / "metrics"
        self.summary_dir = self.output_dir / "summary"
        
        for dir_path in [self.pred_json_dir, self.visualization_dir, 
                        self.metrics_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 帧率设置
        self.fps = fps
        
        # 加载模型
        self._load_models(model_weights_path, wav2vec_path)
        
        # 性能统计
        self.inference_times = []
        self.audio_lengths = []
        
    def _load_models(self, model_weights_path: str, wav2vec_path: str):
        """加载所有需要的模型"""
        try:
            self.logger.info("加载wav2vec2模型...")
            self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(self.device)
            self.wav2vec_model.eval()
            
            self.logger.info("加载Transformer解码器...")
            self.decoder = TransformerStackedDecoder(
                input_dim=768,
                output_dim=136,  # 假设模型输出136维
                num_heads=16,
                num_layers=9
            ).to(self.device)
            
            # 加载权重
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.decoder.load_state_dict(state_dict)
            self.decoder.eval()
            
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _process_audio_segment(self, audio_segment: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        处理单个音频片段
        
        Returns:
            预测的表情参数序列 (T, 136)
        """
        # 确保音频长度正确
        target_length = sr * 5  # 5秒
        if len(audio_segment) < target_length:
            audio_segment = np.pad(
                audio_segment, 
                (0, target_length - len(audio_segment)), 
                mode='constant', 
                constant_values=0
            )
        
        # 提取特征
        with torch.no_grad():
            inputs = self.processor(
                audio_segment, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            # 移动到设备
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else None
            
            # wav2vec2前向传播
            wav_features = self.wav2vec_model(
                input_values, 
                attention_mask=attention_mask
            ).last_hidden_state
            
            # 解码器前向传播
            pred = self.decoder(wav_features)
            pred = pred.squeeze(0).cpu().numpy()
            
        return pred
    
    def merge_segments(self, segments: List[np.ndarray]) -> List[List[float]]:
        """
        合并处理后的音频片段
        
        Args:
            segments: 各个片段的预测结果列表
        
        Returns:
            合并后的完整预测序列
        """
        merged = []
        for segment in segments:
            for frame in segment:
                merged.append(frame.tolist())
        
        return merged
    
    def inference_single_audio(
        self, 
        wav_path: str, 
        segment_length: int = 5
    ) -> Tuple[List[List[float]], Dict[str, float]]:
        """
        对单个音频文件进行推理
        
        Returns:
            pred_sequence: 预测的表情参数序列 (face_pred)
            metrics: 推理性能指标
        """
        start_time = time.time()
        
        try:
            # 加载音频
            self.logger.info(f"处理音频: {wav_path}")
            wave_data, sr = librosa.load(wav_path, sr=16000)
            audio_duration = len(wave_data) / sr
            
            # 计算总帧数
            total_frames = int(audio_duration * self.fps)
            
            # 分割音频
            segment_len = sr * segment_length
            segment_num = int(np.ceil(len(wave_data) / segment_len))
            
            self.logger.info(f"音频时长: {audio_duration:.2f}s, 分割为 {segment_num} 个片段")
            
            # 处理每个片段
            segments = []
            for i in range(segment_num):
                self.logger.debug(f"处理片段 {i+1}/{segment_num}")
                start_point = segment_len * i
                end_point = min(start_point + segment_len, len(wave_data))
                wav_segment = wave_data[start_point:end_point]
                
                # 处理片段
                pred_segment = self._process_audio_segment(wav_segment, sr)
                segments.append(pred_segment)
            
            # 合并片段
            merged_pred = self.merge_segments(segments)
            
            # 调整帧数匹配音频时长
            if len(merged_pred) > total_frames:
                merged_pred = merged_pred[:total_frames]
            elif len(merged_pred) < total_frames:
                # 重复最后一帧
                last_frame = merged_pred[-1]
                merged_pred.extend([last_frame] * (total_frames - len(merged_pred)))
            
            # 计算性能指标
            inference_time = time.time() - start_time
            real_time_factor = inference_time / audio_duration
            
            metrics = {
                'audio_duration': audio_duration,
                'inference_time': inference_time,
                'real_time_factor': real_time_factor,
                'segment_count': segment_num,
                'frame_count': len(merged_pred),
                'fps': self.fps
            }
            
            self.logger.info(
                f"推理完成 - 时长: {audio_duration:.2f}s, "
                f"推理时间: {inference_time:.2f}s, "
                f"实时比: {real_time_factor:.2f}, "
                f"帧数: {len(merged_pred)}"
            )
            
            return merged_pred, metrics
            
        except Exception as e:
            self.logger.error(f"音频推理失败 {wav_path}: {str(e)}")
            raise
    
    def save_predictions(
        self, 
        face_pred: List[List[float]], 
        output_path: str,
        motion_pred: Optional[List[List[float]]] = None
    ):
        """
        保存预测结果到JSON文件，按照指定格式
        
        Args:
            face_pred: 面部表情预测序列 (T, D)
            output_path: 输出文件路径
            motion_pred: 运动预测序列 (T, 3)，可选
        """
        try:
            total_frames = len(face_pred)
            
            # 创建motion_pred（如果未提供）
            if motion_pred is None:
                # 假设motion_pred是3维，初始化为0
                motion_pred = [[0.0, 0.0, 0.0] for _ in range(total_frames)]
            
            # 构建输出数据
            output_data = {
                "params_type": "set_face_animation",
                "motion_pred": motion_pred,
                "face_pred": face_pred,
                "fps": self.fps,
                "frames": total_frames,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # 保存JSON文件
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"预测结果已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存预测结果失败: {str(e)}")
            raise
    
    def load_gt_json(self, gt_path: str) -> Optional[Dict[str, Any]]:
        """
        加载真实值JSON文件
        
        Args:
            gt_path: GT文件路径
            
        Returns:
            GT数据字典，包含face_pred等字段
        """
        try:
            if not os.path.exists(gt_path):
                self.logger.warning(f"GT文件不存在: {gt_path}")
                return None
            
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            # 验证必要字段
            if 'face_pred' not in gt_data:
                self.logger.warning(f"GT文件缺少face_pred字段: {gt_path}")
                return None
            
            self.logger.info(f"GT文件加载成功: {gt_path}")
            return gt_data
            
        except Exception as e:
            self.logger.error(f"加载GT文件失败 {gt_path}: {str(e)}")
            return None
    
    def extract_gt_face_pred(self, gt_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        从GT数据中提取face_pred序列
        
        Returns:
            face_pred序列数组 (T, D)
        """
        try:
            face_pred = gt_data.get('face_pred', [])
            if not face_pred:
                return None
            
            return np.array(face_pred)
            
        except Exception as e:
            self.logger.error(f"提取face_pred失败: {str(e)}")
            return None
    
    def visualize_predictions(
        self, 
        pred_face: np.ndarray, 
        gt_face: Optional[np.ndarray] = None,
        output_path: str = None,
        title: str = "Audio2Face预测结果",
        max_params_to_plot: int = 6
    ):
        """
        可视化预测结果
        
        Args:
            pred_face: 预测的face_pred序列 (T, D)
            gt_face: 真实的face_pred序列 (T, D)，可选
            output_path: 保存路径
            title: 图表标题
            max_params_to_plot: 最多可视化的参数数量
        """
        try:
            # 确定要绘制的参数数量
            num_params = pred_face.shape[1]
            params_to_plot = min(num_params, max_params_to_plot)
            
            # 选择关键参数索引
            # 可以均匀选择，或者选择方差最大的参数
            if num_params > max_params_to_plot:
                # 选择方差最大的参数
                variances = np.var(pred_face, axis=0)
                param_indices = np.argsort(variances)[-params_to_plot:]
            else:
                param_indices = range(params_to_plot)
            
            # 创建图形
            rows = int(np.ceil(params_to_plot / 2))
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
            
            if params_to_plot == 1:
                axes = np.array([axes])
            
            fig.suptitle(title, fontsize=16)
            
            # 绘制每个参数
            for idx, param_idx in enumerate(param_indices):
                ax = axes.flat[idx] if params_to_plot > 1 else axes[0]
                
                # 绘制预测
                ax.plot(pred_face[:, param_idx], 'b-', label='预测', alpha=0.7, linewidth=1.5)
                
                # 绘制真实值（如果存在）
                # if gt_face is not None and len(gt_face) == len(pred_face):
                if gt_face is not None:
                    ax.plot(gt_face[:, param_idx], 'r--', label='真实', alpha=0.7, linewidth=1.5)
                
                ax.set_title(f'参数 {param_idx}')
                ax.set_xlabel('帧')
                ax.set_ylabel('值')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for idx in range(params_to_plot, rows*cols):
                axes.flat[idx].set_visible(False)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"可视化结果已保存: {output_path}")
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
            plt.close('all')
    
    def compute_metrics(
        self, 
        pred_face: np.ndarray, 
        gt_face: np.ndarray
    ) -> Dict[str, float]:
        """
        计算预测与真实值之间的评估指标
        
        Returns:
            包含各项指标的字典
        """
        try:
            # 确保长度一致
            min_len = min(len(pred_face), len(gt_face))
            pred_face = pred_face[:min_len]
            gt_face = gt_face[:min_len]
            
            metrics = {}
            
            # 均方误差 (MSE)
            mse = np.mean((pred_face - gt_face) ** 2)
            metrics['mse'] = float(mse)
            
            # 平均绝对误差 (MAE)
            mae = np.mean(np.abs(pred_face - gt_face))
            metrics['mae'] = float(mae)
            
            # 相关系数（逐参数计算）
            correlations = []
            for i in range(pred_face.shape[1]):
                if (np.std(pred_face[:, i]) > 1e-6 and 
                    np.std(gt_face[:, i]) > 1e-6):
                    corr = np.corrcoef(pred_face[:, i], gt_face[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                metrics['avg_correlation'] = float(np.mean(correlations))
                metrics['max_correlation'] = float(np.max(correlations))
                metrics['min_correlation'] = float(np.min(correlations))
            else:
                metrics['avg_correlation'] = 0.0
                metrics['max_correlation'] = 0.0
                metrics['min_correlation'] = 0.0
            
            # RMSE
            metrics['rmse'] = float(np.sqrt(mse))
            
            # 对称平均绝对百分比误差 (SMAPE)
            denominator = np.abs(pred_face) + np.abs(gt_face) + 1e-8
            smape = 200 * np.mean(np.abs(pred_face - gt_face) / denominator)
            metrics['smape'] = float(smape)
            
            self.logger.info(f"评估指标计算完成: MSE={mse:.4f}, MAE={mae:.4f}, 平均相关系数={metrics.get('avg_correlation', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算评估指标失败: {str(e)}")
            return {}
    
    def run_test_suite(
        self, 
        test_data_dir: str,
        result_name: str = "test_run"
    ) -> pd.DataFrame:
        """
        运行完整的测试套件
        
        Args:
            test_data_dir: 测试数据目录，包含wav/和json/子目录
            result_name: 测试运行的名称
            
        Returns:
            包含所有测试结果的DataFrame
        """
        test_dir = Path(test_data_dir)
        wav_dir = test_dir / "wav"
        gt_json_dir = test_dir / "json"
        
        # 检查目录
        if not wav_dir.exists():
            raise FileNotFoundError(f"wav目录不存在: {wav_dir}")
        
        # 获取所有wav文件
        wav_files = list(wav_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"在{wav_dir}中未找到wav文件")
        
        self.logger.info(f"开始测试套件，共{len(wav_files)}个音频文件")
        
        # 存储所有结果
        all_results = []
        
        for wav_path in wav_files:
            audio_name = wav_path.stem
            self.logger.info(f"处理测试样本: {audio_name}")
            
            try:
                # 1. 推理
                pred_face, perf_metrics = self.inference_single_audio(str(wav_path))
                
                # 2. 保存预测结果（按照指定格式）
                pred_output_path = self.pred_json_dir / f"{audio_name}_pred.json"
                self.save_predictions(pred_face, str(pred_output_path))
                
                # 3. 加载真实值（GT）
                gt_data = None
                gt_face = None
                
                # 查找GT文件（支持__converted.json后缀）
                possible_gt_files = [
                    gt_json_dir / f"{audio_name}.json",
                    gt_json_dir / f"{audio_name}__converted.json"
                ]
                
                gt_path = None
                for gt_file in possible_gt_files:
                    if gt_file.exists():
                        gt_path = gt_file
                        break
                
                if gt_path:
                    gt_data = self.load_gt_json(str(gt_path))
                    if gt_data:
                        gt_face = self.extract_gt_face_pred(gt_data)
                
                # 4. 可视化
                viz_output_path = self.visualization_dir / f"{audio_name}_viz.png"
                self.visualize_predictions(
                    np.array(pred_face), 
                    gt_face,
                    str(viz_output_path),
                    f"音频: {audio_name}"
                )
                
                # 5. 计算评估指标（如果有GT）
                eval_metrics = {}
                if gt_face is not None:
                    # 确保维度匹配
                    pred_face_array = np.array(pred_face)
                    
                    # 如果维度不一致，尝试截断或填充
                    if pred_face_array.shape[1] != gt_face.shape[1]:
                        self.logger.warning(
                            f"预测维度({pred_face_array.shape[1]})与GT维度({gt_face.shape[1]})不匹配，"
                            f"样本: {audio_name}"
                        )
                        # 取最小维度
                        min_dim = min(pred_face_array.shape[1], gt_face.shape[1])
                        pred_face_array = pred_face_array[:, :min_dim]
                        gt_face = gt_face[:, :min_dim]
                    
                    eval_metrics = self.compute_metrics(pred_face_array, gt_face)
                    
                    # 保存评估指标
                    if eval_metrics:
                        metrics_path = self.metrics_dir / f"{audio_name}_metrics.json"
                        with open(metrics_path, 'w') as f:
                            json.dump(eval_metrics, f, indent=2)
                
                # 6. 收集结果
                result = {
                    'audio_name': audio_name,
                    'audio_duration': perf_metrics['audio_duration'],
                    'inference_time': perf_metrics['inference_time'],
                    'real_time_factor': perf_metrics['real_time_factor'],
                    'frame_count': len(pred_face),
                    'fps': self.fps,
                    'has_gt': gt_face is not None
                }
                result.update(eval_metrics)  # 添加评估指标
                
                all_results.append(result)
                
                # 记录性能数据
                self.inference_times.append(perf_metrics['inference_time'])
                self.audio_lengths.append(perf_metrics['audio_duration'])
                
                self.logger.info(f"完成样本: {audio_name}")
                
            except Exception as e:
                self.logger.error(f"处理测试样本失败 {wav_path}: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue
        
        # 生成测试报告
        report = self._generate_test_report(all_results, result_name)
        
        return report
    
    def _generate_test_report(self, all_results: List[Dict], result_name: str) -> pd.DataFrame:
        """生成测试报告"""
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        if df.empty:
            self.logger.warning("没有有效的测试结果")
            return df
        
        # 汇总统计
        summary = {
            'total_samples': len(df),
            'samples_with_gt': df['has_gt'].sum() if 'has_gt' in df.columns else 0,
            'avg_inference_time': df['inference_time'].mean(),
            'avg_real_time_factor': df['real_time_factor'].mean(),
            'total_inference_time': df['inference_time'].sum(),
            'total_audio_duration': df['audio_duration'].sum(),
        }
        
        # 添加评估指标统计（如果有GT）
        if 'mse' in df.columns and df['has_gt'].any():
            summary['avg_mse'] = df[df['has_gt']]['mse'].mean()
            summary['avg_mae'] = df[df['has_gt']]['mae'].mean()
            summary['avg_rmse'] = df[df['has_gt']]['rmse'].mean()
            summary['avg_correlation'] = df[df['has_gt']]['avg_correlation'].mean()
        
        # 保存汇总报告
        summary_path = self.summary_dir / f"{result_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存详细结果CSV
        csv_path = self.summary_dir / f"{result_name}_detailed.csv"
        df.to_csv(csv_path, index=False)
        
        # 创建可视化报告
        self._create_summary_visualization(df, summary, result_name)
        
        self.logger.info(f"测试报告已保存到: {self.summary_dir}")
        self.logger.info(f"汇总结果: {summary}")
        
        return df
    
    def _create_summary_visualization(self, df: pd.DataFrame, summary: Dict, result_name: str):
        """创建汇总可视化"""
        try:
            # 确定子图数量
            has_gt = 'has_gt' in df.columns and df['has_gt'].any()
            num_plots = 3 if has_gt else 2
            
            fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
            if num_plots == 1:
                axes = [axes]
            
            # 1. 推理时间分布
            ax = axes[0]
            ax.hist(df['inference_time'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df['inference_time'].mean(), color='red', linestyle='--', 
                      label=f'均值: {df["inference_time"].mean():.2f}s')
            ax.set_xlabel('推理时间 (秒)')
            ax.set_ylabel('样本数')
            ax.set_title('推理时间分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 实时比分布
            ax = axes[1]
            ax.hist(df['real_time_factor'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.axvline(df['real_time_factor'].mean(), color='red', linestyle='--',
                      label=f'均值: {df["real_time_factor"].mean():.2f}')
            ax.set_xlabel('实时比')
            ax.set_ylabel('样本数')
            ax.set_title('实时比分布 (越低越好)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. 如果有GT，显示MSE分布
            if has_gt and 'mse' in df.columns:
                ax = axes[2]
                valid_mse = df[df['has_gt']]['mse']
                ax.hist(valid_mse, bins=10, alpha=0.7, color='salmon', edgecolor='black')
                ax.axvline(valid_mse.mean(), color='red', linestyle='--',
                          label=f'均值: {valid_mse.mean():.4f}')
                ax.set_xlabel('MSE')
                ax.set_ylabel('样本数')
                ax.set_title('均方误差分布 (越低越好)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'测试套件汇总 - {result_name}', fontsize=14)
            plt.tight_layout()
            
            summary_viz_path = self.summary_dir / f"{result_name}_summary.png"
            plt.savefig(summary_viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 创建性能对比图
            if len(df) > 1:
                self._create_performance_comparison(df, result_name)
            
        except Exception as e:
            self.logger.warning(f"创建汇总可视化失败: {str(e)}")
    
    def _create_performance_comparison(self, df: pd.DataFrame, result_name: str):
        """创建性能对比图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 按推理时间排序
            df_sorted = df.sort_values('inference_time')
            
            x = range(len(df_sorted))
            width = 0.35
            
            # 绘制推理时间
            bars1 = ax.bar(x, df_sorted['inference_time'], width, 
                          label='推理时间 (秒)', color='skyblue')
            
            # 绘制实时比（使用次坐标轴）
            ax2 = ax.twinx()
            bars2 = ax2.bar([i + width for i in x], df_sorted['real_time_factor'], width,
                           label='实时比', color='lightgreen')
            
            # 设置标签
            ax.set_xlabel('测试样本')
            ax.set_ylabel('推理时间 (秒)', color='skyblue')
            ax2.set_ylabel('实时比', color='lightgreen')
            
            # 设置x轴刻度
            ax.set_xticks([i + width/2 for i in x])
            ax.set_xticklabels(df_sorted['audio_name'], rotation=45, ha='right')
            
            # 添加图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title('测试样本性能对比')
            plt.tight_layout()
            
            comparison_path = self.summary_dir / f"{result_name}_performance_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"创建性能对比图失败: {str(e)}")

def main():
    """主函数"""
    # 配置参数
    CONFIG = {
        'model_weights': "./Weights/transformer_decoder_V3.pth",  # 模型权重路径
        'test_data_dir': "./test",  # 测试数据目录
        'output_dir': "./test_results",  # 输出目录
        'device': None,  # 设备: "cuda" 或 "cpu"，None表示自动选择
        'fps': 25,  # 输出帧率
        'result_name': "audio2face_evaluation"  # 测试结果名称
    }
    
    try:
        # 创建测试器
        tester = Audio2FaceTester(
            model_weights_path=CONFIG['model_weights'],
            device=CONFIG['device'],
            output_dir=CONFIG['output_dir'],
            fps=CONFIG['fps']
        )
        
        # 运行测试套件
        results_df = tester.run_test_suite(
            test_data_dir=CONFIG['test_data_dir'],
            result_name=CONFIG['result_name']
        )
        
        # 打印总结
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
        if not results_df.empty:
            print(f"总测试样本数: {len(results_df)}")
            print(f"有GT的样本数: {results_df['has_gt'].sum() if 'has_gt' in results_df.columns else 0}")
            print(f"平均推理时间: {results_df['inference_time'].mean():.2f}s")
            print(f"平均实时比: {results_df['real_time_factor'].mean():.2f}")
            
            if 'mse' in results_df.columns and results_df['has_gt'].any():
                gt_samples = results_df[results_df['has_gt']]
                print(f"平均MSE: {gt_samples['mse'].mean():.6f}")
                print(f"平均MAE: {gt_samples['mae'].mean():.6f}")
                print(f"平均相关系数: {gt_samples['avg_correlation'].mean():.4f}")
        
        print(f"详细结果保存在: {CONFIG['output_dir']}")
        print("="*60)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
