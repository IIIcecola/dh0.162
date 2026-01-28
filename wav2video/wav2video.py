import requests
import os
import time
import json
import subprocess
from pydub import AudioSegment
import shutil
import argparse
import logging
from typing import List, Dict, Optional
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """配置类"""
    def __init__(self, args):
        self.json_path = args.json_path
        self.wave_path = args.wave_path
        self.save_video_path = args.save_video_path
        self.default_wave_path = args.default_wave_path
        self.overwrite = args.overwrite
        self.server_url = args.server_url
        self.video_fps = args.fps
        self.image_save_dir = "./MainEMAGE/save_images"

class VideoProcessor:
    """视频处理器"""
    
    @staticmethod
    def make_video_with_audio(image_dir: str, audio_path: str, output_mp4: str, fps: int = 25) -> bool:
        """使用图片和音频创建视频"""
        try:
            # 获取排序的图片
            image_files = []
            for f in sorted(os.listdir(image_dir)):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_files.append(os.path.abspath(os.path.join(image_dir, f)))
            
            if not image_files:
                logger.error(f"没有图片文件在目录: {image_dir}")
                return False
            
            # 创建临时文件列表
            temp_txt = "temp_image_list.txt"
            with open(temp_txt, "w", encoding="utf-8") as f:
                for img_path in image_files:
                    f.write(f"file '{img_path}'\n")
            
            # 构建FFmpeg命令
            ffmpeg_cmd = [
                'ffmpeg',
                '-r', str(fps),
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_txt,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',  # 视频时长与音频一致
                '-y',
                output_mp4
            ]
            
            logger.info(f"生成视频: {output_mp4} (使用音频: {audio_path})")
            
            # 执行FFmpeg
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if os.path.exists(output_mp4):
                file_size = os.path.getsize(output_mp4) / 1024 / 1024
                logger.info(f"视频生成成功: {output_mp4} ({file_size:.2f} MB)")
                return True
            else:
                logger.error("视频文件未生成")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg执行失败: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"生成视频失败: {e}")
            return False
        finally:
            # 清理临时文件
            if os.path.exists(temp_txt):
                os.remove(temp_txt)

class ProgressTracker:
    """进度跟踪器，用于显示实时进度"""
    
    @staticmethod
    def format_progress(current: int, total: int, width: int = 50) -> str:
        """格式化进度条"""
        if total <= 0:
            return ""
        
        percent = min(current / total, 1.0)
        filled_width = int(width * percent)
        bar = '█' * filled_width + '░' * (width - filled_width)
        percentage = percent * 100
        
        return f"[{bar}] {current}/{total} ({percentage:.1f}%)"
    
    @staticmethod
    def print_progress_line(current: int, total: int, prefix: str = ""):
        """打印进度行，在同一行更新"""
        if total <= 0:
            return
        
        progress_bar = ProgressTracker.format_progress(current, total)
        line = f"{prefix} {progress_bar}" if prefix else progress_bar
        
        # 使用回车符回到行首
        sys.stdout.write('\r' + line)
        sys.stdout.flush()
        
        # 如果完成，换行
        if current >= total:
            print()

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processed_videos = self._get_processed_videos()
    
    def _get_processed_videos(self) -> set:
        """获取已处理的视频文件集合"""
        if not os.path.exists(self.config.save_video_path):
            return set()
        
        return {f for f in os.listdir(self.config.save_video_path) 
                if f.lower().endswith('.mp4')}
    
    def _find_audio_file(self, json_filename: str) -> Optional[str]:
        """查找音频文件"""
        # 1. 尝试查找对应的wav文件
        wav_name = f"{json_filename[:-5]}.wav"
        wav_path = os.path.join(self.config.wave_path, wav_name)
        
        if os.path.exists(wav_path):
            logger.info(f"找到音频文件: {wav_path}")
            return wav_path
        
        # 2. 尝试使用默认音频
        if self.config.default_wave_path and os.path.exists(self.config.default_wave_path):
            logger.info(f"使用默认音频: {self.config.default_wave_path}")
            return self.config.default_wave_path
        
        logger.error(f"未找到音频文件: {wav_name}")
        return None
    
    def _send_to_server(self, json_data: Dict, segment_id: int) -> bool:
        """发送JSON到服务器"""
        try:
            # 添加必要字段
            json_data["segment_type"] = segment_id
            json_data["params_type"] = "set_face_animation"
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.config.server_url, json=json_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Segment {segment_id}: 发送成功")
                return True
            else:
                logger.error(f"Segment {segment_id}: 服务器错误: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Segment {segment_id}: 发送失败: {e}")
            return False
    
    def _wait_for_images(self, save_dir: str, expected_frames: int, segment_id: int, timeout: int = 300) -> bool:
        """等待图片生成，显示实时进度"""
        start_time = time.time()
        last_count = -1
        last_progress_time = time.time()
        
        logger.info(f"Segment {segment_id}: 等待图片生成，期望 {expected_frames} 帧")
        
        while True:
            current_count = 0
            if os.path.exists(save_dir):
                current_count = len([f for f in os.listdir(save_dir) 
                                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            # 只在数量变化或每5秒更新一次进度
            should_update = (current_count != last_count or 
                           time.time() - last_progress_time >= 5 or
                           current_count >= expected_frames)
            
            if should_update:
                elapsed_time = time.time() - start_time
                time_str = time.strftime('%M:%S', time.gmtime(elapsed_time))
                
                # 使用进度条显示
                ProgressTracker.print_progress_line(
                    current_count, 
                    expected_frames,
                    prefix=f"Segment {segment_id}: [{time_str}] "
                )
                
                # 如果完成了，额外打印一条完成消息
                if current_count >= expected_frames:
                    print()  # 换行
                    logger.info(f"Segment {segment_id}: 图片生成完成! 共 {current_count}/{expected_frames} 帧")
                    return True
                
                last_count = current_count
                last_progress_time = time.time()
            
            # 检查超时
            if time.time() - start_time > timeout:
                print()  # 换行
                logger.error(f"Segment {segment_id}: 等待图片超时 ({timeout}秒)")
                logger.error(f"Segment {segment_id}: 当前生成 {current_count}/{expected_frames} 帧")
                return False
            
            # 检查目录是否存在（如果不存在，可能服务器还没开始生成）
            if not os.path.exists(save_dir):
                if time.time() - start_time > 30:  # 30秒后还没创建目录
                    print()  # 换行
                    logger.warning(f"Segment {segment_id}: 图片目录尚未创建，等待中...")
            
            time.sleep(2)
    
    def _process_single_json(self, json_path: str, json_file: str, segment_id: int) -> bool:
        """处理单个JSON文件"""
        try:
            # 检查输出文件是否已存在
            output_video = os.path.join(self.config.save_video_path, f"{json_file[:-5]}.mp4")
            if os.path.exists(output_video) and not self.config.overwrite:
                logger.info(f"Segment {segment_id}: 跳过已存在的文件: {json_file}")
                return True
            
            logger.info(f"Segment {segment_id}: 开始处理 {json_file}")
            
            # 查找音频文件
            audio_path = self._find_audio_file(json_file)
            if not audio_path:
                return False
            
            # 读取JSON
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            expected_frames = json_data.get("frames", 0)
            if expected_frames <= 0:
                logger.error(f"Segment {segment_id}: 无效的帧数: {expected_frames}")
                return False
            
            logger.info(f"Segment {segment_id}: 期望生成 {expected_frames} 帧图片")
            
            # 清空并创建图片保存目录
            image_save_dir = os.path.join(self.config.image_save_dir, str(segment_id))
            if os.path.exists(image_save_dir):
                shutil.rmtree(image_save_dir)
            os.makedirs(image_save_dir, exist_ok=True)
            
            # 发送到服务器
            if not self._send_to_server(json_data, segment_id):
                return False
            
            logger.info(f"Segment {segment_id}: 已发送JSON到服务器，等待图片生成...")
            time.sleep(2)  # 给服务器一点启动时间
            
            # 等待图片生成（带进度显示）
            if not self._wait_for_images(image_save_dir, expected_frames, segment_id):
                return False
            
            time.sleep(2)  # 额外等待
            
            # 生成视频
            os.makedirs(self.config.save_video_path, exist_ok=True)
            logger.info(f"Segment {segment_id}: 开始生成视频...")
            
            success = VideoProcessor.make_video_with_audio(
                image_save_dir, audio_path, output_video, self.config.video_fps
            )
            
            if success:
                logger.info(f"Segment {segment_id}: 处理完成: {json_file} -> {output_video}")
            else:
                logger.error(f"Segment {segment_id}: 处理失败: {json_file}")
            
            return success
            
        except Exception as e:
            logger.error(f"Segment {segment_id}: 处理文件 {json_file} 失败: {e}")
            return False
    
    def run(self):
        """运行批量处理"""
        logger.info("=" * 60)
        logger.info("开始批量处理...")
        logger.info(f"JSON路径: {self.config.json_path}")
        logger.info(f"音频路径: {self.config.wave_path}")
        logger.info(f"输出路径: {self.config.save_video_path}")
        if self.config.default_wave_path:
            logger.info(f"默认音频: {self.config.default_wave_path}")
        logger.info(f"帧率: {self.config.video_fps} FPS")
        logger.info(f"覆盖模式: {'是' if self.config.overwrite else '否'}")
        logger.info("=" * 60)
        
        # 检查输入目录
        if not os.path.exists(self.config.json_path):
            logger.error(f"JSON目录不存在: {self.config.json_path}")
            return
        
        # 获取所有JSON文件
        json_files = [f for f in os.listdir(self.config.json_path) if f.endswith('.json')]
        if not json_files:
            logger.error(f"没有找到JSON文件")
            return
        
        logger.info(f"找到 {len(json_files)} 个JSON文件")
        
        # 清空图片保存目录
        if os.path.exists(self.config.image_save_dir):
            logger.info("清空图片保存目录...")
            shutil.rmtree(self.config.image_save_dir)
        
        # 处理每个JSON文件
        success_count = 0
        logger.info("开始处理各个Segment...")
        
        for idx, json_file in enumerate(sorted(json_files), 1):
            json_path = os.path.join(self.config.json_path, json_file)
            
            logger.info("-" * 40)
            logger.info(f"处理进度: {idx}/{len(json_files)} - {json_file}")
            
            if self._process_single_json(json_path, json_file, idx):
                success_count += 1
                logger.info(f"✅ Segment {idx} 处理成功")
            else:
                logger.error(f"❌ Segment {idx} 处理失败")
        
        logger.info("=" * 60)
        logger.info(f"批量处理完成!")
        logger.info(f"总文件数: {len(json_files)}")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {len(json_files) - success_count}")
        logger.info("=" * 60)
        
        # 清理图片目录
        if os.path.exists(self.config.image_save_dir):
            logger.info("清理临时图片目录...")
            shutil.rmtree(self.config.image_save_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="音频后处理模拟器")
    parser.add_argument("--json-path", required=True, help="JSON文件目录路径")
    parser.add_argument("--wave-path", required=True, help="WAV文件目录路径")
    parser.add_argument("--save-video-path", required=True, help="视频保存路径")
    parser.add_argument("--default-wave-path", help="默认音频文件路径（可选）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的视频文件")
    parser.add_argument("--server-url", default="http://127.0.0.1:8084/send_json", help="服务器URL")
    parser.add_argument("--fps", type=int, default=25, help="视频帧率")
    
    args = parser.parse_args()
    
    # 创建配置和处理器
    config = Config(args)
    processor = BatchProcessor(config)
    
    # 运行处理
    processor.run()

if __name__ == "__main__":
    main()
