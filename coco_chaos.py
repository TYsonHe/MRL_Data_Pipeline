import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from pathlib import Path
import argparse


class MSCOCOImageProcessor:
    def __init__(self, base_dir, output_dir):
        """
        初始化图像处理器

        Args:
            base_dir: MSCOCO数据集根目录
            output_dir: 输出目录
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        # 创建输出目录结构
        for split in ['train2014', 'val2014', 'test2014']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir /
             f'{split}_occluded').mkdir(parents=True, exist_ok=True)
            (self.output_dir /
             f'{split}_noisy').mkdir(parents=True, exist_ok=True)
            (self.output_dir /
             f'{split}_rotated').mkdir(parents=True, exist_ok=True)

    def apply_occlusion(self, image, occlusion_ratio=0.2):
        """
        应用遮挡扰动（20%方块）

        Args:
            image: 输入图像
            occlusion_ratio: 遮挡比例

        Returns:
            遮挡后的图像
        """
        h, w = image.shape[:2]

        # 计算遮挡方块大小
        block_size = int(min(h, w) * occlusion_ratio)

        # 随机选择遮挡位置
        x = random.randint(0, w - block_size)
        y = random.randint(0, h - block_size)

        # 应用遮挡（用黑色方块遮挡）
        result = image.copy()
        result[y:y+block_size, x:x+block_size] = 0

        return result

    def apply_gaussian_noise(self, image, sigma=0.1):
        """
        应用高斯噪声

        Args:
            image: 输入图像
            sigma: 噪声标准差

        Returns:
            加噪后的图像
        """
        # 将图像转换为浮点型并归一化到[0,1]
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)

        # 生成高斯噪声
        noise = np.random.normal(0, sigma, image_float.shape)

        # 添加噪声并限制在[0,1]范围内
        noisy_image = image_float + noise
        noisy_image = np.clip(noisy_image, 0, 1)

        # 转换回原始数据类型
        if image.dtype == np.uint8:
            return (noisy_image * 255).astype(np.uint8)
        else:
            return noisy_image.astype(image.dtype)

    def apply_rotation(self, image, max_angle=10):
        """
        应用旋转扰动（±10度）

        Args:
            image: 输入图像
            max_angle: 最大旋转角度

        Returns:
            旋转后的图像
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 随机选择旋转角度
        angle = random.uniform(-max_angle, max_angle)

        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 应用旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101)

        return rotated_image

    def process_image(self, image_path, output_split_dir):
        """
        处理单张图像，生成三种扰动版本

        Args:
            image_path: 输入图像路径
            output_split_dir: 输出分割目录
        """
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取图像: {image_path}")
                return

            # 获取文件名
            filename = image_path.name

            # 1. 应用遮挡
            occluded_image = self.apply_occlusion(image)
            cv2.imwrite(str(output_split_dir / 'occluded' /
                        filename), occluded_image)

            # 2. 应用高斯噪声（随机选择sigma值）
            sigma = random.choice([0.05, 0.1])
            noisy_image = self.apply_gaussian_noise(image, sigma)
            cv2.imwrite(
                str(output_split_dir / 'noisy' / filename), noisy_image)

            # 3. 应用旋转
            rotated_image = self.apply_rotation(image)
            cv2.imwrite(
                str(output_split_dir / 'rotated' / filename), rotated_image)

            print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")

    def process_dataset(self, splits=['train2014', 'val2014', 'test2014']):
        """
        处理整个数据集

        Args:
            splits: 要处理的数据集分割
        """
        for split in splits:
            print(f"\n开始处理 {split} 分割...")

            # 输入和输出目录
            input_dir = self.base_dir / split
            output_split_dir = self.output_dir / split

            # 创建输出子目录
            for perturbation in ['occluded', 'noisy', 'rotated']:
                (output_split_dir / perturbation).mkdir(parents=True, exist_ok=True)

            # 处理所有图像
            image_files = list(input_dir.glob('*.jpg')) + \
                list(input_dir.glob('*.png'))

            if not image_files:
                print(f"在 {input_dir} 中未找到图像文件")
                continue

            print(f"找到 {len(image_files)} 张图像")

            for i, image_path in enumerate(image_files, 1):
                self.process_image(image_path, output_split_dir)

                # 每处理100张图像显示进度
                if i % 100 == 0:
                    print(f"已处理 {i}/{len(image_files)} 张图像")

            print(f"{split} 分割处理完成！")

    def create_processing_log(self, splits=['train2014', 'val2014', 'test2014']):
        """
        创建处理日志文件

        Args:
            splits: 处理的数据集分割
        """
        log_info = {
            'processing_date': '2025-11-24',
            'dataset': 'MSCOCO',
            'splits_processed': splits,
            'perturbations': {
                'occlusion': {
                    'type': 'square_block',
                    'ratio': 0.2,
                    'description': '20%方块遮挡'
                },
                'gaussian_noise': {
                    'type': 'gaussian',
                    'sigma_values': [0.05, 0.1],
                    'description': '高斯噪声，σ∈{0.05, 0.1}'
                },
                'rotation': {
                    'type': 'random_rotation',
                    'angle_range': '±10°',
                    'description': '随机旋转±10度'
                }
            }
        }

        # 保存日志文件
        with open(self.output_dir / 'processing_log.json', 'w', encoding='utf-8') as f:
            json.dump(log_info, f, indent=2, ensure_ascii=False)

        print(f"处理日志已保存到: {self.output_dir / 'processing_log.json'}")


def main():
    parser = argparse.ArgumentParser(description='MSCOCO图像扰动处理工具')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='MSCOCO数据集根目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--splits', nargs='+', default=['train2014', 'val2014', 'test2014'],
                        help='要处理的数据集分割 (默认: train2014 val2014 test2014)')

    args = parser.parse_args()

    # 创建处理器
    processor = MSCOCOImageProcessor(args.input_dir, args.output_dir)

    # 处理数据集
    processor.process_dataset(args.splits)

    # 创建处理日志
    processor.create_processing_log(args.splits)

    print("\n所有图像处理完成！")
    print(f"输出目录: {args.output_dir}")
    print("\n输出结构:")
    for split in args.splits:
        print(f"  {split}/")
        print(f"    ├── occluded/    # 遮挡图像")
        print(f"    ├── noisy/       # 加噪图像")
        print(f"    └── rotated/     # 旋转图像")


if __name__ == "__main__":
    main()
