import os
import cv2
import numpy as np
import random
from pathlib import Path

# 确保结果可复现（可选）
# random.seed(42)
# np.random.seed(42)


class ImageNoiseProcessor:
    """
    一个用于对图像数据集应用多种噪声处理的类。
    它能保持原有的目录结构，并将处理后的图像保存到新的根目录中。
    """

    def __init__(self):
        """初始化处理器，可以在这里设置默认参数。"""
        pass

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

    def process_and_save_image(self, input_path: Path, output_base_dir: Path):
        """
        处理单张图像，生成三种扰动版本并保存。

        Args:
            input_path: 输入图像的完整路径
            output_base_dir: 输出图像的基准目录（不含 'occluded', 'noisy' 等子文件夹）
        """
        try:
            # 读取图像
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"警告：无法读取图像: {input_path}")
                return

            filename = input_path.name

            # 定义三种噪声的输出目录并创建
            occluded_dir = output_base_dir / 'occluded'
            noisy_dir = output_base_dir / 'noisy'
            rotated_dir = output_base_dir / 'rotated'

            # 使用 Path.mkdir(parents=True, exist_ok=True) 可以安全地创建多级目录
            occluded_dir.mkdir(parents=True, exist_ok=True)
            noisy_dir.mkdir(parents=True, exist_ok=True)
            rotated_dir.mkdir(parents=True, exist_ok=True)

            # 1. 应用遮挡并保存
            occluded_image = self.apply_occlusion(image)
            cv2.imwrite(str(occluded_dir / filename), occluded_image)

            # 2. 应用高斯噪声并保存（随机选择sigma值）
            sigma = random.choice([0.05, 0.1])
            noisy_image = self.apply_gaussian_noise(image, sigma)
            cv2.imwrite(str(noisy_dir / filename), noisy_image)

            # 3. 应用旋转并保存
            rotated_image = self.apply_rotation(image)
            cv2.imwrite(str(rotated_dir / filename), rotated_image)

            print(
                f"✅ 已处理: {input_path.name} -> {output_base_dir.relative_to(Path(self.dest_root))}")

        except Exception as e:
            print(f"❌ 处理图像 {input_path} 时出错: {str(e)}")

    def process_dataset(self, source_root: str, dest_root: str):
        """
        遍历整个数据集，处理所有图像文件。

        Args:
            source_root: 源数据集根目录
            dest_root: 目标数据集根目录
        """
        self.source_root = source_root
        self.dest_root = dest_root

        source_path = Path(source_root)
        dest_path = Path(dest_root)

        # 定义支持的图像文件扩展名
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        print(f"🔍 开始扫描源目录: {source_path}")
        # 使用 rglob 递归查找所有文件，然后过滤出图像文件
        image_files = [f for f in source_path.rglob(
            '*') if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print("未找到任何图像文件，请检查源目录路径和图像文件扩展名。")
            return

        print(f"📊 找到 {len(image_files)} 个图像文件。开始处理...")

        for i, image_path in enumerate(image_files, 1):
            # 计算相对于源根目录的相对路径
            # 例如: CheXpert/train/patient00001/view1.jpg -> train/patient00001
            relative_path = image_path.relative_to(source_path)
            output_base_dir = dest_path / relative_path.parent

            # 处理并保存单张图片
            self.process_and_save_image(image_path, output_base_dir)

            # 可选：打印进度
            if (i % 100 == 0) or (i == len(image_files)):
                print(f"--- 进度: {i}/{len(image_files)} ---")

        print("\n🎉 所有图像处理完成！")


if __name__ == "__main__":
    # --- 请在这里修改你的路径 ---
    # 假设你的 CheXpert 数据集与此脚本在同一目录下
    SOURCE_DATASET_PATH = "./CheXpert"
    # 新生成的数据集将保存在这里
    DESTINATION_DATASET_PATH = "./chexpert-chaos"
    # -------------------------

    # 检查源目录是否存在
    if not Path(SOURCE_DATASET_PATH).exists():
        print(f"错误：源目录 '{SOURCE_DATASET_PATH}' 不存在！")
        print("请修改脚本中的 'SOURCE_DATASET_PATH' 变量为正确的路径。")
    else:
        # 创建处理器实例
        processor = ImageNoiseProcessor()

        # 开始处理整个数据集
        processor.process_dataset(
            SOURCE_DATASET_PATH, DESTINATION_DATASET_PATH)
