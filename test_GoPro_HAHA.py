import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte
import glob

# =============================================
# 导入带物理先验约束的 HAHA 版本模型
# =============================================
from MISCFilterNet_HAHA import MISCKernelNet_HAHA as myNet
import utils
from tools.get_parameter_number import get_parameter_number
from models.layers_Deform import window_partitionx, window_reversex


# -----------------------------------------
# 图像缩放函数
# -----------------------------------------
def resize_image(input_path, output_path, size, interpolation=cv2.INTER_LANCZOS4):
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ 无法读取图像: {input_path}")
        return False

    resized = cv2.resize(img, size, interpolation=interpolation)
    cv2.imwrite(output_path, resized)
    print(f"✔ 已保存缩放图像: {output_path}")
    return True


# -----------------------------------------
# MISCFilter-HAHA (Deformable + Physical Prior) 推理器
# -----------------------------------------
class IterativeDeblurrer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def setup_model(self, weights_path,
                    use_deform_in_feat=True,
                    use_deform_in_encoder=True,
                    rigid_smooth_weight=0.05,
                    rotation_aware_weight=0.03,
                    flow_gradient_weight=0.02):
        """
        初始化带物理先验约束的 HAHA 模型

        参数:
        - weights_path:           权重文件路径
        - use_deform_in_feat:     是否在特征提取中使用可变形卷积
        - use_deform_in_encoder:  是否在编码器/解码器中使用可变形卷积
        - rigid_smooth_weight:    刚性运动平滑约束权重（需与训练时一致）
        - rotation_aware_weight:  旋转感知约束权重（需与训练时一致）
        - flow_gradient_weight:   流梯度平滑约束权重（需与训练时一致）
        """
        model_restoration = myNet(
            inference=False,
            use_deform_in_feat=use_deform_in_feat,
            use_deform_in_encoder=use_deform_in_encoder,
            rigid_smooth_weight=rigid_smooth_weight,
            rotation_aware_weight=rotation_aware_weight,
            flow_gradient_weight=flow_gradient_weight,
        )

        # 打印模型参数量
        total_num, trainable_num = get_parameter_number(model_restoration)
        print(f"模型参数量 - Total: {total_num}, Trainable: {trainable_num}")

        # 加载权重
        utils.load_checkpoint(model_restoration, weights_path)
        print(f"=== 使用权重: {weights_path}")
        print(f"=== 可变形卷积配置: feat={use_deform_in_feat}, encoder={use_deform_in_encoder}")
        print(f"=== 物理先验权重: rigid_smooth={rigid_smooth_weight}, "
              f"rotation_aware={rotation_aware_weight}, "
              f"flow_gradient={flow_gradient_weight}")

        if self.device.type == 'cuda':
            model_restoration.cuda()
            model_restoration = nn.DataParallel(model_restoration)
        else:
            model_restoration = model_restoration.to(self.device)

        model_restoration.eval()
        return model_restoration

    def run_once(self, image_path, output_path, weights_path, win_size,
                 use_deform_in_feat=True,
                 use_deform_in_encoder=True,
                 rigid_smooth_weight=0.05,
                 rotation_aware_weight=0.03,
                 flow_gradient_weight=0.02):
        """
        单次推理

        参数:
        - image_path:             输入图像路径
        - output_path:            输出图像路径
        - weights_path:           权重文件路径
        - win_size:               窗口大小
        - use_deform_in_feat:     是否在特征提取中使用可变形卷积
        - use_deform_in_encoder:  是否在编码器/解码器中使用可变形卷积
        - rigid_smooth_weight:    刚性运动平滑约束权重
        - rotation_aware_weight:  旋转感知约束权重
        - flow_gradient_weight:   流梯度平滑约束权重
        """
        model = self.setup_model(
            weights_path,
            use_deform_in_feat=use_deform_in_feat,
            use_deform_in_encoder=use_deform_in_encoder,
            rigid_smooth_weight=rigid_smooth_weight,
            rotation_aware_weight=rotation_aware_weight,
            flow_gradient_weight=flow_gradient_weight,
        )

        with torch.no_grad():
            inp = Image.open(image_path).convert('RGB')
            inp = np.array(inp).astype(np.float32) / 255.0
            inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)

            _, _, Hx, Wx = inp.shape

            # 分块
            input_re, batch_list = window_partitionx(inp, win_size)

            # 推理（HAHA 训练模式返回三元组：outputs, outputs_fil, Kernal_Loss）
            restored, _, _ = model(input_re)
            restored = restored[0]

            # 反拼接
            restored = window_reversex(restored, win_size, Hx, Wx, batch_list)
            restored = torch.clamp(restored, 0, 1)

            # 保存图像
            final_img = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
            final_img = img_as_ubyte(final_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, final_img)

            print(f"✔ 推理完成 → {output_path}")


# -----------------------------------------
# 批量处理函数
# -----------------------------------------
def process_all_images(input_folder, output_folder, weights_path,
                       win_size=512,
                       final_size=(1024, 1024),
                       use_deform_in_feat=True,
                       use_deform_in_encoder=True,
                       rigid_smooth_weight=0.05,
                       rotation_aware_weight=0.03,
                       flow_gradient_weight=0.02):
    """
    处理文件夹中的所有图像

    参数:
    - input_folder:           输入图像文件夹路径
    - output_folder:          输出图像文件夹路径
    - weights_path:           模型权重路径
    - win_size:               窗口大小
    - final_size:             最终输出图像大小
    - use_deform_in_feat:     是否在特征提取中使用可变形卷积
    - use_deform_in_encoder:  是否在编码器/解码器中使用可变形卷积
    - rigid_smooth_weight:    刚性运动平滑约束权重
    - rotation_aware_weight:  旋转感知约束权重
    - flow_gradient_weight:   流梯度平滑约束权重
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 创建临时文件夹
    temp_folder = "./temp_processing_haha"
    os.makedirs(temp_folder, exist_ok=True)

    # 支持多种图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    # 收集所有图像文件
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    # 去重并排序
    image_files = list(set(image_files))
    image_files.sort()

    print(f"找到 {len(image_files)} 个图像文件")
    print(f"使用 HAHA 模型（Deformable + Physical Prior Constraints）")
    print(f"  可变形卷积: feat={use_deform_in_feat}, encoder={use_deform_in_encoder}")
    print(f"  物理先验权重: rigid_smooth={rigid_smooth_weight}, "
          f"rotation_aware={rotation_aware_weight}, "
          f"flow_gradient={flow_gradient_weight}")

    if len(image_files) == 0:
        print("❌ 未找到图像文件！")
        return

    # 初始化去模糊器
    deb = IterativeDeblurrer()

    # 处理每个图像
    for i, image_path in enumerate(image_files, 1):
        try:
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]

            print(f"\n{'=' * 50}")
            print(f"处理图像 {i}/{len(image_files)}: {filename}")
            print(f"{'=' * 50}")

            # 临时文件路径
            tmp_512 = os.path.join(temp_folder, f"{name_without_ext}_512.jpg")
            out_512 = os.path.join(temp_folder, f"{name_without_ext}_output_512.jpg")

            # 最终输出路径
            final_output = os.path.join(output_folder, f"{name_without_ext}_deblurred_haha.jpg")

            print(f"=== Step1：缩放到 512×512 ===")
            if not resize_image(image_path, tmp_512, (512, 512)):
                continue

            print(f"\n=== Step2：MISCFilter-HAHA（win={win_size}）===")
            deb.run_once(
                tmp_512, out_512, weights_path, win_size=win_size,
                use_deform_in_feat=use_deform_in_feat,
                use_deform_in_encoder=use_deform_in_encoder,
                rigid_smooth_weight=rigid_smooth_weight,
                rotation_aware_weight=rotation_aware_weight,
                flow_gradient_weight=flow_gradient_weight,
            )

            print(f"\n=== Step3：将结果放大到 {final_size[0]}×{final_size[1]} ===")
            resize_image(out_512, final_output, final_size)

            print(f"\n✅ 处理完成: {final_output}")

        except Exception as e:
            print(f"❌ 处理图像 {image_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 清理临时文件
    try:
        import shutil
        shutil.rmtree(temp_folder)
        print(f"\n已清理临时文件夹: {temp_folder}")
    except Exception:
        print(f"\n临时文件夹清理失败: {temp_folder}")

    print(f"\n{'=' * 50}")
    print(f"全部完成！共处理 {len(image_files)} 个图像")
    print(f"输出文件夹: {output_folder}")
    print(f"{'=' * 50}")


# -----------------------------------------
# 单张图像处理函数
# -----------------------------------------
def process_single_image(input_path, output_path, weights_path,
                         win_size=512,
                         use_deform_in_feat=True,
                         use_deform_in_encoder=True,
                         rigid_smooth_weight=0.05,
                         rotation_aware_weight=0.03,
                         flow_gradient_weight=0.02):
    """
    处理单张图像（不缩放，直接处理原图）

    参数:
    - input_path:             输入图像路径
    - output_path:            输出图像路径
    - weights_path:           模型权重路径
    - win_size:               窗口大小
    - use_deform_in_feat:     是否在特征提取中使用可变形卷积
    - use_deform_in_encoder:  是否在编码器/解码器中使用可变形卷积
    - rigid_smooth_weight:    刚性运动平滑约束权重
    - rotation_aware_weight:  旋转感知约束权重
    - flow_gradient_weight:   流梯度平滑约束权重
    """
    print(f"{'=' * 50}")
    print(f"处理单张图像: {input_path}")
    print(f"使用 HAHA 模型（Deformable + Physical Prior Constraints）")
    print(f"  可变形卷积: feat={use_deform_in_feat}, encoder={use_deform_in_encoder}")
    print(f"  物理先验权重: rigid_smooth={rigid_smooth_weight}, "
          f"rotation_aware={rotation_aware_weight}, "
          f"flow_gradient={flow_gradient_weight}")
    print(f"{'=' * 50}")

    deb = IterativeDeblurrer()
    deb.run_once(
        input_path, output_path, weights_path, win_size=win_size,
        use_deform_in_feat=use_deform_in_feat,
        use_deform_in_encoder=use_deform_in_encoder,
        rigid_smooth_weight=rigid_smooth_weight,
        rotation_aware_weight=rotation_aware_weight,
        flow_gradient_weight=flow_gradient_weight,
    )

    print(f"\n✅ 处理完成: {output_path}")


# -----------------------------------------
# 主流程：批量处理
# -----------------------------------------
def main():
    # =============================================
    # 配置参数（在这里修改）
    # =============================================

    # 模型权重路径（HAHA 版本）
    weights_path = r"/media/JYJ/新加卷/ZJL/MISCFilter-main/checkpoints_haha/GoPro/MISCFilter_HAHA_GoPro/model_best.pth"

    # 输入文件夹路径（包含所有模糊图像）
    input_folder = r"/media/JYJ/新加卷/ZJL/MOHU2"

    # 输出文件夹路径（保存去模糊后的图像）
    output_folder = r"/media/JYJ/新加卷/ZJL/SHARPMOHU2_HAHA"

    # 可变形卷积配置（需要和训练时一致）
    use_deform_in_feat = True    # 特征提取使用可变形卷积
    use_deform_in_encoder = True  # 编码器/解码器使用可变形卷积

    # 物理先验约束权重（需要和训练时一致）
    rigid_smooth_weight = 0.05    # 刚性运动平滑约束权重
    rotation_aware_weight = 0.03  # 旋转感知约束权重
    flow_gradient_weight = 0.02   # 流梯度平滑约束权重

    # 窗口大小和输出尺寸
    win_size = 512
    final_size = (1024, 1024)

    # =============================================
    # 批量处理所有图像
    # =============================================
    process_all_images(
        input_folder=input_folder,
        output_folder=output_folder,
        weights_path=weights_path,
        win_size=win_size,
        final_size=final_size,
        use_deform_in_feat=use_deform_in_feat,
        use_deform_in_encoder=use_deform_in_encoder,
        rigid_smooth_weight=rigid_smooth_weight,
        rotation_aware_weight=rotation_aware_weight,
        flow_gradient_weight=flow_gradient_weight,
    )

    # =============================================
    # 或者处理单张图像（取消注释使用）
    # =============================================
    # process_single_image(
    #     input_path=r"/path/to/input.jpg",
    #     output_path=r"/path/to/output.jpg",
    #     weights_path=weights_path,
    #     win_size=512,
    #     use_deform_in_feat=use_deform_in_feat,
    #     use_deform_in_encoder=use_deform_in_encoder,
    #     rigid_smooth_weight=rigid_smooth_weight,
    #     rotation_aware_weight=rotation_aware_weight,
    #     flow_gradient_weight=flow_gradient_weight,
    # )


if __name__ == '__main__':
    main()
