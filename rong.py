import torch
import torch.nn.functional as F
import numpy as np


def check_feature_redundancy(x_haar, x_res, threshold=0.7):
    """
    检查两个特征图张量(BCHW)是否存在冗余

    参数:
        x_haar (torch.Tensor): 小波变换特征图, 形状为 [B,C,H,W]
        x_res (torch.Tensor): ResNet特征图, 形状为 [B,C,H,W]
        threshold (float): 冗余判断阈值

    返回:
        dict: 包含冗余分析结果
    """
    results = {}

    # 1. 全局余弦相似度 - 将整个张量展平后计算
    def cosine_similarity_global(t1, t2):
        # 确保张量形状一致并展平
        t1_flat = t1.reshape(t1.shape[0], -1)  # [B, C*H*W]
        t2_flat = t2.reshape(t2.shape[0], -1)  # [B, C*H*W]

        # 批量计算余弦相似度
        cos_sim = F.cosine_similarity(t1_flat, t2_flat, dim=1)  # [B]
        return cos_sim.mean().item()  # 返回批次平均值

    # 2. 通道级别皮尔逊相关系数 - 分析各通道之间的相关性
    def pearson_correlation_channels(t1, t2):
        # 均值和标准差将在空间维度(H,W)上计算
        b, c1, h, w = t1.shape
        b, c2, h, w = t2.shape

        # 将每个通道展平为向量: [B, C, H*W]
        t1_flat = t1.reshape(b, c1, -1)
        t2_flat = t2.reshape(b, c2, -1)

        # 计算每个样本、每个通道的均值和标准差
        t1_mean = t1_flat.mean(dim=2, keepdim=True)  # [B, C1, 1]
        t2_mean = t2_flat.mean(dim=2, keepdim=True)  # [B, C2, 1]

        t1_std = t1_flat.std(dim=2, keepdim=True)  # [B, C1, 1]
        t2_std = t2_flat.std(dim=2, keepdim=True)  # [B, C2, 1]

        # 避免除零错误
        t1_std = torch.clamp(t1_std, min=1e-8)
        t2_std = torch.clamp(t2_std, min=1e-8)

        # 归一化
        t1_norm = (t1_flat - t1_mean) / t1_std  # [B, C1, H*W]
        t2_norm = (t2_flat - t2_mean) / t2_std  # [B, C2, H*W]

        # 如果通道数不同，只计算共同的通道数
        min_channels = min(c1, c2)

        # 每批次、每通道的相关系数
        corr = torch.zeros(b, min_channels)

        for i in range(b):
            for j in range(min_channels):
                # 计算相关系数: (t1_norm * t2_norm).mean()
                corr[i, j] = (t1_norm[i, j] * t2_norm[i, j]).mean()

        # 计算每批次通道的平均相关系数
        avg_channel_corr = corr.mean(dim=1)  # [B]
        # 返回批次间的平均值
        return avg_channel_corr.mean().item()

    # 计算余弦相似度
    cos_sim = cosine_similarity_global(x_haar, x_res)
    results['cosine_similarity'] = cos_sim
    results['cosine_redundant'] = cos_sim > threshold

    # 计算皮尔逊相关系数
    pearson_corr = pearson_correlation_channels(x_haar, x_res)
    results['pearson_correlation'] = pearson_corr
    results['pearson_redundant'] = abs(pearson_corr) > threshold

    # 综合判断
    results['is_redundant'] = results['cosine_redundant'] and results['pearson_redundant']

    return results


# 使用示例
def test_redundancy(x_haar, x_res):
    """
    测试并输出冗余分析结果
    """
    results = check_feature_redundancy(x_haar, x_res)

    print(f"全局余弦相似度: {results['cosine_similarity']:.4f}")
    print(f"通道皮尔逊相关系数: {results['pearson_correlation']:.4f}")

    if results['is_redundant']:
        print("结论: 特征图存在明显冗余，建议考虑优化网络结构")
    else:
        print("结论: 特征图没有明显冗余，两种特征提取方式可能各有价值")

    return results


# 假设我们有以下特征图
# x_haar = self.HWD1(x)
# x_res = self.resnet1(x)

# 测试冗余性