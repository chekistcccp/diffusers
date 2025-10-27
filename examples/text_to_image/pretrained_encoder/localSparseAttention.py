import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSparseAttention(nn.Module):
    """
    实现局部稀疏注意力机制（滑动窗口注意力）
    """
    def __init__(self, input_dim, window_size=64):
        """
        初始化局部稀疏注意力层
        
        Args:
            input_dim: 输入特征维度
            window_size: 注意力窗口大小（每个token只能关注前后window_size//2个token）
        """
        super(LocalSparseAttention, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        
        # 线性投影层
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        
        # 缩放因子
        self.scale = input_dim ** 0.5

    def create_local_mask(self, seq_len):
        """
        创建局部注意力掩码
        
        Args:
            seq_len: 序列长度
            
        Returns:
            mask: 局部注意力掩码 (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len)
        half_window = self.window_size // 2
        
        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 1
        
        return mask.bool()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            output: 输出张量 (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        queries = self.query_proj(x)  # (batch_size, seq_len, input_dim)
        keys = self.key_proj(x)       # (batch_size, seq_len, input_dim)
        values = self.value_proj(x)   # (batch_size, seq_len, input_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)
        
        # 应用局部注意力掩码
        mask = self.create_local_mask(seq_len).to(x.device)  # (seq_len, seq_len)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)
        
        # 将窗口外的注意力分数设为负无穷
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 应用注意力权重到值
        output = torch.matmul(attn_weights, values)  # (batch_size, seq_len, input_dim)
        
        return output