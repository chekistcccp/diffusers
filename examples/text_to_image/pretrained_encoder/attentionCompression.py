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

class AttentionCompression(nn.Module):
    def __init__(self, input_dim, output_dim, num_tokens, num_compressed_tokens, use_local_sparse_attention=True, window_size=64):
        super(AttentionCompression, self).__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_compressed_tokens = num_compressed_tokens
        self.use_local_sparse_attention = use_local_sparse_attention

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_compressed_tokens, input_dim))

        # Linear projections for keys and values
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.dim_compress = nn.Linear(input_dim, output_dim)
        
        # 局部稀疏注意力模块
        if self.use_local_sparse_attention:
            self.local_sparse_attention = LocalSparseAttention(input_dim, window_size)

    def forward(self, x):
        # x shape: (batch_size, num_tokens, input_dim)
        #print('<><><> x.shape in attentionCompression.py: ', x.shape) #torch.Size([1, 4000, 1024])
        batch_size = x.shape[0]
        
        # 如果启用局部稀疏注意力，先对输入进行局部注意力处理
        if self.use_local_sparse_attention:
            x = self.local_sparse_attention(x)  # (batch_size, num_tokens, input_dim)

        # Project input to keys and values
        keys = self.key_proj(x)  # (batch_size, num_tokens, input_dim)
        values = self.value_proj(x)  # (batch_size, num_tokens, input_dim)

        # Repeat query tokens for batch size
        queries = self.query_tokens.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_compressed_tokens, input_dim)

        # Compute attention scores
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)  # (batch_size, num_compressed_tokens, num_tokens)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_compressed_tokens, num_tokens)

        # Apply attention to values
        compressed_output = torch.bmm(attn_weights, values)  # (batch_size, num_compressed_tokens, input_dim)
        compressed_output = self.dim_compress(compressed_output)
        #print('<><><> compressed_output shape: ', compressed_output.shape) # torch.Size([1, 51, 768]) # changed to 51 X 768 05/18/2025
        return compressed_output



