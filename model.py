import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l].transpose(1, 2)
        x = self.activation(x)

        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = nn.functional.softplus(self.dt_proj(delta))

        y = torch.zeros_like(x)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        
        for t in range(l):
            dt = delta[:, t, :].unsqueeze(-1)
            b_t = B[:, t, :].unsqueeze(1)
            c_t = C[:, t, :].unsqueeze(-1)
            x_t = x[:, t, :].unsqueeze(-1)
            
            dA = torch.exp(dt * A.unsqueeze(0))
            dB = dt * b_t
            h = dA * h + dB * x_t
            y[:, t, :] = torch.matmul(h, c_t).squeeze(-1) + x[:, t, :] * self.D

        return self.out_proj(y * self.activation(z))

class BidirectionalMambaEncoder(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        x_fwd = self.forward_mamba(x)
        x_bwd = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return self.proj(torch.cat([x_fwd, x_bwd], dim=-1))

class SemanticWeightedPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        scores = self.attn_net(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)

class EHRMambaTransformer(nn.Module):
    def __init__(self, num_numerical, num_categorical, cat_dims, d_model=64, nhead=4, num_layers=2, d_state=16, dropout=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, d_model) for dim in cat_dims])
        self.num_proj = nn.Linear(num_numerical, d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.feature_fuse = nn.Linear(d_model * (2 + len(cat_dims)), d_model)
        
        self.mamba_encoder = BidirectionalMambaEncoder(d_model, d_state=d_state)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = SemanticWeightedPooling(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )

    def forward(self, num_feats, cat_feats, deltas):
        b, l, _ = num_feats.shape
        z_num = self.num_proj(num_feats)
        z_time = self.time_proj(deltas)
        
        z_cat_list = [embed_layer(cat_feats[:, i]).unsqueeze(1).expand(-1, l, -1) for i, embed_layer in enumerate(self.cat_embeddings)]
        fused = torch.cat([z_num, z_time] + z_cat_list, dim=-1)
        x = self.feature_fuse(fused)
        
        x = self.mamba_encoder(x)
        x = self.transformer(x)
        g_repr = self.pooling(x)
        return self.classifier(g_repr).squeeze(-1)