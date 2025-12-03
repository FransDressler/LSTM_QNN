import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class ExponentialGating(nn.Module):
    def __init__(self, input_size: int):
        super(ExponentialGating, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.linear(x))


class sLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        self.exp_gating = ExponentialGating(hidden_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_prev, c_prev, n_prev = hidden

        combined = torch.cat([x, h_prev], dim=1)

        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        g_t = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        c_t = f_t * c_prev + i_t * g_t
        n_t = f_t * n_prev + i_t * self.exp_gating(g_t)

        h_t = o_t * (c_t / n_t)

        return h_t, c_t, n_t


class mLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        self.key_proj = nn.Linear(input_size, hidden_size)
        self.value_proj = nn.Linear(input_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        self.exp_gating = ExponentialGating(hidden_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_prev, C_prev, n_prev = hidden

        combined = torch.cat([x, h_prev], dim=1)

        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        k_t = self.key_proj(x)
        v_t = self.value_proj(x)
        q_t = self.query_proj(h_prev)

        C_t = f_t.unsqueeze(-1) * C_prev + i_t.unsqueeze(-1) * torch.outer(v_t, k_t)
        n_t = f_t * n_prev + i_t * self.exp_gating(k_t)

        h_t = o_t * (torch.matmul(C_t, q_t) / n_t)

        return h_t, C_t, n_t


class xLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cell_type: str = 'slstm'):
        super(xLSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        if cell_type == 'slstm':
            self.lstm_cell = sLSTMCell(input_size, hidden_size)
        elif cell_type == 'mlstm':
            self.lstm_cell = mLSTMCell(input_size, hidden_size)
        else:
            raise ValueError("cell_type must be 'slstm' or 'mlstm'")

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            if self.cell_type == 'slstm':
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                c = torch.zeros(batch_size, self.hidden_size, device=x.device)
                n = torch.zeros(batch_size, self.hidden_size, device=x.device)
            else:  # mlstm
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=x.device)
                n = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden = (h, c if self.cell_type == 'slstm' else C, n)

        outputs = []
        for t in range(seq_len):
            hidden = self.lstm_cell(x[:, t, :], hidden)
            outputs.append(hidden[0])

        output = torch.stack(outputs, dim=1)

        if self.residual_proj is not None:
            x = self.residual_proj(x)

        output = self.layer_norm(output + x)
        return output, hidden


class xLSTMTS(nn.Module):
    def __init__(self, input_size: int = 1, embedding_dim: int = 64,
                 num_layers: int = 2, hidden_size: int = 64,
                 num_classes: int = 2, dropout: float = 0.1):
        super(xLSTMTS, self).__init__()

        self.embedding = nn.Linear(input_size, embedding_dim)

        self.xlstm_blocks = nn.ModuleList([
            xLSTMBlock(embedding_dim if i == 0 else hidden_size, hidden_size,
                      cell_type='slstm' if i % 2 == 0 else 'mlstm')
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        hidden_states = []
        for block in self.xlstm_blocks:
            x, hidden = block(x)
            hidden_states.append(hidden)

        x = self.dropout(x)
        x = x[:, -1, :]  # Take last time step
        x = self.classifier(x)

        return x