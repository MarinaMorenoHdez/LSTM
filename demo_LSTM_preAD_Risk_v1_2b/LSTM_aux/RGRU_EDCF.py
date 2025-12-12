import torch
import torch.nn as nn
from LSTM_aux.edcf_module import EDCFModule

class LocalMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super(LocalMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size debe ser divisible por num_heads"

        # Proyecciones lineales para Q, K, V (Ec. 13-14 del paper)
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Hidden_Size]
        batch_size, seq_len, hidden_size = x.size()
        
        # Calcular Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Atención: (Q @ K^T) / sqrt(d_k) (Ec. 19 del paper)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Aquí la "atención local" implica que atendemos a pasos anteriores
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V) # [Batch, Heads, Seq, Head_Dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Residual connection + Norm + Feed Forward (Ec. 20-23 del paper)
        x_attn = self.layer_norm(x + self.fc_out(context))
        output = self.final_layer_norm(x_attn + self.feed_forward(x_attn))
        
        return output

class RGRU_EDCF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(RGRU_EDCF, self).__init__()
        
        # 1. Módulo de Imputación (Paper sección 3.3.2)
        # Asumimos que EDCFModule toma input_dim y devuelve el mismo dim imputado
        self.edcf = EDCFModule(input_dim=input_size)
        
        # 2. Residual Sharing GRU (Paper sección 3.3.3)
        # Nota: El paper usa una proyección lineal si el input no coincide con hidden
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Local Multi-Head Attention (Paper sección 3.3.3 - b)
        self.attention = LocalMultiHeadAttention(hidden_size)
        
        # 4. Cabeceras de Predicción (Paper sección 3.3.4 - Fig 2)
        # Regresión (NMs futuras)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size) # Predice las mismas NMs
        )
        
        # Clasificación (MCI/AD conversion)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1), # Probabilidad binaria (o 3 si es multiclase)
            nn.Sigmoid()
        )

    def forward(self, x, mask_missing):
        # x: [Batch, Seq, Features]
        # mask_missing: [Batch, Seq, Features] (1 si presente, 0 si falta)
        
        # PASO 1: Imputación con EDCF
        # El paper sugiere entrenar end-to-end. Aquí pasamos cada time-step por EDCF.
        # (Simplificación: aplicamos EDCF a todo el tensor a la vez colapsando dimensiones)
        b, s, f = x.size()
        x_flat = x.view(-1, f) # [Batch*Seq, Features]
        # Nota: EDCFModule en tu código pide (x_dense, x_cat). Ajustar según tus datos.
        # Aquí asumo que x contiene todo.
        x_imputed = self.edcf(x_flat, x_flat) 
        x_imputed = x_imputed.view(b, s, f)
        
        # Mezclar imputados con reales usando la máscara (opcional, el paper implica sustitución)
        x_final = x * mask_missing + x_imputed * (1 - mask_missing)
        
        # PASO 2: Residual GRU
        x_projected = self.input_projection(x_final)
        gru_out, _ = self.gru(x_projected) # gru_out: [Batch, Seq, Hidden]
        
        # Conexión residual simple (suma entrada proyectada con salida GRU)
        res_out = gru_out + x_projected 
        
        # PASO 3: Local Attention
        attn_out = self.attention(res_out)
        
        # Tomamos el último estado para la predicción final
        # (Ojo: El paper de Jia predice secuencias, aquí predecimos el siguiente paso o estado final)
        last_step = attn_out[:, -1, :]
        
        # PASO 4: Predicción
        pred_regression = self.regressor(last_step)
        pred_classification = self.classifier(last_step)
        
        return pred_regression, pred_classification, x_imputed