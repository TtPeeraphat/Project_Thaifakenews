import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import re

# ==========================================
# 1. CONFIG & MODEL CLASS
# ==========================================
MODEL_PATH = "best_model.pth"  # ชื่อไฟล์โมเดลของคุณ
BERT_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=2): 
        # ✅ แก้ hidden_channels เป็น 256 ตามที่ Model เทรนมา
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2. Second Layer
        x = self.conv2(x, edge_index)
        
        # 3. Pooling (รวม Node เป็น Graph เดียว)
        x = global_mean_pool(x, batch) 
        
        return x

# ==========================================
# 2. LOAD RESOURCES (โหลดครั้งเดียว)
# ==========================================
device = torch.device('cpu') # บังคับใช้ CPU เพื่อความชัวร์
print("🔄 Loading AI Models...")

try:
    # โหลด Tokenizer & BERT
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    
    # โหลด GNN Model
    model = GCNNet(hidden_channels=256).to(device) # ✅ ต้องตรงกับ Class ข้างบน
    
    # โหลด State Dict (น้ำหนักโมเดล)
    # map_location=device ช่วยกัน Error กรณีเทรนบน GPU แต่มารันบน CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("✅ AI Models Loaded Successfully!")
    
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model = None # Mark as failed

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================
def predict_news(text: str):
    """
    รับ Text ข่าว -> แปลงเป็น Graph -> ทำนายผล
    """
    if model is None:
        return {"result": "System Error", "confidence": 0.0}

    try:
        # A. Preprocessing (BERT Embedding)
        inputs = tokenizer(text, return_tensors="pt", max_length=400, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # ใช้ embedding ของ [CLS] token หรือ mean pooling
            node_features = bert_output.last_hidden_state.mean(dim=1) # Shape: [1, 768]

        # B. Construct Graph Data (Simple Single Node Graph)
        # เนื่องจากเราทำนาย 1 ข่าว เราจะสร้าง Graph ที่มี 1 Node
        x = node_features 
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device) # Self-loop
        
        data = Data(x=x, edge_index=edge_index).to(device)
        data.batch = torch.tensor([0], dtype=torch.long).to(device) # Batch index for 1 graph

        # C. Prediction
        with torch.no_grad():
            out = model(data) # Shape: [1, 2]
            temperature = 2.0 
            probs = F.softmax(out / temperature, dim=1)
            
            # ค่าความมั่นใจ
            confidence_fake = probs[0][0].item() * 100
            confidence_real = probs[0][1].item() * 100
            
            # ตัดสินผล
            pred_idx = torch.argmax(out, dim=1).item()
            
            if pred_idx == 0:
                return {"result": "Fake", "confidence": confidence_fake}
            else:
                return {"result": "Real", "confidence": confidence_real}

    except Exception as e:
        print(f"❌ Prediction Error: {e}") # ดู Error ตรงนี้ใน Terminal
        return {"result": "Error", "confidence": 0.0}
    