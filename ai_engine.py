import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, AutoModel # 🔥 พระเอกของเรา

# ==========================================
# 1. โหลด WangchanBERTa (แทน Vectorizer ตัวเก่า)
# ==========================================
print("🔄 กำลังโหลด WangchanBERTa...")
try:
    # ชื่อโมเดลมาตรฐานของ WangchanBERTa
    MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_model = AutoModel.from_pretrained(MODEL_NAME)
    
    # สั่งให้ BERT ไม่ต้องเรียนรู้เพิ่ม (เอามาใช้เป็นตัวแปลภาษาเฉยๆ)
    bert_model.eval()
    print("✅ โหลด WangchanBERTa สำเร็จ!")
except Exception as e:
    print(f"❌ Error loading WangchanBERTa: {e}")
    tokenizer = None
    bert_model = None

# ==========================================
# 2. Class GCNNet (ตัวเดิมของคุณ)
# ==========================================
class GCNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet, self).__init__()
        # ⚠️ ปกติ WangchanBERTa จะให้ Vector ขนาด 768
        # เช็คว่าตอนเทรนคุณใช้ GCNConv(768, ...) หรือเปล่า?
        self.conv1 = GCNConv(num_features, 16) 
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index): # 🔥 รับ x และ edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ==========================================
# 3. โหลดโมเดล GCN (ที่เทรนเสร็จแล้ว)
# ==========================================
model_gnn = None
try:
    # ⚠️ INPUT_FEATURES ต้องตรงกับตอนเทรน (WangchanBERTa ปกติคือ 768)
    INPUT_FEATURES = 768 
    
    model_gnn = GCNNet(num_features=INPUT_FEATURES, num_classes=2)
    model_gnn.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model_gnn.eval()
    print("✅ โหลด GCN Model สำเร็จ!")
except Exception as e:
    print(f"❌ Error loading GCN Model: {e}")

# ==========================================
# 4. ฟังก์ชันทำนาย
# ==========================================
def predict_news(text):
    if tokenizer is None or bert_model is None or model_gnn is None:
        return {"result": "System Error", "confidence": 0}

    try:
        # --- ขั้นตอนที่ 4.1: ใช้ WangchanBERTa แปลง Text เป็น Vector (Node Features) ---
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # เอา Vector ของ [CLS] token ตัวแรกมาใช้เป็นตัวแทนประโยค
            text_vector = outputs.last_hidden_state[:, 0, :] # Shape: [1, 768]
        
        # --- ขั้นตอนที่ 4.2: สร้าง Graph จำลอง (Edge Index) ---
        # ⚠️ จุดสำคัญ: GCN ต้องการ "กราฟ" (เส้นเชื่อม) 
        # ถ้าคุณใช้ GCN กับข่าวเดียว (Single Document) ปกติเราจะใส่ Self-loop
        # หรือถ้าคุณเทรนแบบอื่น ต้องแก้ตรงนี้ให้เหมือนตอนเทรนครับ
        
        # สร้างเส้นเชื่อมตัวเอง (Self-loop) ง่ายๆ เพื่อให้ GCN ทำงานได้
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # --- ขั้นตอนที่ 4.3: ส่งเข้า GCN ---
        with torch.no_grad():
            # ส่ง vector (x) และ edge (edge_index) เข้าไป
            pred_logits = model_gnn(text_vector, edge_index)
            
            probs = torch.exp(pred_logits) # แปลง log_softmax เป็น probability
            confidence, predicted_class = torch.max(probs, 1)
            
            label = "Fake" if predicted_class.item() == 1 else "Real" # เช็ค Label 0/1 อีกทีนะครับ
            
            return {
                "result": label,
                "confidence": confidence.item() * 100
            }

    except Exception as e:
        print(f"Prediction logic error: {e}")
        return {"result": "Error", "confidence": 0}