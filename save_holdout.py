# =============================================================================
# save_holdout.py  ── เพิ่มต่อท้าย train_fixed.py (Section 11)
# =============================================================================
#
# บันทึก holdout arrays แยกออกมา เพื่อให้ evaluate_inductive.py ใช้ได้
#
# ทำไมต้อง save holdout แยก:
# - evaluate_inductive.py รันแยกจาก training
# - ต้องการ holdout ที่แน่ใจ 100% ว่าไม่เคยอยู่ใน graph
# - บันทึก random_state=42 ด้วยเพื่อ reproducibility
# =============================================================================

import numpy as np
import sys
import os


# เพิ่มใน Section 11 ของ train_fixed.py หลังจาก save artifacts

np.save("holdout_x.npy", x_holdout)
np.save("holdout_y.npy", y_holdout)
print(f"Saved holdout_x.npy: {x_holdout.shape}")
print(f"Saved holdout_y.npy: {y_holdout.shape}")

# Optional: บันทึก category ของ holdout ด้วย (สำหรับ per-category analysis)
y_cat_holdout = y_cat_bal[idx_holdout]
np.save("holdout_y_cat.npy", y_cat_holdout)
print(f"Saved holdout_y_cat.npy: {y_cat_holdout.shape}")
