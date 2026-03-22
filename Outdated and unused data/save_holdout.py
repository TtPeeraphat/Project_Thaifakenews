# save_holdout.py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


# โหลด artifacts
with open('artifacts.pkl', 'rb') as f:
    arts = pickle.load(f)

x_np      = arts['x_np']        # x_train ที่บันทึกไว้
y_np      = arts['y_label_np']
y_cat_np  = arts.get('y_cat_np')

# แบ่ง holdout 15% จาก x_train
idx_all = np.arange(len(x_np))
idx_train, idx_holdout = train_test_split(
    idx_all, test_size=0.15,
    stratify=y_np, random_state=42
)

x_holdout = x_np[idx_holdout]
y_holdout  = y_np[idx_holdout]

np.save('holdout_x.npy', x_holdout)
np.save('holdout_y.npy', y_holdout)

print(f"Saved holdout_x.npy: {x_holdout.shape}")
print(f"Saved holdout_y.npy: {y_holdout.shape}")