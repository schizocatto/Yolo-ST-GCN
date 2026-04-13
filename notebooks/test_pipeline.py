"""
test_pipeline.py
Local smoke-test for stgcn-notebook-with-viz.
Replaces Kaggle .mat loading with synthetic .mat files that mimic
Penn Action format exactly (same keys, same shapes).
Skips cells 3 & 4 (need real video frames — Kaggle only).
Runs 3 epochs with batch_size=8.
"""

import matplotlib
matplotlib.use('Agg')   # no display needed — saves all plots as PNG

import os, glob, tempfile
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score,
                             confusion_matrix, classification_report)
import seaborn as sns
from collections import Counter
from matplotlib.patches import Patch

# ── output dir for saved plots ────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f"  saved {name}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 — Generate synthetic .mat files that mimic Penn Action format
# ═══════════════════════════════════════════════════════════════════════════
print("\n[0] Generating synthetic .mat files...")

exercise_classes = [
    'bench_press', 'clean_and_jerk', 'jump_rope', 'jumping_jacks',
    'pullup', 'pushup', 'situp', 'squat'
]

FAKE_DIR = tempfile.mkdtemp(prefix='penn_fake_')
LABELS_DIR = os.path.join(FAKE_DIR, 'labels')
os.makedirs(LABELS_DIR)

N_PER_CLASS = 6   # 6 videos per class = 48 total → 38 train / 10 val
np.random.seed(0)
vid_id = 1
for action in exercise_classes:
    for _ in range(N_PER_CLASS):
        n_frames = np.random.randint(30, 100)
        # Penn Action .mat keys: x, y (T×13), action, nframes, visibility (T×13)
        mat = {
            'x':          np.random.randint(50, 400, (n_frames, 13)).astype(float),
            'y':          np.random.randint(50, 400, (n_frames, 13)).astype(float),
            'action':     np.array([[action]]),
            'nframes':    np.array([[n_frames]]),
            'visibility': np.ones((n_frames, 13)),
        }
        scipy.io.savemat(os.path.join(LABELS_DIR, f'{vid_id:04d}.mat'), mat)
        vid_id += 1

print(f"  Created {vid_id-1} .mat files in {LABELS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — scan .mat files → DataFrame
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] Scanning .mat files into DataFrame...")

mat_files = sorted(glob.glob(os.path.join(LABELS_DIR, '*.mat')))
data_info = []
for mat_file in mat_files:
    video_id = os.path.basename(mat_file).replace('.mat', '')
    mat_content = scipy.io.loadmat(mat_file)
    action = mat_content['action'][0]
    if isinstance(action, np.ndarray) and len(action) > 0:
        action = action[0]
    nframes = mat_content['nframes'][0][0]
    data_info.append({'video_id': video_id, 'action': str(action), 'nframes': int(nframes)})

df = pd.DataFrame(data_info)
df_exercise = df[df['action'].isin(exercise_classes)].copy()
print(f"  Total videos: {len(df)}  |  Exercise subset: {len(df_exercise)}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — distribution chart (all actions)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Action distribution chart...")
plt.figure(figsize=(12, 5))
df['action'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Action distribution'); plt.tight_layout()
savefig('cell2_action_dist.png')


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — Graph_PennAction_14Nodes
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Building Graph_PennAction_14Nodes...")

class Graph_PennAction_14Nodes():
    def __init__(self):
        self.num_node = 14
        self.center_node = 13
        self.edges = [
            (0,1),(0,2),(1,3),(3,5),(2,4),(4,6),
            (1,7),(2,8),(7,8),(7,9),(9,11),(8,10),(10,12)
        ]
        self.edges.extend([(1,13),(2,13),(7,13),(8,13)])
        self.A = self.get_spatial_partition_matrix()

    def get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[j, i] = 1; A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.num_node)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.num_node - 1, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_spatial_partition_matrix(self):
        hop_dis = self.get_hop_distance()
        node_traversal_rate = hop_dis[self.center_node]
        A = np.zeros((3, self.num_node, self.num_node))
        for i, j in self.edges:
            A[0, i, i] = 1; A[0, j, j] = 1
            if node_traversal_rate[i] > node_traversal_rate[j]:
                A[1, j, i] = 1; A[2, i, j] = 1
            elif node_traversal_rate[i] < node_traversal_rate[j]:
                A[1, i, j] = 1; A[2, j, i] = 1
            else:
                A[0, i, j] = 1; A[0, j, i] = 1
        for i in range(3):
            D = np.diag(np.sum(A[i], axis=1))
            D_inv = np.zeros_like(D)
            D_inv[D > 0] = 1.0 / D[D > 0]
            A[i] = np.dot(A[i], D_inv)
        return torch.tensor(A, dtype=torch.float32)

graph = Graph_PennAction_14Nodes()
print(f"  A shape: {graph.A.shape}")
assert graph.A.shape == (3, 14, 14), "Graph shape wrong"


# ═══════════════════════════════════════════════════════════════════════════
# VIZ-1a — Skeleton graph
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VIZ-1a] Skeleton graph...")

import matplotlib.patches as mpatches

JOINT_POS = {
    0:(0.50,0.95), 1:(0.35,0.76), 2:(0.65,0.76),
    3:(0.22,0.58), 4:(0.78,0.58), 5:(0.14,0.40),
    6:(0.86,0.40), 7:(0.40,0.52), 8:(0.60,0.52),
    9:(0.38,0.30),10:(0.62,0.30),11:(0.36,0.08),
   12:(0.64,0.08),13:(0.50,0.64),
}
JOINT_NAMES = [
    'head','l_sho','r_sho','l_elbow','r_elbow',
    'l_wrist','r_wrist','l_hip','r_hip',
    'l_knee','r_knee','l_ankle','r_ankle','center*'
]

A = graph.A.numpy()
C_SAME='#3498db'; C_IN='#27ae60'; C_OUT='#e74c3c'

fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(-0.05,1.15); ax.set_ylim(-0.05,1.05); ax.axis('off')
ax.set_title('Penn Action Skeleton Graph\nSpatial Configuration Partitioning (14 joints)',
             fontsize=13, fontweight='bold', pad=12)
for (i,j) in graph.edges:
    xi,yi=JOINT_POS[i]; xj,yj=JOINT_POS[j]
    color = C_IN if (A[1,j,i]>0 or A[1,i,j]>0) else C_OUT if (A[2,j,i]>0 or A[2,i,j]>0) else C_SAME
    ax.plot([xi,xj],[yi,yj],color=color,linewidth=2.8,zorder=1,alpha=0.85)
for idx,(x,y) in JOINT_POS.items():
    c='#f39c12' if idx==13 else '#2c3e50'
    ax.scatter(x,y,s=280 if idx==13 else 200,color=c,zorder=3,edgecolors='white',linewidths=1.2)
    ha='left' if x<=0.5 else 'right'
    ax.text(x+(0.04 if x<=0.5 else -0.04),y,f'{idx}: {JOINT_NAMES[idx]}',
            fontsize=8,va='center',ha=ha,zorder=4,color='#1a1a2e',
            fontweight='bold' if idx==13 else 'normal')
patches=[
    mpatches.Patch(color=C_IN,      label='Centripetal (toward center)'),
    mpatches.Patch(color=C_OUT,     label='Centrifugal (away from center)'),
    mpatches.Patch(color=C_SAME,    label='Same distance / self-loop'),
    mpatches.Patch(color='#f39c12', label='Virtual center joint (added)'),
]
ax.legend(handles=patches,loc='lower center',fontsize=9,ncol=2,framealpha=0.9)
plt.tight_layout()
savefig('viz_skeleton_graph.png')


# ═══════════════════════════════════════════════════════════════════════════
# VIZ-1b — Adjacency matrix heatmaps
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VIZ-1b] Adjacency matrix heatmaps...")

A_np = graph.A.numpy()
titles = ['A[0]  Self-loop / Same-distance',
          'A[1]  Centripetal  (→ center)',
          'A[2]  Centrifugal  (← center)']

fig, axes = plt.subplots(1,3,figsize=(18,5.5))
fig.suptitle('ST-GCN Adjacency Matrices — Spatial Configuration Partitioning',
             fontsize=13,fontweight='bold',y=1.01)
for k, ax in enumerate(axes):
    mask_zero = (A_np[k] == 0)
    # single heatmap — let colormap handle zeros naturally (they stay white/light)
    sns.heatmap(A_np[k], ax=ax, cmap='YlOrRd',
                xticklabels=JOINT_NAMES, yticklabels=JOINT_NAMES,
                linewidths=0.4, linecolor='#e0e0e0',
                vmin=0, vmax=A_np[k].max()+0.01,
                cbar_kws={'shrink':0.75,'label':'weight'})
    ax.set_title(titles[k],fontsize=11,fontweight='bold',pad=8)
    ax.tick_params(axis='x',rotation=45,labelsize=7.5)
    ax.tick_params(axis='y',rotation=0, labelsize=7.5)
plt.tight_layout()
savefig('viz_adjacency_matrices.png')


# ═══════════════════════════════════════════════════════════════════════════
# CELL 10 — STGCN_Block
# ═══════════════════════════════════════════════════════════════════════════
print("\n[10] Defining STGCN_Block...")

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.spatial_kernel_size = A.size(0)
        self.A = nn.Parameter(A, requires_grad=False)
        self.gcn_conv = nn.Conv2d(in_channels, out_channels * self.spatial_kernel_size, kernel_size=1)
        self.tcn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(9,1),
                                  padding=(4,0), stride=(stride,1))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        N, C, T, V = x.size()
        x = self.gcn_conv(x)                                         # (N, out*3, T, V)
        x = x.view(N, self.spatial_kernel_size, -1, T, V)            # (N, 3, out, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A))             # (N, out, T, V)
        x = self.tcn_conv(x)
        return self.relu(x + res)


# ═══════════════════════════════════════════════════════════════════════════
# CELL 11 — Model_STGCN
# ═══════════════════════════════════════════════════════════════════════════
print("\n[11] Defining Model_STGCN...")

class Model_STGCN(nn.Module):
    def __init__(self, num_classes=8, in_channels=2):
        super().__init__()
        self.graph = Graph_PennAction_14Nodes()
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        self.st_gcn_networks = nn.ModuleList([
            STGCN_Block(in_channels, 64,  A, residual=False),
            STGCN_Block(64,  64,  A),
            STGCN_Block(64,  128, A, stride=2),
            STGCN_Block(128, 128, A),
            STGCN_Block(128, 256, A, stride=2),
            STGCN_Block(256, 256, A),
        ])
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0,4,3,1,2).contiguous().view(N, M*V*C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0,1,3,4,2).contiguous().view(N*M, C, T, V)
        for gcn in self.st_gcn_networks:
            x = gcn(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# CELL 12 — Sanity check
# ═══════════════════════════════════════════════════════════════════════════
print("\n[12] Sanity check forward pass...")
dummy = torch.randn(4, 2, 64, 14, 1)
model_test = Model_STGCN(num_classes=8)
out = model_test(dummy)
print(f"  Output shape: {out.shape}")
assert out.shape == (4, 8), f"Expected (4,8), got {out.shape}"


# ═══════════════════════════════════════════════════════════════════════════
# VIZ-3 — Architecture table
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VIZ-3] Architecture table...")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off'); fig.patch.set_facecolor('#fdfefe')
columns = ['Layer / Block','Out channels (C)','Frames (T)','Joints (V)','Stride','Residual']
rows = [
    ['Input skeleton',          '2  (x, y)','64','14','—','—'],
    ['BatchNorm1d',             '2',         '64','14','—','—'],
    ['ST-GCN Block 1',          '64',        '64','14','1','✗ disabled'],
    ['ST-GCN Block 2',          '64',        '64','14','1','✓'],
    ['ST-GCN Block 3 ▼',        '128',       '32','14','2','✓ (1×1 conv)'],
    ['ST-GCN Block 4',          '128',       '32','14','1','✓'],
    ['ST-GCN Block 5 ▼',        '256',       '16','14','2','✓ (1×1 conv)'],
    ['ST-GCN Block 6',          '256',       '16','14','1','✓'],
    ['Global Avg Pool',         '256',        '1', '1','—','—'],
    ['Conv2d 1×1 (classifier)', '8 (classes)','1', '1','—','—'],
]
table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.15, 2.0)
row_colors={0:'#2c3e50',1:'#ecf0f1',2:'#ecf0f1',3:'#eaf4fb',4:'#eaf4fb',
            5:'#fdebd0',6:'#eaf4fb',7:'#fdebd0',8:'#ecf0f1',9:'#ecf0f1',10:'#d5f5e3'}
for r,color in row_colors.items():
    for c in range(len(columns)):
        cell=table[(r,c)]; cell.set_facecolor(color)
        if r==0: cell.set_text_props(color='white',fontweight='bold')
for r in [5,7]:
    table[(r,0)].set_text_props(fontweight='bold',color='#a04000')
    table[(r,4)].set_text_props(fontweight='bold',color='#a04000')
ax.set_title('Model_STGCN — Tensor Shape Through Each Layer\n'
             '(▼ = stride-2 temporal downsampling,  green = final output)',
             fontsize=12,fontweight='bold',pad=18,color='#1a252f')
plt.tight_layout()
savefig('viz_model_architecture.png')


# ═══════════════════════════════════════════════════════════════════════════
# CELL 14 — Full preprocessing (uses the fake .mat files)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[14] Preprocessing .mat files...")

TARGET_FRAMES = 64
class_to_id   = {cls: idx for idx, cls in enumerate(exercise_classes)}

def add_virtual_center_joint(kpts):
    l_sho, r_sho = kpts[:,1,:], kpts[:,2,:]
    l_hip, r_hip = kpts[:,7,:], kpts[:,8,:]
    center = (l_sho + r_sho + l_hip + r_hip) / 4.0
    return np.concatenate((kpts, center[:,np.newaxis,:]), axis=1)

def temporal_align(kpts, target_frames):
    T = kpts.shape[0]
    if T == target_frames: return kpts
    return kpts[np.linspace(0, T-1, target_frames).astype(int)]

all_data, all_labels, raw_frame_counts = [], [], []

for mat_file in sorted(glob.glob(os.path.join(LABELS_DIR, '*.mat'))):
    try:
        mat_data = scipy.io.loadmat(mat_file)
        action   = mat_data['action'][0]
        if isinstance(action, np.ndarray) and len(action)>0: action = action[0]
        action   = str(action)
        if action not in exercise_classes: continue

        kpts = np.stack((mat_data['x'], mat_data['y']), axis=-1).astype(np.float32)
        raw_frame_counts.append(kpts.shape[0])
        kpts = add_virtual_center_joint(temporal_align(kpts, TARGET_FRAMES))
        tensor_data = np.expand_dims(np.transpose(kpts, (2,0,1)), axis=-1)  # (2,64,14,1)
        all_data.append(tensor_data)
        all_labels.append(class_to_id[action])
    except Exception as e:
        print(f"  Skip {os.path.basename(mat_file)}: {e}")

real_data_tensor   = np.array(all_data,   dtype=np.float32)
real_labels_tensor = np.array(all_labels, dtype=np.int64)
print(f"  Data: {real_data_tensor.shape}  Labels: {real_labels_tensor.shape}")
assert real_data_tensor.shape[1:] == (2, 64, 14, 1), "Unexpected data shape"

np.save(os.path.join(OUT_DIR, 'penn_action_data_14nodes.npy'), real_data_tensor)
np.save(os.path.join(OUT_DIR, 'penn_action_labels.npy'),       real_labels_tensor)


# ═══════════════════════════════════════════════════════════════════════════
# VIZ-2 — Data statistics + sample skeleton
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VIZ-2] Data stats + sample skeleton...")

id_to_class   = {v:k for k,v in class_to_id.items()}
label_counts  = Counter(all_labels)
counts_sorted = [label_counts[class_to_id[c]] for c in exercise_classes]
colors        = plt.cm.Set2(np.linspace(0, 1, len(exercise_classes)))

fig, axes = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Penn Action — Exercise Subset Analysis',fontsize=13,fontweight='bold')
bars = axes[0].bar(exercise_classes, counts_sorted, color=colors, edgecolor='white', linewidth=1.2)
axes[0].set_title('Sample Count per Action Class')
axes[0].set_xlabel('Action'); axes[0].set_ylabel('Number of videos')
axes[0].tick_params(axis='x',rotation=35)
for bar,cnt in zip(bars,counts_sorted):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 str(cnt),ha='center',va='bottom',fontsize=9,fontweight='bold')
axes[1].hist(raw_frame_counts,bins=15,color='#5dade2',edgecolor='white',linewidth=0.8)
axes[1].axvline(x=64,color='#e74c3c',linewidth=2,linestyle='--',label='Target T=64')
axes[1].set_title('Raw Video Length Distribution\n(before temporal alignment)')
axes[1].set_xlabel('Number of frames'); axes[1].set_ylabel('Number of videos')
axes[1].legend(fontsize=10)
plt.tight_layout()
savefig('viz_data_stats.png')

# sample skeleton
xy = all_data[0][:, 0, :, 0]  # (2, 14)
edges_13      = [(0,1),(0,2),(1,3),(3,5),(2,4),(4,6),(1,7),(2,8),(7,8),(7,9),(9,11),(8,10),(10,12)]
edges_virtual = [(1,13),(2,13),(7,13),(8,13)]
fig, ax = plt.subplots(figsize=(5,7))
ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
ax.set_title(f'Sample Skeleton — class: {id_to_class[all_labels[0]]}\nFrame 0 | ★ = virtual center',fontsize=11)
for (i,j) in edges_13:
    ax.plot([xy[0,i],xy[0,j]],[xy[1,i],xy[1,j]],color='#2c3e50',linewidth=2.5,zorder=1)
for (i,j) in edges_virtual:
    ax.plot([xy[0,i],xy[0,j]],[xy[1,i],xy[1,j]],color='#e67e22',linewidth=2.0,linestyle='--',zorder=1)
ax.scatter(xy[0,:13],xy[1,:13],s=120,color='#2980b9',zorder=3,edgecolors='white',linewidths=1.2,label='GT joint (13)')
ax.scatter(xy[0,13], xy[1,13], s=200,color='#27ae60',marker='*',zorder=4,edgecolors='white',linewidths=1.0,label='Virtual center (14th)')
for i in range(14):
    ax.text(xy[0,i]+1,xy[1,i]-1,JOINT_NAMES[i],fontsize=7.5,color='#1a1a2e',zorder=5)
ax.legend(fontsize=9,loc='upper right')
plt.tight_layout()
savefig('viz_sample_skeleton.png')


# ═══════════════════════════════════════════════════════════════════════════
# CELL 16 — Dataset + DataLoader
# ═══════════════════════════════════════════════════════════════════════════
print("\n[16] Dataset / DataLoader...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  device: {device}")

class PennActionDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data   = torch.FloatTensor(data_tensor)
        self.labels = torch.LongTensor(label_tensor)
    def __len__(self):   return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

real_data   = np.load(os.path.join(OUT_DIR, 'penn_action_data_14nodes.npy'))
real_labels = np.load(os.path.join(OUT_DIR, 'penn_action_labels.npy'))

X_train, X_val, y_train, y_val = train_test_split(
    real_data, real_labels, test_size=0.2, random_state=42, stratify=real_labels)

train_dataset = PennActionDataset(X_train, y_train)
val_dataset   = PennActionDataset(X_val,   y_val)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True,  drop_last=False)
val_loader    = DataLoader(val_dataset,   batch_size=8, shuffle=False, drop_last=False)
print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 18 — Training + all VIZ-4 plots
# ═══════════════════════════════════════════════════════════════════════════
print("\n[18] Training (3 epochs)...")

model     = Model_STGCN(num_classes=8, in_channels=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 3
history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[],'val_f1':[]}

for epoch in range(num_epochs):
    model.train()
    total_loss, preds_tr, lbls_tr = 0, [], []
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        out  = model(batch_data)
        loss = criterion(out, batch_labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        _, p = torch.max(out, 1)
        preds_tr.extend(p.cpu().numpy()); lbls_tr.extend(batch_labels.cpu().numpy())
    scheduler.step()

    model.eval()
    val_loss, preds_val, lbls_val = 0, [], []
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            out  = model(batch_data)
            loss = criterion(out, batch_labels)
            val_loss += loss.item()
            _, p = torch.max(out, 1)
            preds_val.extend(p.cpu().numpy()); lbls_val.extend(batch_labels.cpu().numpy())

    tr_acc  = accuracy_score(lbls_tr,  preds_tr)
    val_acc = accuracy_score(lbls_val, preds_val)
    val_f1  = f1_score(lbls_val, preds_val, average='macro', zero_division=0)

    history['train_loss'].append(total_loss / len(train_loader))
    history['val_loss'].append(val_loss     / len(val_loader))
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    print(f"  Epoch {epoch+1}/{num_epochs}  train_loss={history['train_loss'][-1]:.4f}"
          f"  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

torch.save(model.state_dict(), os.path.join(OUT_DIR, 'stgcn_penn_action.pth'))
print("  Model saved.")

# VIZ-4a: training curves
ep = range(1, num_epochs+1)
fig, axes = plt.subplots(1,3,figsize=(18,5))
fig.suptitle('Training History — ST-GCN on Penn Action',fontsize=13,fontweight='bold')
axes[0].plot(ep,history['train_loss'],label='Train',color='#e74c3c',linewidth=2)
axes[0].plot(ep,history['val_loss'],  label='Val',  color='#3498db',linewidth=2)
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep,history['train_acc'],label='Train',color='#e74c3c',linewidth=2)
axes[1].plot(ep,history['val_acc'],  label='Val',  color='#3498db',linewidth=2)
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0,1); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(ep,history['val_f1'],color='#8e44ad',linewidth=2)
axes[2].set_title('Validation Macro F1'); axes[2].set_xlabel('Epoch')
axes[2].set_ylim(0,1); axes[2].grid(alpha=0.3)
plt.tight_layout()
savefig('viz_training_curves.png')

# VIZ-4b: confusion matrix
conf_mat = confusion_matrix(lbls_val, preds_val)
fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=exercise_classes, yticklabels=exercise_classes,
            linewidths=0.5, linecolor='#f0f0f0', ax=ax,
            cbar_kws={'label':'count'})
ax.set_title('Confusion Matrix — Validation Set',fontsize=12,fontweight='bold',pad=12)
ax.set_xlabel('Predicted',fontsize=11); ax.set_ylabel('True Label',fontsize=11)
ax.tick_params(axis='x',rotation=35,labelsize=9)
ax.tick_params(axis='y',rotation=0, labelsize=9)
plt.tight_layout()
savefig('viz_confusion_matrix.png')

# VIZ-4c: per-class F1
f1_per_class = f1_score(lbls_val, preds_val, average=None, zero_division=0)
bar_colors   = ['#27ae60' if f>=0.8 else '#f39c12' if f>=0.5 else '#e74c3c'
                for f in f1_per_class]
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(exercise_classes, f1_per_class, color=bar_colors, edgecolor='white', linewidth=1.2)
ax.axhline(y=np.mean(f1_per_class), color='#2c3e50', linewidth=1.8, linestyle='--')
ax.set_ylim(0,1.1)
ax.set_title('Per-class F1 Score — Validation Set',fontsize=12,fontweight='bold')
ax.set_xlabel('Action class'); ax.set_ylabel('F1 Score')
ax.tick_params(axis='x',rotation=35)
for bar,f in zip(bars,f1_per_class):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{f:.2f}',ha='center',va='bottom',fontsize=9,fontweight='bold')
ax.legend(handles=[
    Patch(color='#27ae60', label='F1 >= 0.80'),
    Patch(color='#f39c12', label='0.50 <= F1 < 0.80'),
    Patch(color='#e74c3c', label='F1 < 0.50'),
    plt.Line2D([0],[0], color='#2c3e50', linewidth=2, linestyle='--',
               label=f'Macro F1 = {np.mean(f1_per_class):.3f}'),
], fontsize=9, loc='lower right')
plt.tight_layout()
savefig('viz_per_class_f1.png')

print("\n--- Classification Report ---")
print(classification_report(lbls_val, preds_val,
                             target_names=exercise_classes, zero_division=0))

print(f"\n✓ All done. Outputs in: {OUT_DIR}")
