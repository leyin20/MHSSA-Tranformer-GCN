# ============================
# test.py
# ============================
import os, torch, numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset,DataLoader
from prepare_data import REGION_LIST, REGION_PATCH_COUNTS, REGION_ID_DICT, evaluate_adj, crop_optical_flow_block

def make_dataset(split_dir,crop_d,kp_d):
    Xp,Xid,Xm,Xkp,y=[],[],[],[],[]
    for cls in os.listdir(split_dir):
        for f in os.listdir(os.path.join(split_dir,cls)):
            if f not in crop_d: continue
            patches,ids,masks,kps=[],[],[],[]
            for r in REGION_LIST:
                need=REGION_PATCH_COUNTS[r];pad=12-need
                reg_patches=crop_d[f][r][:need]
                reg_kps=kp_d[f][r][:need]
                for p in reg_patches+[np.zeros((7,7,3))]*pad:
                    patches.append(p.flatten());ids.append(REGION_ID_DICT[r]);masks.append(1 if p.any() else 0)
                kps+=reg_kps+[(0.0,0.0)]*pad
            Xp.append(np.array(patches));Xid.append(np.array(ids));Xm.append(np.array(masks));Xkp.append(np.array(kps));y.append(int(cls))
    return Xp,Xid,Xm,Xkp,y

def test_subject(sub,data_root,crop_d,kp_d,adj,device):
    te_dir=os.path.join(data_root,sub,'u_test')
    if not os.path.isdir(te_dir):return
    Xp,Xid,Xm,Xkp,y=make_dataset(te_dir,crop_d,kp_d)
    if not Xp: return
    dl=DataLoader(TensorDataset(torch.tensor(Xp,dtype=torch.float32),torch.tensor(Xid),torch.tensor(Xm,dtype=torch.float32),torch.tensor(Xkp,dtype=torch.float32),torch.tensor(y)),batch_size=256)
    model=HTNet(num_regions=6,patch_dim=147,region_embedding_dim=32,intra_transformer_depth=3,intra_transformer_heads=4,inter_transformer_depth=3,inter_transformer_heads=4,gcn_layers=3,num_classes=3,dropout=0.1,au_adj=adj).to(device)
    w_path=f'ourmodel_weights/{sub}.pth'
    if not os.path.exists(w_path):print(f"{w_path} 不存在");return
    model.load_state_dict(torch.load(w_path,map_location=device));model.eval()
    y_true,y_pred=[],[]
    with torch.no_grad():
        for p,r,m,k,lbl in dl:
            p,r,m,k,lbl=[x.to(device) for x in (p,r,m,k,lbl)]
            y_true+=lbl.cpu().tolist()
            y_pred+=model(p,r,m,k).argmax(1).cpu().tolist()
    print(f"{sub} 结果:\n",classification_report(y_true,y_pred,digits=4))


def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adj=torch.tensor(evaluate_adj('casme_class2_for_optical_flow.csv'),dtype=torch.float32).to(device)
    crop_d,kp_d=crop_optical_flow_block()
    data_root='./datasets/three_norm_u_v_os'
    for sub in os.listdir(data_root):
        test_subject(sub,data_root,crop_d,kp_d,adj,device)

if __name__=='__main__':
    main()
