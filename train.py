# ============================
# train.py
# ============================
import torch, os, time, numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from prepare_data import REGION_LIST, REGION_PATCH_COUNTS, REGION_ID_DICT, evaluate_adj, crop_optical_flow_block

def make_dataset(split_dir,crop_d,kp_d):
    Xp,Xid,Xm,Xkp,y=[],[],[],[],[]
    for cls in os.listdir(split_dir):
        cls_dir=os.path.join(split_dir,cls)
        for f in os.listdir(cls_dir):
            if f not in crop_d: continue
            patches,ids,masks,kps=[],[],[],[]
            for r in REGION_LIST:
                need=REGION_PATCH_COUNTS[r];pad=12-need
                reg_patches=crop_d[f][r][:need]
                reg_kps=kp_d[f][r][:need]
                for p in reg_patches+ [np.zeros((7,7,3))]*pad:
                    patches.append(p.flatten());ids.append(REGION_ID_DICT[r]);masks.append(1 if p.any() else 0)
                kps+=reg_kps+[(0.0,0.0)]*pad
            Xp.append(np.array(patches));Xid.append(np.array(ids));Xm.append(np.array(masks));Xkp.append(np.array(kps));y.append(int(cls))
    return Xp,Xid,Xm,Xkp,y

def train_subject(sub,data_root,crop_d,kp_d,adj,device,epochs=100,lr=5e-5,bs=128):
    tr_dir=os.path.join(data_root,sub,'u_train');te_dir=os.path.join(data_root,sub,'u_test')
    if not os.path.isdir(tr_dir):return
    Xp_tr,Xid_tr,Xm_tr,Xkp_tr,y_tr=make_dataset(tr_dir,crop_d,kp_d)
    Xp_te,Xid_te,Xm_te,Xkp_te,y_te=make_dataset(te_dir,crop_d,kp_d)
    if not Xp_tr or not Xp_te: return
    train_ds=TensorDataset(torch.tensor(Xp_tr,dtype=torch.float32),torch.tensor(Xid_tr),torch.tensor(Xm_tr,dtype=torch.float32),torch.tensor(Xkp_tr,dtype=torch.float32),torch.tensor(y_tr))
    test_ds=TensorDataset(torch.tensor(Xp_te,dtype=torch.float32),torch.tensor(Xid_te),torch.tensor(Xm_te,dtype=torch.float32),torch.tensor(Xkp_te,dtype=torch.float32),torch.tensor(y_te))
    dl_tr=DataLoader(train_ds,batch_size=bs,shuffle=True)
    dl_te=DataLoader(test_ds,batch_size=bs)
    model=HTNet(num_regions=6,patch_dim=147,region_embedding_dim=32,intra_transformer_depth=3,intra_transformer_heads=4,inter_transformer_depth=3,inter_transformer_heads=4,gcn_layers=3,num_classes=3,dropout=0.1,au_adj=adj).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr);crit=nn.CrossEntropyLoss()
    best_acc=0;best_state=None
    for ep in range(epochs):
        model.train()
        for p,rid,m, kp,lbl in dl_tr:
            p,rid,m,kp,lbl=[x.to(device) for x in (p,rid,m,kp,lbl)]
            opt.zero_grad();out=model(p,rid,m,kp);loss=crit(out,lbl);loss.backward();opt.step()
        # val
        model.eval();cor=tot=0
        with torch.no_grad():
            for p,rid,m,kp,lbl in dl_te:
                p,rid,m,kp,lbl=[x.to(device) for x in (p,rid,m,kp,lbl)]
                cor+=(model(p,rid,m,kp).argmax(1)==lbl).sum().item();tot+=lbl.size(0)
        acc=cor/tot
        if acc>best_acc:best_acc,best_state=acc,model.state_dict()
        if (ep+1)%10==0:print(f"{sub} ep{ep+1} acc={acc:.3f}")
    os.makedirs('ourmodel_weights',exist_ok=True)
    torch.save(best_state,f'ourmodel_weights/{sub}.pth')
    print(f"{sub} best acc={best_acc:.3f}")


def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adj=torch.tensor(evaluate_adj('casme_class2_for_optical_flow.csv'),dtype=torch.float32).to(device)
    crop_d,kp_d=crop_optical_flow_block()
    data_root='./datasets/three_norm_u_v_os'
    for sub in os.listdir(data_root):
        train_subject(sub,data_root,crop_d,kp_d,adj,device)

if __name__=='__main__':
    main()