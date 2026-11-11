"""Class-conditional attacks (attacker knows the label): z-score distance to the synthetic class mean,
and low-rank class Mahalanobis. Members = train of class c, non-members = held-out test of class c."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0,"../FactorGen"); sys.path.insert(0,".")
from factor_core import load_expression
from eval_suite import auc
OUT=os.environ["OUT"]
def scores(S, T):
    mu, sd = S.mean(0), S.std(0)+1e-6
    Zs, Zt = (S-mu)/sd, (T-mu)/sd
    s1 = -(Zt**2).sum(1)
    k = min(10, len(S)//3)
    U, s, Vt = np.linalg.svd(Zs, full_matrices=False); V = Vt[:k].T; lam = s[:k]**2/len(S)
    P = Zt@V; res = (Zt**2).sum(1)-(P**2).sum(1); rv = ((Zs**2).sum()-(s[:k]**2).sum())/len(S)
    s2 = -((P**2/lam).sum(1) + res/ (rv/ (Zs.shape[1]-k)))
    return s1, s2
rows=[]
for ds,tsv,ann,lc,dirs in [("BRCA",os.environ["BRCA_TSV"],os.environ["BRCA_ANN"],"Subtype",["FULL_BRCA/copula_diff_k30","FULL_BRCA/copula_gauss_k30","FULL_BRCA/factordiff_k50","FULL_BRCA/pgm_star_epsinf","FULL_BRCA/pgm_star_eps10","FULLDP_BRCA/dpcopula_eps10","FULLDP_BRCA/dpcopula_eps30","REAL"]),
                           ("COMB",os.environ["COMB_TSV"],os.environ["COMB_ANN"],"cancer_type",["FULL_COMB/copula_diff_k30","FULL_COMB/copula_gauss_k30","FULL_COMB/factordiff_k50","FULL_COMB/pgm_star_epsinf","FULL_COMB/pgm_star_eps10","FULLDP_COMB/dpcopula_eps10","FULLDP_COMB/dpcopula_eps30","REAL"])]:
    expr=load_expression(tsv); a=pd.read_csv(ann); lab=a.set_index(a["samplesID"].astype(str))[lc].astype(str)
    for run in dirs:
        for f in range(1,6):
            base=f"{OUT}/{dirs[0]}"; ids=json.load(open(f"{base}/split_{f}_ids.json"))
            Xtr,Xte=expr.loc[ids["train"]].values,expr.loc[ids["test"]].values; ytr,yte=lab.loc[ids["train"]].values,lab.loc[ids["test"]].values
            if run=="REAL": S,ys=Xtr,ytr
            else:
                S=pd.read_csv(f"{OUT}/{run}/synthetic_data_split_{f}.csv").values; ys=pd.read_csv(f"{OUT}/{run}/synthetic_labels_split_{f}.csv").iloc[:,0].astype(str).values
            for c in np.unique(ytr):
                Sc=S[ys==c]
                if len(Sc)<10 or (yte==c).sum()<5: continue
                i1,i2=scores(Sc,Xtr[ytr==c]); o1,o2=scores(Sc,Xte[yte==c])
                rows.append(dict(ds=ds,run=run.split("/")[-1],fold=f,cls=c,n_tr=(ytr==c).sum(),z=auc(i1,o1),lr=auc(i2,o2)))
df=pd.DataFrame(rows); df.to_csv(sys.argv[1],index=False)
pd.set_option("display.width",250)
for ds in ["BRCA","COMB"]:
    d=df[df.ds==ds]
    for m in ["z","lr"]:
        t=d.pivot_table(index="run",columns="cls",values=m,aggfunc="mean")
        t=t[d.groupby("cls").n_tr.mean().sort_values().index]; t.columns=[f"{c}({int(d[d.cls==c].n_tr.mean())})" for c in t.columns]
        t["mean"]=d.groupby("run")[m].mean()
        print(ds,"attack:",m); print(t.round(3).to_string())
