import os
import importlib.util, json, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
V4=Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "/0-temporal-feature-audit-v4-allminmax-3seed10fold")
sys.argv=["x"]
spec=importlib.util.spec_from_file_location("m28", V4/"28-edge-earlystop-repair-v1.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
base,plc=m.base,m.plc
legacy=base.load_legacy_driver(); _,specs=base.resolve_condition_specs(legacy)
q=list(json.loads((m.SRC/"metadata"/"feature_contract.json").read_text())["q_features"])
df=plc.augmented_prepared(base,{"prepared":m.SRC/"prepared_data"},"edge_iiotset")
rows=[]
for cond in ["A_q_t","B_q"]:
    sp=specs[cond]; f=list(sp["features"])
    sc=MinMaxScaler().fit(base.condition_raw_matrix(df[df.split=="fit_normal"],f,sp["preprocess"]))
    _,xcal=base.transform_partition(df,"calibration_normal",sp,sc); tp,xt=base.transform_partition(df,"test",sp,sc)
    y=tp.y.to_numpy(); qi=base.q_indices(f,q)
    j=f.index("frame.time_epoch") if "frame.time_epoch" in f else None
    for mdl in ["Dense_AE","LSTM_AE","CNN_AE","SNN_AE"]:
        for s in [0,1,2]:
            rd=m.OUT/"runs"/"FITVAL"/cond/mdl/f"seed_{s:02d}"
            model=base.build_model(legacy,mdl,len(f),"edge_iiotset",0,compile_model=False); model.load_weights(rd/"model.weights.h5")
            cq,_,_=base.reconstruction_scores(xcal,base.predict_flat(legacy,model,xcal,mdl),qi)
            tq,_,_=base.reconstruction_scores(xt,base.predict_flat(legacy,model,xt,mdl),qi)
            thr=base.percentile_threshold(cq); r={"cond":cond,"model":mdl,"seed":s,"thr":thr,"fnr":100*np.mean(tq[y==1]<=thr),"fpr":100*np.mean(tq[y==0]>thr)}
            if j is not None:
                t=xcal[:,j]; ins=(t>=0)&(t<=1)
                r["cal_q_median_tOOS"]=float(np.median(cq[~ins])); r["cal_q_median_tIN"]=float(np.median(cq[ins]))
                thr_in=base.percentile_threshold(cq[ins]); r["thr_inSupportCal"]=thr_in
                r["fnr_inSupportCal"]=100*np.mean(tq[y==1]<=thr_in); r["fpr_inSupportCal"]=100*np.mean(tq[y==0]>thr_in)
                ta=xt[y==1,j]; r["attack_t_median"]=float(np.median(ta))
                r["attack_q_median"]=float(np.median(tq[y==1]))
            else:
                r["attack_q_median"]=float(np.median(tq[y==1]))
            rows.append(r)
d=pd.DataFrame(rows)
d.to_csv("$TEMPORAL_AUDIT_ROOT/temporal_audit_edge_earlystop_repair_v1/posthoc/threshold_inflation_diagnostic_runs.csv",index=False); d.groupby(["cond","model"]).mean(numeric_only=True).drop(columns="seed").to_csv("$TEMPORAL_AUDIT_ROOT/temporal_audit_edge_earlystop_repair_v1/posthoc/threshold_inflation_diagnostic_summary.csv"); print(d.groupby(["cond","model"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
