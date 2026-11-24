import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":8,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6})
cats=["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
paper=[99.9,91.9,88.4,100,100,97.6,100,100,95.8,89.0,100,98.5,96.6,99.6,99.6]
pstd=[0.1,1.2,11.3,0,0,3.6,0,0,0.8,2.1,0.1,1.5,0.8,0.3,0.1]
ours=[100.0,77.77,91.62,100.0,99.25,99.82,100.0,100.0,92.61,69.97,100.0,90.83,84.17,99.74,97.77]
O="#2563A8"; P="#9AA5B1"
# Fig 2: per-category horizontal bars
y=np.arange(len(cats))[::-1]; h=0.38
fig,ax=plt.subplots(figsize=(3.4,3.6))
ax.barh(y+h/2,paper,h,xerr=[pstd,[min(e,100-v) for e,v in zip(pstd,paper)]],color=P,label="Paper (DINOv2-G, 672 px, 5 seeds)",error_kw=dict(lw=0.6,capsize=1.5))
ax.barh(y-h/2,ours,h,color=O,label="Ours (DINOv2-L, 448 px, seed 42)")
ax.set_yticks(y); ax.set_yticklabels(cats)
ax.set_xlim(60,101); ax.set_xlabel("Image AUROC (%)")
for yi,o,p in zip(y,ours,paper):
    if p-o>3: ax.text(o-0.4,yi-h/2,f"{o-p:+.1f}",va="center",ha="right",fontsize=6.5,color="white",fontweight="bold")
ax.legend(loc="lower left",fontsize=6.5,frameon=False,bbox_to_anchor=(0,1.0),ncol=1)
fig.tight_layout(); fig.savefig("report_figs/fig_percat.png",dpi=300); plt.close(fig)
# Fig 3: k-shot
k=[1,2,4]
fig,axs=plt.subplots(1,2,figsize=(3.4,1.75))
a=axs[0]
a.plot(k,[97.1,97.6,97.9],"o--",color=P,ms=3,lw=1,label="Paper I-AUROC")
a.plot(k,[93.57,94.36,95.21],"o-",color=O,ms=3,lw=1.2,label="Ours I-AUROC")
a.plot(k,[97.5,97.7,97.8],"s--",color=P,ms=3,lw=1,label="Paper P-AUROC")
a.plot(k,[97.21,97.38,97.58],"s-",color="#C2410C",ms=3,lw=1.2,label="Ours P-AUROC")
a.set_xticks(k); a.set_xlabel("k (normal images)"); a.set_ylabel("Mean AUROC (%)"); a.set_ylim(92.5,98.5)
a.set_title("(a) MVTec-AD mean",fontsize=7.5)
a.set_ylim(90.5,98.6); 
a.text(1.1,96.2,"pixel AUROC",fontsize=6,color="#555"); a.text(1.1,92.3,"image AUROC",fontsize=6,color="#555")
a.legend(handles=[matplotlib.lines.Line2D([],[],color=P,ls="--",lw=1,label="paper"),matplotlib.lines.Line2D([],[],color="#333",lw=1.2,label="ours")],fontsize=6,frameon=False,loc="lower right")
b=axs[1]
for name,vals,c,m in [("transistor",[84.17,91.42,92.67],"#7C3AED","o"),("cable",[77.77,79.72,83.34],"#059669","s"),("screw",[69.97,71.12,69.22],"#DC2626","^")]:
    b.plot(k,vals,"-",marker=m,color=c,ms=3,lw=1.2,label=name)
b.set_xticks(k); b.set_xlabel("k (normal images)"); b.set_ylabel("Image AUROC (%)"); b.set_ylim(48,97)
b.set_title("(b) hardest categories (ours)",fontsize=7.5)
b.legend(fontsize=5.6,frameon=False,loc="lower right",ncol=3,handlelength=1.2,columnspacing=0.6)
fig.tight_layout(w_pad=0.6); fig.savefig("report_figs/fig_kshot.png",dpi=300); plt.close(fig)
# Fig 4: memory
res=["256","448","672"]; attn=[0.17,1.63,8.19]; w=[1.22,1.22,1.22]
fig,ax=plt.subplots(figsize=(3.4,1.6))
x=np.arange(3)
ax.bar(x,w,0.5,color=P,label="weights (fp32)")
ax.bar(x,attn,0.5,bottom=w,color=O,label="stored attention maps (24 layers)")
ax.axhline(6.0,color="#DC2626",lw=0.9,ls="--"); ax.text(1.5,6.2,"6 GB GPU limit",color="#DC2626",fontsize=6.5,ha="center")
for xi,t,s in zip(x,[a+b for a,b in zip(attn,w)],["runs","runs","OOM"]):
    ax.text(xi,t+0.2,f"{t:.1f} GB ({s})",ha="center",fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([r+" px" for r in res]); ax.set_ylabel("Estimated GPU memory (GB)")
ax.set_ylim(0,11.5); ax.legend(fontsize=6,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig("report_figs/fig_memory.png",dpi=300); plt.close(fig)
print("ok")
