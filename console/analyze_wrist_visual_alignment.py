"""Compare actual RTX renders with recorded images using explicit ROI metrics."""
import json
import importlib.util
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sim_validation/wrist_visual_alignment_20261008'
module=importlib.util.spec_from_file_location('fit_visual',ROOT/'fit_wrist_visual_alignment.py')
fit=importlib.util.module_from_spec(module);module.loader.exec_module(fit)
ROIS={'left':[365,315,445,390],'right':[280,310,400,400]}

def detect_near(rgb,kind,reference_uv):
    # The previous blue CAD jaws also pass the cup color threshold. Select
    # the closest connected component to a fixed recorded image landmark for
    # this offline comparison only; nothing here enters live observations.
    r,g,b=np.moveaxis(rgb.astype(float),-1,0)
    mask=((b-r>18)&(b-g>1)&(g-r>8)&(g>65) if kind=='cup' else
          (g-r>7)&(g-b>3)&(g>80)&(r>60))
    n,lab,stats,centres=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    valid=[i for i in range(1,n) if stats[i,4]>150]
    if not valid:return None
    i=min(valid,key=lambda j:np.linalg.norm(centres[j]-reference_uv))
    y,x=np.where(lab==i)
    return dict(uv=centres[i].tolist(),area=int(stats[i,4]),
                bbox=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],
                rgb_median=np.median(rgb[lab==i],axis=0).tolist())

def metrics(rgb,side,reference=None):
    x0,y0,x1,y1=ROIS[side];y=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    values=dict(table_roi_xyxy=ROIS[side],table_roi_luma=float(np.median(y[y0:y1,x0:x1])),
                near_black_fraction=float((rgb.max(axis=-1)<5).mean()),
                clipped_fraction=float((rgb.min(axis=-1)>250).mean()),
                upper_quarter_luma_std=float(y[:120].std()))
    for k in ['cup','plate']:
        try:values[k]=(detect_near(rgb,k,reference[k]['uv']) if reference else fit.detect(rgb,k))
        except ValueError:values[k]=None
    return values

def main():
    data={};rows=[]
    for side in ['left','right']:
        for frame in [0,15,30]:
            names=['train','legacy','aligned'];cells=[]
            for phase in names:
                path=OUT/(f'train_{side}_{frame:03}.png' if phase=='train' else f'renders/{phase}_{frame:03}_{side}_wrist.png')
                bgr=cv2.imread(str(path));assert bgr is not None,path
                key=f'{side}_{frame:03}';data.setdefault(key,{})
                data[key][phase]=metrics(bgr[:,:,::-1],side,data[key].get('train'))
                cell=cv2.resize(bgr,(480,360));label=f'{side} frame {frame} | '+{'train':'Recorded','legacy':'Before','aligned':'Aligned'}[phase]
                cv2.rectangle(cell,(0,0),(480,28),(255,255,255),-1);cv2.putText(cell,label,(8,20),0,.52,(20,20,20),1,cv2.LINE_AA);cells.append(cell)
            for phase in ['legacy','aligned']:
                errors={}
                for k in ['cup','plate']:
                    a=data[key]['train'][k];b=data[key][phase][k]
                    if a and b:errors[k]=float(np.linalg.norm(np.array(a['uv'])-b['uv']))
                data[key][phase]['centroid_error_px']=errors
            rows.append(np.hstack(cells))
    cv2.imwrite(str(OUT/'wrist_comparison_all.jpg'),np.vstack(rows))
    cv2.imwrite(str(OUT/'wrist_comparison_initial.png'),np.vstack([rows[0],rows[3]]))
    (OUT/'render_metrics.json').write_text(json.dumps(data,indent=2)+'\n')
    summary={}
    for side in ['left','right']:
        d=data[f'{side}_000'];summary[side]={p:{k:v for k,v in d[p].items() if k in ['centroid_error_px','table_roi_luma','near_black_fraction','clipped_fraction']} for p in d}
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
