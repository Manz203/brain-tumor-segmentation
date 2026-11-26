const by=id=>document.getElementById(id), src=x=>"data:image/png;base64,"+x;
let report=null, ready=false, busy=false, models=[], benchmark=null, samples=[];
const keys=["dice","iou","precision","recall","specificity"];
const fmt=v=>typeof v==="number"?v.toFixed(3):"N/A";
const pretty=v=>v==="unet"?"U-Net":"Attention U-Net";
function metric(container,label,value){const el=document.createElement("div"),l=document.createElement("small"),v=document.createElement("strong");l.textContent=label;v.textContent=value;el.append(l,v);container.append(el);}
function refreshModel(){const m=models.find(x=>x.architecture===by("model").value);if(m)by("model-version").textContent=pretty(m.architecture)+" · epoch "+m.epoch+" · validation Dice "+fmt(m.validation_dice);by("perf").replaceChildren();if(!benchmark){by("perf-note").textContent="No verified benchmark report installed.";return;}if(benchmark.architecture!==by("model").value){by("perf-note").textContent="No held-out test report for "+pretty(by("model").value)+". Only "+pretty(benchmark.architecture)+" (the selected champion) was tested. Validation scores are above.";return;}by("perf-note").textContent=pretty(benchmark.architecture)+" · "+benchmark.test_patients+" held-out patients / "+benchmark.test_images+" slices";for(const k of keys){const ci=benchmark.patient_bootstrap_95_ci?.[k];metric(by("perf"),k+(ci?" (95% CI "+ci.map(fmt).join("–")+")":""),fmt(benchmark.patient_macro[k]));}}
const mask=by("overlay-mask");
function setReady(){by("run").disabled=!ready||busy;}
async function init(){
  try{
    const r=await fetch("/api/model/info");if(!r.ok)throw Error("No checkpoint installed");
    const d=await r.json();
    models=d.models;
    d.models.forEach(m=>{const o=document.createElement("option");o.value=m.architecture;o.textContent=pretty(m.architecture);by("model").append(o);});
    by("model").value=d.default;by("model-status").textContent="Model ready";ready=true;
    models.forEach(m=>metric(by("validation"),pretty(m.architecture)+" validation Dice",fmt(m.validation_dice)));
    refreshModel();
  }catch(e){by("model-status").textContent=e.message;by("model-version").textContent="Train and install a checkpoint to enable prediction.";}
  setReady();
  try{const r=await fetch("/api/model/performance");if(r.ok)benchmark=await r.json();}catch(e){}
  refreshModel();
  try{const r=await fetch("/api/samples");if(r.ok){const d=await r.json();samples=d.samples||[];for(const sample of samples){const o=document.createElement("option");o.value=sample.id;o.textContent=sample.title;by("sample").append(o);}by("sample-note").textContent=samples.length?"Cheng dataset (CC BY 4.0). Samples are functional demos; split membership is not verified.":"No annotated samples installed. Upload your own matching mask.";}}catch(e){}

}
function reset(){by("result").hidden=true;by("placeholder").hidden=false;by("report").disabled=true;report=null;}
by("file").addEventListener("change",()=>{by("filename").textContent=by("file").files[0]?.name||"";reset();});
for(const name of ["gt","source"])by(name).addEventListener("change",reset);
by("model").addEventListener("change",()=>{reset();refreshModel();});
by("sample").addEventListener("change",()=>{by("sample-load").disabled=!by("sample").value||busy;});
by("sample-load").addEventListener("click",async()=>{if(busy||!by("sample").value)return;busy=true;setReady();by("sample-load").disabled=true;by("status").textContent="Loading matched sample…";try{const id=encodeURIComponent(by("sample").value);const [a,b]=await Promise.all(["image","mask"].map(kind=>fetch("/api/samples/"+id+"/"+kind)));if(!a.ok||!b.ok)throw Error("Sample not available");for(const [input,response,kind] of [["file",a,"image"],["gt",b,"mask"]]){const dt=new DataTransfer();dt.items.add(new File([await response.blob()],id+"_"+kind+".png",{type:"image/png"}));by(input).files=dt.files;}by("source").value="preprocessed";by("file").dispatchEvent(new Event("change"));by("status").textContent="Matched MRI and annotation loaded. Click Run segmentation.";}catch(e){by("status").textContent=e.message;}finally{busy=false;setReady();by("sample-load").disabled=!by("sample").value;}});
const drop=by("drop");drop.addEventListener("dragover",e=>{e.preventDefault();drop.classList.add("active");});
drop.addEventListener("dragleave",()=>drop.classList.remove("active"));
drop.addEventListener("drop",e=>{e.preventDefault();drop.classList.remove("active");if(e.dataTransfer.files.length){by("file").files=e.dataTransfer.files;by("file").dispatchEvent(new Event("change"));}});
function overlay(){mask.style.opacity=Number(by("opacity").value)/100;by("opacity-value").textContent=by("opacity").value+"%";}
by("opacity").addEventListener("input",overlay);
by("zoom").addEventListener("input",()=>{by("result").style.setProperty("--zoom",Number(by("zoom").value)/100);by("zoom-value").textContent=by("zoom").value+"%";});
by("form").addEventListener("submit",async e=>{
 e.preventDefault();if(!ready||busy||!by("file").files[0])return;busy=true;setReady();reset();
 for(const name of ["file","gt","model","source","sample","sample-load"])by(name).disabled=true;
 by("status").textContent="Running segmentation…";
 try{
  const fd=new FormData();fd.append("file",by("file").files[0]);if(by("gt").files[0])fd.append("ground_truth",by("gt").files[0]);
  fd.append("architecture",by("model").value);fd.append("source",by("source").value);
  const r=await fetch("/api/predict",{method:"POST",body:fd});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==="string"?d.detail:"Prediction failed");
  by("original").src=src(d.image_b64);by("overlay-base").src=src(d.image_b64);by("mask").src=src(d.mask_b64);by("download").href=src(d.mask_b64);
  const image=new Image();image.src=src(d.mask_b64);await image.decode();
  const c=document.createElement("canvas");c.width=image.width;c.height=image.height;const ctx=c.getContext("2d");ctx.drawImage(image,0,0);const p=ctx.getImageData(0,0,c.width,c.height);
  for(let i=0;i<p.data.length;i+=4){const a=p.data[i];p.data[i]=255;p.data[i+1]=75;p.data[i+2]=90;p.data[i+3]=a;}ctx.putImageData(p,0,0);mask.src=c.toDataURL();overlay();
  by("area").textContent=d.predicted_foreground_area_pct.toFixed(2)+"%";by("latency").textContent=d.inference_ms.toFixed(1)+" ms";const cpt=d.comparison;for(const k of keys)by(k).textContent=cpt?fmt(cpt[k]):"—";by("gt-area").textContent=cpt?cpt.ground_truth_area_pct.toFixed(2)+"%":"—";by("live-note").textContent=cpt?(cpt.empty_both?"Both masks empty; Dice/IoU defined as 1, undefined ratios shown as N/A.":"Calculated now against the uploaded ground truth at original image resolution. N/A means the denominator is zero."):"No ground truth supplied; accuracy cannot be calculated.";by("pixel-counts").textContent=cpt?"TP "+cpt.tp+" · FP "+cpt.fp+" · FN "+cpt.fn+" · TN "+cpt.tn:"";by("error-panel").hidden=!d.error_map_b64;by("legend").hidden=!d.error_map_b64;if(d.error_map_b64)by("error-map").src=src(d.error_map_b64);
  by("ground-panel").hidden=!d.ground_truth_b64;if(d.ground_truth_b64)by("ground").src=src(d.ground_truth_b64);
  report={model:d.model,predicted_foreground_area_pct:d.predicted_foreground_area_pct,comparison:d.comparison??null,inference_ms:d.inference_ms,request_ms:d.request_ms,notice:d.notice};
  by("report").disabled=false;by("placeholder").hidden=true;by("result").hidden=false;by("status").textContent="Prediction complete.";
 }catch(e){by("status").textContent=e.message;}finally{busy=false;setReady();for(const name of ["file","gt","model","source","sample","sample-load"])by(name).disabled=false;by("sample-load").disabled=!by("sample").value;}
});
by("report").addEventListener("click",()=>{if(!report)return;const u=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="neuroseg-report.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
init();
