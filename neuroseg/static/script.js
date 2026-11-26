const by=id=>document.getElementById(id), src=x=>"data:image/png;base64,"+x;
let report=null, ready=false, busy=false;
const mask=by("overlay-mask");
function setReady(){by("run").disabled=!ready||busy;}
async function init(){
  try{
    const r=await fetch("/api/model/info");if(!r.ok)throw Error("No checkpoint installed");
    const d=await r.json();
    d.models.forEach(m=>{const o=document.createElement("option");o.value=m.architecture;o.textContent=m.architecture.replaceAll("_"," ");by("model").append(o);});
    by("model").value=d.default;by("model-status").textContent="Model ready";by("model-version").textContent="Checkpoint epoch "+d.epoch+" · validation Dice "+d.validation_dice.toFixed(3);ready=true;
  }catch(e){by("model-status").textContent=e.message;by("model-version").textContent="Train and install a checkpoint to enable prediction.";}
  setReady();
  try{
    const r=await fetch("/api/model/performance");if(!r.ok)return;
    const d=await r.json();
    by("perf-note").textContent=d.architecture+" · "+d.test_patients+" held-out patients / "+d.test_images+" slices";
    for(const [k,v] of Object.entries(d.patient_macro)){
      const el=document.createElement("div"),label=document.createElement("small"),value=document.createElement("strong");
      label.textContent=k;value.textContent=Number(v).toFixed(3);
      const ci=d.patient_bootstrap_95_ci?.[k];if(ci)label.textContent+=" (95% CI "+ci.map(x=>x.toFixed(3)).join("–")+")";
      el.append(label,value);by("perf").append(el);
    }
  }catch(e){by("perf-note").textContent="Evaluation report unavailable.";}
}
function reset(){by("result").hidden=true;by("placeholder").hidden=false;by("report").disabled=true;report=null;}
by("file").addEventListener("change",()=>{by("filename").textContent=by("file").files[0]?.name||"";reset();});
for(const name of ["gt","model","source"])by(name).addEventListener("change",reset);
const drop=by("drop");drop.addEventListener("dragover",e=>{e.preventDefault();drop.classList.add("active");});
drop.addEventListener("dragleave",()=>drop.classList.remove("active"));
drop.addEventListener("drop",e=>{e.preventDefault();drop.classList.remove("active");if(e.dataTransfer.files.length){by("file").files=e.dataTransfer.files;by("file").dispatchEvent(new Event("change"));}});
function overlay(){mask.style.opacity=Number(by("opacity").value)/100;by("opacity-value").textContent=by("opacity").value+"%";}
by("opacity").addEventListener("input",overlay);
by("zoom").addEventListener("input",()=>{by("result").style.setProperty("--zoom",Number(by("zoom").value)/100);by("zoom-value").textContent=by("zoom").value+"%";});
by("form").addEventListener("submit",async e=>{
 e.preventDefault();if(!ready||busy||!by("file").files[0])return;busy=true;setReady();reset();
 for(const name of ["file","gt","model","source"])by(name).disabled=true;
 by("status").textContent="Running segmentation…";
 try{
  const fd=new FormData();fd.append("file",by("file").files[0]);if(by("gt").files[0])fd.append("ground_truth",by("gt").files[0]);
  fd.append("architecture",by("model").value);fd.append("source",by("source").value);
  const r=await fetch("/api/predict",{method:"POST",body:fd});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==="string"?d.detail:"Prediction failed");
  by("original").src=src(d.image_b64);by("overlay-base").src=src(d.image_b64);by("mask").src=src(d.mask_b64);by("download").href=src(d.mask_b64);
  const image=new Image();image.src=src(d.mask_b64);await image.decode();
  const c=document.createElement("canvas");c.width=image.width;c.height=image.height;const ctx=c.getContext("2d");ctx.drawImage(image,0,0);const p=ctx.getImageData(0,0,c.width,c.height);
  for(let i=0;i<p.data.length;i+=4){const a=p.data[i];p.data[i]=255;p.data[i+1]=75;p.data[i+2]=90;p.data[i+3]=a;}ctx.putImageData(p,0,0);mask.src=c.toDataURL();overlay();
  by("area").textContent=d.predicted_foreground_area_pct.toFixed(2)+"%";by("dice").textContent=d.comparison?.dice.toFixed(3)??"—";by("iou").textContent=d.comparison?.iou.toFixed(3)??"—";
  by("ground-panel").hidden=!d.ground_truth_b64;if(d.ground_truth_b64)by("ground").src=src(d.ground_truth_b64);
  report={model:d.model,predicted_foreground_area_pct:d.predicted_foreground_area_pct,comparison:d.comparison??null,notice:d.notice};
  by("report").disabled=false;by("placeholder").hidden=true;by("result").hidden=false;by("status").textContent="Prediction complete.";
 }catch(e){by("status").textContent=e.message;}finally{busy=false;setReady();for(const name of ["file","gt","model","source"])by(name).disabled=false;}
});
by("report").addEventListener("click",()=>{if(!report)return;const u=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="neuroseg-report.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
init();
