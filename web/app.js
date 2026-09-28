const BASE="https://raw.githubusercontent.com/Evan-CRD/egfr-resistance-predictor/main/strong_model_outputs/";
const URLS={oof:BASE+"best_model_oof_predictions.csv",comparison:BASE+"feature_set_comparison.csv",meta:BASE+"best_model_metadata.json"};
let oof=[], comparison=[], meta={}, currentRows=[];

function parseCSV(text){
  const lines=text.trim().split(/\r?\n/); const headers=lines.shift().split(",");
  return lines.filter(Boolean).map(line=>{
    const vals=[]; let s="",q=false;
    for(let i=0;i<line.length;i++){const c=line[i]; if(c=='"'){q=!q}else if(c==","&&!q){vals.push(s);s=""}else{s+=c}} vals.push(s);
    return Object.fromEntries(headers.map((h,i)=>[h,vals[i]??""]));
  });
}
function num(v){return Number(v)}
function fmt(v,n=3){return Number.isFinite(Number(v))?Number(v).toFixed(n):"—"}
function mean(a){return a.reduce((x,y)=>x+y,0)/a.length}
function rank(a){const sorted=[...a].map((v,i)=>[v,i]).sort((x,y)=>x[0]-y[0]);const r=new Array(a.length);let i=0;while(i<sorted.length){let j=i;while(j+1<sorted.length&&sorted[j+1][0]===sorted[i][0])j++;const avg=(i+j+2)/2;for(let k=i;k<=j;k++)r[sorted[k][1]]=avg;i=j+1}return r}
function pearson(a,b){const ma=mean(a),mb=mean(b);let n=0,da=0,db=0;for(let i=0;i<a.length;i++){const x=a[i]-ma,y=b[i]-mb;n+=x*y;da+=x*x;db+=y*y}return n/Math.sqrt(da*db)}
function spearman(a,b){return pearson(rank(a),rank(b))}
function metric(label,value){return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`}

document.querySelectorAll(".nav").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".nav").forEach(x=>x.classList.remove("active")); b.classList.add("active");
  document.querySelectorAll(".page").forEach(x=>x.classList.remove("active")); document.getElementById(b.dataset.page).classList.add("active");
});

async function load(){
  try{
    const [ot,ct,mr]=await Promise.all([fetch(URLS.oof).then(r=>r.text()),fetch(URLS.comparison).then(r=>r.text()),fetch(URLS.meta).then(r=>r.json())]);
    oof=parseCSV(ot); comparison=parseCSV(ct); meta=mr;
    initPrediction(); initComparison(); initValidation(); initSource();
    document.getElementById("status").classList.add("ok");
  }catch(e){document.getElementById("status").innerHTML="Could not load project data from GitHub. Refresh the page or check the repository connection."; console.error(e)}
}
function initPrediction(){
  const muts=[...new Set(oof.map(r=>r.mutation))].sort();
  const sel=document.getElementById("mutationSelect"); sel.innerHTML=muts.map(m=>`<option>${m}</option>`).join("");
  sel.onchange=updateMutation; updateMutation(); document.getElementById("showProfile").onclick=showProfile;
}
function updateMutation(){
  const m=document.getElementById("mutationSelect").value, rows=oof.filter(r=>r.mutation===m), r=rows[0]||{};
  document.getElementById("groupField").value=r.structure_group||"—"; document.getElementById("exonField").value=r.exon1||"Not specified";
  document.getElementById("mutationNote").innerHTML=`<strong>${m}</strong> is linked to the published <strong>${r.structure_group||"—"}</strong> structure–function class.`;
  document.getElementById("profileResults").classList.add("hidden");
}
function showProfile(){
  const m=document.getElementById("mutationSelect").value; currentRows=oof.filter(r=>r.mutation===m).map(r=>({...r,response:num(r.response),prediction:num(r.prediction)})).sort((a,b)=>b.response-a.response);
  const errors=currentRows.map(r=>Math.abs(r.response-r.prediction)), a=currentRows.map(r=>r.response), b=currentRows.map(r=>r.prediction);
  document.getElementById("mGroup").textContent=currentRows[0]?.structure_group||"—"; document.getElementById("mDrugs").textContent=currentRows.length;
  document.getElementById("mMae").textContent=fmt(mean(errors)); document.getElementById("mSpearman").textContent=fmt(spearman(a,b));
  renderBars(currentRows); renderProfileTable(currentRows); document.getElementById("profileResults").classList.remove("hidden");
}
function renderBars(rows){
  const vals=rows.flatMap(r=>[r.response,r.prediction]), min=Math.min(...vals,0),max=Math.max(...vals,0), span=max-min||1;
  const h=v=>Math.max(3,Math.abs(v)*210/Math.max(Math.abs(min),Math.abs(max)));
  document.getElementById("profileChart").innerHTML=rows.map(r=>`<div class="bar-group" title="${r.drug}: experimental ${fmt(r.response)}, predicted ${fmt(r.prediction)}"><div class="bar exp" style="height:${h(r.response)}px"></div><div class="bar pred" style="height:${h(r.prediction)}px"></div><span class="bar-label">${r.drug}</span></div>`).join("");
  document.getElementById("profileChart").insertAdjacentHTML("beforebegin",'<div class="legend"><span class="dot exp"></span>Experimental <span class="dot pred"></span>Held-out prediction</div>');
}
function renderProfileTable(rows){
  const t=document.getElementById("profileTable"); t.innerHTML="<thead><tr><th>Drug</th><th>Experimental log₂ ratio</th><th>Held-out predicted log₂ ratio</th><th>Absolute error</th><th>Experimental fold vs WT</th><th>Predicted fold vs WT</th></tr></thead><tbody>"+rows.map(r=>`<tr><td>${r.drug}</td><td>${fmt(r.response)}</td><td>${fmt(r.prediction)}</td><td>${fmt(Math.abs(r.response-r.prediction))}</td><td>${fmt(2**r.response,2)}</td><td>${fmt(2**r.prediction,2)}</td></tr>`).join("")+"</tbody>";
}
document.getElementById("downloadCsv").onclick=()=>{
  const head="drug,structure_group,experimental_value,held_out_prediction,absolute_error,experimental_fold_vs_WT,held_out_fold_vs_WT\n";
  const body=currentRows.map(r=>[r.drug,r.structure_group,r.response,r.prediction,Math.abs(r.response-r.prediction),2**r.response,2**r.prediction].join(",")).join("\n");
  const blob=new Blob([head+body],{type:"text/csv"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=document.getElementById("mutationSelect").value+"_experimental_and_predicted_profile.csv";a.click();URL.revokeObjectURL(a.href);
};
function initComparison(){
  const by=Object.fromEntries(comparison.map(r=>[r.model,r])); const ex=by.drug_plus_exon, st=by.drug_plus_structure;
  if(ex&&st) document.getElementById("comparisonMetrics").innerHTML=metric("Exon model R²",fmt(ex.R2))+metric("Structure-group R²",fmt(st.R2))+metric("Exon Spearman",fmt(ex.Spearman))+metric("Structure-group Spearman",fmt(st.Spearman));
  const cols=["model","n_features","MAE","RMSE","R2","Spearman"]; document.getElementById("comparisonTable").innerHTML="<thead><tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead><tbody>"+comparison.map(r=>"<tr>"+cols.map(c=>`<td>${c==="model"?r[c]:fmt(r[c])}</td>`).join("")+"</tr>").join("")+"</tbody>";
  const vals=comparison.flatMap(r=>[num(r.R2),num(r.Spearman)]); const max=Math.max(...vals,1);
  document.getElementById("comparisonChart").innerHTML=comparison.map(r=>`<div class="bar-group" title="${r.model}"><div class="bar exp" style="height:${Math.max(3,num(r.R2)/max*210)}px"></div><div class="bar pred" style="height:${Math.max(3,num(r.Spearman)/max*210)}px"></div><span class="bar-label">${r.model}</span></div>`).join("");
  document.getElementById("comparisonChart").insertAdjacentHTML("beforebegin",'<div class="legend"><span class="dot exp"></span>R² <span class="dot pred"></span>Spearman</div>');
}
function initValidation(){
  const a=oof.map(r=>num(r.response)),b=oof.map(r=>num(r.prediction)), errors=a.map((v,i)=>Math.abs(v-b[i])), rmse=Math.sqrt(mean(a.map((v,i)=>(v-b[i])**2))), ma=mean(a), sse=a.reduce((s,v,i)=>s+(v-b[i])**2,0),sst=a.reduce((s,v)=>s+(v-ma)**2,0),r2=1-sse/sst;
  document.getElementById("validationMetrics").innerHTML=metric("R²",fmt(r2))+metric("Spearman",fmt(spearman(a,b)))+metric("MAE",fmt(mean(errors)))+metric("RMSE",fmt(rmse));
  const all=[...a,...b],mn=Math.min(...all),mx=Math.max(...all),span=mx-mn||1, box=document.getElementById("scatterPlot");
  box.innerHTML=oof.map(r=>{const x=(num(r.response)-mn)/span*94+3,y=97-(num(r.prediction)-mn)/span*94;return `<i class="point" style="left:${x}%;top:${y}%" title="${r.mutation} · ${r.drug}"></i>`}).join("")+'<span class="axis-label" style="bottom:5px;left:42%">Experimental response →</span><span class="axis-label" style="left:6px;top:8px">Held-out prediction ↑</span>';
}
function initSource(){
  const muts=new Set(oof.map(r=>r.mutation)).size, drugs=new Set(oof.map(r=>r.drug)).size;
  document.getElementById("sourceMetrics").innerHTML=metric("Mutation labels",meta.n_mutations??muts)+metric("TKIs",meta.n_drugs??drugs)+metric("Mutation–drug rows",(meta.n_rows??oof.length).toLocaleString())+metric("Response used","Median replicate");
}
load();