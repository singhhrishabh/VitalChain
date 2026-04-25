// VitalChain-Env Dashboard — Interactive Logic
(function(){
'use strict';

// ── Particles ──
const pc=document.getElementById('particles');
function spawnParticle(){
  const p=document.createElement('div');
  p.className='particle';
  const s=Math.random()*6+3;
  p.style.cssText=`width:${s}px;height:${s}px;left:${Math.random()*100}%;animation-duration:${Math.random()*12+8}s;animation-delay:${Math.random()*5}s`;
  pc.appendChild(p);
  setTimeout(()=>p.remove(),20000);
}
setInterval(spawnParticle,800);
for(let i=0;i<12;i++) spawnParticle();

// ── Nav scroll ──
const nav=document.getElementById('nav');
window.addEventListener('scroll',()=>{
  nav.classList.toggle('scrolled',window.scrollY>60);
});

// ── Mobile toggle ──
const toggle=document.getElementById('nav-toggle');
const links=document.getElementById('nav-links');
if(toggle) toggle.addEventListener('click',()=>{
  links.style.display=links.style.display==='flex'?'none':'flex';
  links.style.flexDirection='column';
  links.style.position='absolute';
  links.style.top='60px';
  links.style.right='24px';
  links.style.background='white';
  links.style.padding='16px';
  links.style.borderRadius='12px';
  links.style.boxShadow='0 8px 32px rgba(0,0,0,.1)';
});

// ── Reveal on scroll ──
const reveals=document.querySelectorAll('[data-reveal]');
const io=new IntersectionObserver((entries)=>{
  entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('revealed');io.unobserve(e.target);}});
},{threshold:.15});
reveals.forEach(el=>io.observe(el));

// ── Smooth scroll for anchors ──
document.querySelectorAll('a[href^="#"]').forEach(a=>{
  a.addEventListener('click',e=>{
    e.preventDefault();
    const t=document.querySelector(a.getAttribute('href'));
    if(t) t.scrollIntoView({behavior:'smooth',block:'start'});
  });
});

// ── Live Demo ──
const BASE='';
let currentObs=null;

const statePanel=document.getElementById('state-panel');
const actionsPanel=document.getElementById('actions-panel');
const rewardsPanel=document.getElementById('rewards-panel');
const stepBadge=document.getElementById('step-badge');
const actionCount=document.getElementById('action-count');
const totalReward=document.getElementById('total-reward');
const logBody=document.getElementById('log-body');
const btnReset=document.getElementById('btn-reset');
const btnOracle=document.getElementById('btn-oracle');
const taskSelect=document.getElementById('task-select');

function log(msg){
  const d=document.createElement('div');
  d.className='log-entry';
  const t=new Date().toLocaleTimeString();
  d.innerHTML=`<span class="log-time">${t}</span>${msg}`;
  logBody.prepend(d);
}

document.getElementById('log-clear').addEventListener('click',()=>{logBody.innerHTML='';});

function renderState(obs){
  currentObs=obs;
  stepBadge.textContent=`Step ${obs.step_number}`;
  actionCount.textContent=`${obs.available_actions.length} actions`;
  
  // Inventory
  let inv='';
  if(obs.inventory_summary&&obs.inventory_summary.length){
    obs.inventory_summary.forEach(i=>{
      inv+=`<div class="inv-item"><span class="inv-type">${i.type} (${i.blood_type})</span><span class="inv-detail">${i.units}u · ${i.expiry_hours}h left</span></div>`;
    });
  } else inv='<div class="inv-item"><span class="inv-detail">Empty inventory</span></div>';
  
  // Patients
  let pat='';
  if(obs.patient_queue&&obs.patient_queue.length){
    obs.patient_queue.forEach(p=>{
      const cls=p.urgency>=4?'critical':p.urgency>=5?'dying':'';
      const stars='!'.repeat(p.urgency);
      pat+=`<div class="patient-item ${cls}"><div class="patient-name">${stars} ${p.urgency_name} — Patient ${p.patient_id}</div><div class="patient-meta">Needs: ${p.needs.join(', ')} · Blood: ${p.blood_type} · Waiting: ${p.hours_waiting}h</div></div>`;
    });
  } else pat='<div class="patient-item"><div class="patient-meta">No patients in queue</div></div>';
  
  statePanel.innerHTML=`<div style="margin-bottom:14px"><strong style="font-size:.75rem;text-transform:uppercase;letter-spacing:1px;color:#8B7086">Inventory</strong>${inv}</div><div><strong style="font-size:.75rem;text-transform:uppercase;letter-spacing:1px;color:#8B7086">Patient Queue</strong>${pat}</div>`;
  
  // Actions
  let acts='';
  obs.available_actions.forEach(a=>{
    const cls=a.action_type==='wait'?'wait-btn':'';
    acts+=`<button class="action-btn ${cls}" data-idx="${a.index}"><span class="action-idx">${a.index}</span>${a.description}</button>`;
  });
  actionsPanel.innerHTML=acts;
  
  // Bind action clicks
  actionsPanel.querySelectorAll('.action-btn').forEach(btn=>{
    btn.addEventListener('click',()=>doStep(parseInt(btn.dataset.idx)));
  });
  
  btnOracle.disabled=false;
}

function renderRewards(rc){
  totalReward.textContent=(rc.total>=0?'+':'')+rc.total.toFixed(1);
  totalReward.style.background=rc.total>=0?'#E8F5E9':'#FDE8E8';
  totalReward.style.color=rc.total>=0?'#2E7D32':'#C41E3A';
  
  const keys=[['patient','Patient',5],['waste','Waste',10],['compat','Compat',3],['equity','Equity',4],['inaction','Inaction',4]];
  let html='';
  keys.forEach(([k,label,max])=>{
    const v=rc[k]||0;
    const pct=Math.min(Math.abs(v)/max*100,100);
    const cls=v>0?'positive':v<0?'negative':'neutral';
    const vcls=v>0?'pos':v<0?'neg':'';
    html+=`<div class="reward-row"><span class="reward-label">${label}</span><div class="reward-bar-wrap"><div class="reward-bar-fill ${cls}" style="width:${pct}%"></div></div><span class="reward-val ${vcls}">${v>=0?'+':''}${v.toFixed(1)}</span></div>`;
  });
  rewardsPanel.innerHTML=html;
}

async function doReset(){
  btnOracle.disabled=true;
  log('Resetting episode...');
  try{
    const r=await fetch(BASE+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskSelect.value})});
    const d=await r.json();
    renderState(d.observation);
    rewardsPanel.innerHTML='<div class="empty-state"><p>Rewards shown after each step</p></div>';
    totalReward.textContent='—';
    totalReward.style.background='';totalReward.style.color='';
    log(`Episode started: <strong>${taskSelect.value}</strong>`);
  }catch(e){log('Error: '+e.message);}
}

async function doStep(idx){
  log(`Action ${idx} selected`);
  try{
    const r=await fetch(BASE+'/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:{action_index:idx}})});
    const d=await r.json();
    renderState(d.observation);
    renderRewards(d.reward_components);
    if(d.info&&d.info.events) d.info.events.forEach(ev=>log(ev));
    if(d.done){log('<strong style="color:#C41E3A">Episode finished!</strong>');btnOracle.disabled=true;}
  }catch(e){log('Error: '+e.message);}
}

function oracleStep(){
  if(!currentObs) return;
  const allocs=currentObs.available_actions.filter(a=>a.action_type==='allocate');
  if(allocs.length){
    // Pick highest urgency patient action
    let best=allocs[0];
    for(const a of allocs){
      for(const p of currentObs.patient_queue||[]){
        if(a.description.includes(p.patient_id)&&p.urgency>(currentObs.patient_queue.find(pp=>best.description.includes(pp.patient_id))||{urgency:0}).urgency){
          best=a;
        }
      }
    }
    doStep(best.index);
  } else doStep(1);
}

btnReset.addEventListener('click',doReset);
btnOracle.addEventListener('click',oracleStep);

})();
