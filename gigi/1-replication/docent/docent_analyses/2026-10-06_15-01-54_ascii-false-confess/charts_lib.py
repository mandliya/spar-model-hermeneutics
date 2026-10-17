"""Shared helpers for the Docent build scripts: Wilson intervals, CSV export, charts.

Charts are self-contained HTML (light/dark tokens, hover tooltips) saved to
gigi/1-replication/docent/charts/, with a 2x PNG rendered by headless Chrome for GitHub."""
import csv
import json
import math
import pathlib
import re
import subprocess

DOCENT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS, CHARTS = DOCENT_DIR / "results", DOCENT_DIR / "charts"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
strip = lambda s: re.sub(r"\s*\[T\d+B\d+:<RANGE>.*?</RANGE>\]", "", s or "", flags=re.S).strip()


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)



def save_csv(name, rows, cols):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: strip(v) if isinstance(v, str) else v for k, v in r.items()})



# ---- charts ----
STYLE = """<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--none:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--none:#52514e;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9}}
body{margin:0;padding:16px;background:var(--surface);color:var(--ink);font:13px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-bottom:16px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;flex:none;display:inline-block}
.row{margin:0 0 16px}.lab{display:flex;justify-content:space-between;gap:12px;margin-bottom:6px;font-weight:600}
.lab small,.h small{color:var(--muted);font-weight:400}
.bar{display:flex;gap:2px;height:24px}
.seg{flex-basis:0;min-width:3px;display:flex;align-items:center;justify-content:center;color:#0b0b0b;font-size:12px;font-weight:600;white-space:nowrap}
.seg:last-child{border-radius:0 4px 4px 0}
.g{display:grid;column-gap:14px;align-items:center}
.h{font-weight:600;padding-bottom:8px;border-bottom:1px solid var(--grid)}
.l{color:var(--ink2);padding:7px 0;display:flex;gap:8px;align-items:center}
.t{display:flex;align-items:center;gap:6px;height:30px;border-left:1px solid var(--grid);position:relative}
.b{height:16px;border-radius:0 4px 4px 0}
.ci{position:absolute;height:1px;background:var(--ink2);top:50%}
.v{font-variant-numeric:tabular-nums;color:var(--ink);font-size:12px;white-space:nowrap}.v.z{color:var(--muted)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:6px 8px;border-radius:6px;font-size:12px;opacity:0;max-width:280px}
</style>"""
TIP_JS = """const tip=document.getElementById("tip");
function hover(el,html){el.onmousemove=e=>{tip.innerHTML=html;tip.style.opacity=1;tip.style.left=Math.min(e.clientX+12,innerWidth-290)+"px";tip.style.top=(e.clientY+12)+"px"};el.onmouseleave=()=>tip.style.opacity=0}"""

STACKED_JS = """const legend=document.getElementById("legend");
legend.innerHTML=D.segs.map(s=>`<span><i class="sw" style="background:var(${s[1]})"></i>${s[0]}</span>`).join("");
D.rows.forEach(([label,sub,v])=>{const n=v.reduce((a,b)=>a+b,0);
 const row=document.createElement("div");row.className="row";row.innerHTML=`<div class="lab">${label}<small>${sub} · n = ${n}</small></div>`;
 const bar=document.createElement("div");bar.className="bar";
 v.forEach((k,i)=>{if(!k)return;const s=document.createElement("div");s.className="seg";s.style.flexGrow=k;s.style.background=`var(${D.segs[i][1]})`;
  const t=document.createElement("span");t.textContent=`${Math.round(100*k/n)}% (${k})`;s.appendChild(t);
  hover(s,`<b>${D.segs[i][0]}</b><br>${label}: ${k} of ${n} (${(100*k/n).toFixed(1)}%)`);bar.appendChild(s)});
 row.appendChild(bar);document.getElementById("rows").appendChild(row)});
requestAnimationFrame(()=>document.querySelectorAll(".seg span").forEach(t=>{const s=t.parentElement;if(t.offsetWidth+12>s.clientWidth)t.style.visibility="hidden";const c=getComputedStyle(s).backgroundColor.match(/\d+/g).map(Number);s.style.color=(0.2126*c[0]+0.7152*c[1]+0.0722*c[2])/255<0.45?"#fff":"#0b0b0b"}));"""

TWOCOL_JS = """const g=document.getElementById("g");g.style.gridTemplateColumns="minmax(130px,230px) 1fr 1fr";
const max=Math.max(...D.groups.flatMap(x=>x[2]));
if(D.legend)document.getElementById("legend").innerHTML=D.legend.map(s=>`<span><i class="sw" style="background:var(${s[1]})"></i>${s[0]}</span>`).join("");
g.innerHTML=`<div class="h">${D.rowhead}</div>`+D.cols.map(c=>`<div class="h">${c[0]} <small>n = ${c[1]}</small></div>`).join("");
D.groups.forEach(([name,color,v])=>{g.insertAdjacentHTML("beforeend",`<div class="l"><i class="sw" style="background:var(${color})"></i>${name}</div>`);
 v.forEach((k,j)=>{const t=document.createElement("div");t.className="t";
  t.innerHTML=k?`<div class="b" style="width:calc(${k/max} * (100% - 34px));background:var(${color})"></div><span class="v">${k}</span>`:`<span class="v z">&nbsp;0</span>`;
  hover(t,`<b>${name}</b><br>${D.cols[j][0]}: ${k} of ${D.cols[j][1]} (${(100*k/D.cols[j][1]).toFixed(1)}%)`);g.appendChild(t)})});"""

CI_JS = """const g=document.getElementById("g");g.style.gridTemplateColumns="minmax(130px,230px) 1fr";
g.innerHTML=`<div class="h">${D.rowhead}</div><div class="h">${D.colhead}</div>`;
D.rows.forEach(([name,k,n,lo,hi])=>{g.insertAdjacentHTML("beforeend",`<div class="l">${name} <small style="color:var(--muted)">n = ${n}</small></div>`);
 const p=100*k/n,t=document.createElement("div");t.className="t";const W=f=>`calc(${f/100} * (100% - 130px))`;
 t.innerHTML=`<div class="b" style="width:${W(p)};background:var(--s1);min-width:${k?3:0}px"></div>`
  +`<div class="ci" style="left:${W(lo)};width:${W(hi-lo)}"></div>`
  +`<div class="ci" style="left:${W(lo)};width:1px;height:9px;margin-top:-4px"></div><div class="ci" style="left:${W(hi)};width:1px;height:9px;margin-top:-4px"></div>`
  +`<span class="v" style="position:absolute;left:calc(${W(hi)} + 8px)">${Math.round(p)}% (${k} of ${n})</span>`;
 hover(t,`<b>${name}</b><br>${k} of ${n} = ${p.toFixed(1)}%<br>95% CI ${lo.toFixed(1)}–${hi.toFixed(1)}%`);g.appendChild(t)});"""


def chart(name, kind, data, height):
    body = {"stacked": '<div class="legend" id="legend"></div><div id="rows"></div>',
            "twocol": '<div class="legend" id="legend"></div><div class="g" id="g"></div>',
            "ci": '<div class="g" id="g"></div>'}[kind]
    js = {"stacked": STACKED_JS, "twocol": TWOCOL_JS, "ci": CI_JS}[kind]
    html = f'{STYLE}\n{body}<div id="tip"></div>\n<script>\nconst D={json.dumps(data, ensure_ascii=False)};\n{TIP_JS}\n{js}\n</script>'
    CHARTS.mkdir(exist_ok=True)
    (CHARTS / f"{name}.html").write_text(html)
    page = CHARTS / f"{name}.page.html"
    page.write_text("<!doctype html><meta charset=utf-8>" + html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=760,{height}", f"--screenshot={CHARTS / (name + '.png')}", page.as_uri()],
                   check=True, capture_output=True)
    page.unlink()
    return html


