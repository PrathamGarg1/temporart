#!/usr/bin/env python3
"""Dataset viewer. Standard library only.

  python3 viewer/app.py                       # http://127.0.0.1:8765
  DATA_DIR=/path/to/all python3 app.py --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "dataset" / "all"))
ROWS: list[dict] = []
FIELDS = ["split", "length", "script", "label", "subtype"]


def load() -> None:
    for split in ("train", "val", "test"):
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_i"] = len(ROWS)
                r["_blob"] = " ".join(r["turns"]).lower()
                ROWS.append(r)
    print(f"loaded {len(ROWS):,} rows", flush=True)


def public(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")} | {"id": r["_i"]}


def stats() -> dict:
    out = {k: dict(Counter(str(r[k]) for r in ROWS)) for k in FIELDS}
    out["total"] = len(ROWS)
    out["cells"] = [
        {"length": a, "script": b, "label": c, "n": n}
        for (a, b, c), n in sorted(Counter((r["length"], r["script"], r["label"]) for r in ROWS).items())
    ]
    out["subtypes"] = sorted({(r["label"], r["subtype"]) for r in ROWS})
    return out


def query(q: dict) -> dict:
    def g(k):
        return (q.get(k, [""])[0]) or None

    flt = {k: g(k) for k in FIELDS}
    text = (g("q") or "").lower()
    last = (g("last") or "").strip().lower()
    page = max(int(g("page") or 1), 1)
    size = min(int(g("size") or 50), 100)
    hits = [
        r
        for r in ROWS
        if all(v is None or str(r[k]) == v for k, v in flt.items())
        and (not text or text in r["_blob"])
        and (not last or r["last_turn"].strip().lower() == last)
    ]
    if g("shuffle") == "1":
        random.Random(int(g("seed") or 1)).shuffle(hits)
    return {
        "total": len(hits),
        "page": page,
        "size": size,
        "rows": [public(r) for r in hits[(page - 1) * size : page * size]],
    }


def sample(q: dict) -> dict:
    def g(k):
        return (q.get(k, [""])[0]) or None

    flt = {k: g(k) for k in ("length", "script", "label", "split")}
    pool = [r for r in ROWS if all(v is None or str(r[k]) == v for k, v in flt.items())]
    if not pool:
        return {"total": 0, "rows": []}
    n = min(int(g("n") or 1), 10)
    rng = random.Random(g("seed") or None)
    return {"total": len(pool), "rows": [public(r) for r in rng.sample(pool, min(n, len(pool)))]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self.send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/stats":
            self.send(json.dumps(stats()).encode(), "application/json")
        elif u.path == "/api/rows":
            self.send(json.dumps(query(q), ensure_ascii=False).encode("utf-8"), "application/json")
        elif u.path == "/api/sample":
            self.send(json.dumps(sample(q), ensure_ascii=False).encode("utf-8"), "application/json")
        elif u.path == "/healthz":
            self.send(b"ok", "text/plain")
        else:
            self.send_error(404)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>surakshanet-threads</title>
<style>
:root{--ink:#1f2328;--mut:#656d76;--line:#d8dee4;--bg:#fff;--sub:#f6f8fa;--sel:#ddf4ff;--acc:#0969da}
*{box-sizing:border-box}
body{margin:0;font:13px/1.45 -apple-system,"Segoe UI",Helvetica,Arial,"Noto Sans Devanagari",sans-serif;color:var(--ink);background:var(--bg)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
header{padding:14px 24px 0;border-bottom:1px solid var(--line);background:var(--sub)}
header h1{margin:0 0 10px;font-size:18px;font-weight:600}
header h1 span{color:var(--mut);font-weight:400}
.tabs{display:flex;gap:2px}
.tabs button{border:0;background:none;padding:8px 14px;font:inherit;font-size:14px;color:var(--mut);cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:var(--ink);border-color:#fd8c73;font-weight:600}
.layout{display:flex;align-items:flex-start}
aside{width:230px;flex:none;padding:16px 18px;border-right:1px solid var(--line);min-height:calc(100vh - 82px);background:var(--sub)}
aside label{display:block;font-weight:600;font-size:12px;margin:12px 0 4px}
aside select,aside input[type=text]{width:100%;padding:5px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff}
aside .row{display:flex;gap:6px;margin-top:14px}
.btn{padding:5px 12px;border:1px solid var(--line);background:#f6f8fa;border-radius:6px;font:inherit;cursor:pointer}
.btn:hover{background:#eef1f4}.btn:disabled{opacity:.5;cursor:default}
.chip{display:inline-flex;align-items:center;gap:6px;margin-top:10px;padding:3px 8px;border:1px solid var(--line);border-radius:12px;background:#fff;font-size:12px}
.chip a{cursor:pointer;color:var(--mut)}
section.main{flex:1;min-width:0;padding:16px 24px}
.sum{color:var(--mut);margin:0 0 10px}
.tw{border:1px solid var(--line);border-radius:6px;overflow:auto;max-height:calc(100vh - 220px)}
table{border-collapse:collapse;width:100%}
th{position:sticky;top:0;background:var(--sub);text-align:left;font-weight:600;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th small{display:block;font-weight:400;color:var(--mut);font-family:ui-monospace,Menlo,monospace;font-size:11px}
td{padding:6px 10px;border-bottom:1px solid #eaeef2;vertical-align:top;max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
tr.r{cursor:pointer}tr.r:hover td{background:#f6f8fa}tr.r.on td{background:var(--sel)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.pager{display:flex;gap:10px;align-items:center;margin-top:12px;color:var(--mut)}
#detail{position:fixed;top:0;right:0;bottom:0;width:min(520px,100%);background:#fff;border-left:1px solid var(--line);box-shadow:-6px 0 18px #0001;padding:18px 22px;overflow:auto;transform:translateX(105%);transition:transform .15s}
#detail.on{transform:none}
#detail h2{margin:0 0 12px;font-size:15px;display:flex;justify-content:space-between;align-items:center}
dl{display:grid;grid-template-columns:130px 1fr;gap:5px 12px;margin:0 0 16px}
dt{color:var(--mut)}dd{margin:0}
ol.t{margin:0;padding:0;list-style:none;border:1px solid var(--line);border-radius:6px}
ol.t li{padding:8px 12px;border-bottom:1px solid #eaeef2;display:grid;grid-template-columns:22px 70px 1fr;gap:6px}
ol.t li:last-child{border-bottom:0;background:var(--sel)}
ol.t .n{color:var(--mut)}ol.t .s{color:var(--mut);overflow:hidden;text-overflow:ellipsis}
.samp{padding:18px 24px;max-width:860px}
.samp .tools{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-bottom:14px}
.samp .tools label{display:flex;flex-direction:column;font-weight:600;font-size:12px;gap:4px}
.samp .tools select{padding:5px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff;min-width:150px}
.btn.pri{background:#1f2328;border-color:#1f2328;color:#fff}.btn.pri:hover{background:#32383f}
.conv{border:1px solid var(--line);border-radius:8px;padding:16px 18px;background:#fff}
.conv .meta{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--mut);font-size:12px;padding-bottom:12px;margin-bottom:12px;border-bottom:1px solid #eaeef2}
.conv .meta b{color:var(--ink);font-weight:600}
.m{display:flex;flex-direction:column;margin:10px 0;max-width:74%}
.m.r{margin-left:auto;align-items:flex-end}
.m .who{font-size:11px;color:var(--mut);margin:0 2px 3px}
.m .tx{padding:8px 12px;border-radius:8px;background:#f1f3f5;font-size:14px;line-height:1.5}
.m.r .tx{background:#e7f0fb}
.m.last .tx{border:1.5px solid #1f2328}
.m .tag{font-size:10.5px;color:var(--mut);margin:3px 2px 0;font-family:ui-monospace,Menlo,monospace}
.stats{padding:20px 24px;max-width:1100px}
.stats h3{margin:22px 0 8px;font-size:14px}
.st{display:flex;gap:30px;flex-wrap:wrap}
.stats table{width:auto;border:1px solid var(--line)}
.stats td,.stats th{position:static;max-width:none}
.stats td.n{text-align:right;font-variant-numeric:tabular-nums}
.sbar{display:inline-block;height:8px;background:#54aeff;vertical-align:middle;border-radius:2px}
</style></head><body>
<header><h1>surakshanet <span>/</span> threads</h1>
<div class="tabs"><button data-t="samp" class="on">Sample threads</button><button data-t="viewer">Dataset Viewer</button><button data-t="stats">Statistics</button></div></header>
<div id="samp" class="samp"></div>
<div id="viewer" class="layout" hidden>
 <aside id="side"></aside>
 <section class="main"><p class="sum" id="sum"></p><div class="tw"><table id="tbl"></table></div><div class="pager" id="pager"></div></section>
</div>
<div id="stats" class="stats" hidden></div>
<div id="detail"></div>
<script>
const $=s=>document.querySelector(s),esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const j=u=>fetch(u).then(r=>r.json());
const NICE={hindi_devanagari:"hindi_devanagari",english:"english",hinglish:"hinglish"};
const st={page:1,seed:1,last:"",rows:[]};let S=null;
function sel(id,label,vals){return `<label for="${id}">${label}</label><select id="${id}"><option value="">all</option>${vals.map(v=>`<option>${v}</option>`).join("")}</select>`}
const SS={n:0};
function sampleUI(){
 const ks=k=>Object.keys(S[k]).sort((a,b)=>isNaN(a)?a.localeCompare(b):a-b);
 const o=(id,l,v,first)=>`<label>${l}<select id="${id}">${first?"<option value=''>any</option>":""}${v.map(x=>`<option>${x}</option>`).join("")}</select></label>`;
 $("#samp").innerHTML=`<div class="tools">${o("s_len","length",ks("length"))}${o("s_script","script",ks("script"))}${o("s_label","label",ks("label"))}${o("s_split","split",["train","val","test"],1)}<button class="btn pri" id="s_rand">Random thread</button></div><div id="s_out"></div>`;
 $("#s_len").value="3";$("#s_script").value="hinglish";$("#s_label").value="hostile_abuse";
 ["s_len","s_script","s_label","s_split"].forEach(i=>$("#"+i).onchange=randomThread);
 $("#s_rand").onclick=randomThread;randomThread();
}
let REQ=0;
async function randomThread(){
 const my=++REQ;
 const p=new URLSearchParams({n:1,seed:Math.random().toString(36).slice(2)});
 [["s_len","length"],["s_script","script"],["s_label","label"],["s_split","split"]].forEach(([i,k])=>{const v=$("#"+i).value;if(v)p.set(k,v)});
 const d=await j("/api/sample?"+p);if(my!==REQ)return;const r=d.rows[0];
 if(!r){$("#s_out").innerHTML="<p class='sum'>No rows for this selection.</p>";return}
 const left=r.speakers[0];
 $("#s_out").innerHTML=`<p class="sum">${d.total.toLocaleString()} threads in this selection</p><div class="conv"><div class="meta"><span>label <b>${r.label}</b></span><span>subtype <b>${r.subtype}</b></span><span>length <b>${r.length}</b></span><span>script <b>${r.script}</b></span><span>split <b>${r.split}</b></span><span class="mono">${esc(r.thread_id)}</span></div>`+
  r.turns.map((t,k)=>{const last=k===r.turns.length-1;return `<div class="m ${r.speakers[k]===left?"":"r"} ${last?"last":""}"><span class="who">${esc(r.speakers[k])}</span><div class="tx">${esc(t)}</div>${last?'<span class="tag">last message · classified as '+r.label+'</span>':""}</div>`}).join("")+"</div>";
}
async function init(){
 S=await j("/api/stats");sampleUI();
 const ks=k=>Object.keys(S[k]).sort((a,b)=>isNaN(a)?a.localeCompare(b):a-b);
 $("#side").innerHTML=sel("split","split",["train","val","test"])+sel("length","length",ks("length"))+sel("script","script",ks("script"))+sel("label","label",ks("label"))+sel("subtype","subtype",ks("subtype"))
  +`<label for="q">search</label><input type="text" id="q" placeholder="text in any turn">
   <div id="chip"></div>
   <div class="row"><button class="btn" id="shuf">Shuffle</button><button class="btn" id="reset">Reset</button></div>`;
 ["split","length","script","label","subtype"].forEach(i=>$("#"+i).onchange=()=>{st.page=1;load()});
 $("#q").onkeydown=e=>{if(e.key==="Enter"){st.page=1;load()}};
 $("#shuf").onclick=()=>{st.seed++;st.page=1;st.shuffle=1;load()};
 $("#reset").onclick=()=>{["split","length","script","label","subtype","q"].forEach(i=>$("#"+i).value="");st.last="";st.shuffle=0;st.page=1;load()};
 load();stats();
}
const COLS=[["split","string"],["length","int8"],["script","string"],["label","string"],["subtype","string"],["context","string"],["last_turn","string"]];
async function load(){
 const p=new URLSearchParams();
 ["split","length","script","label","subtype"].forEach(k=>{const v=$("#"+k).value;if(v)p.set(k,v)});
 if($("#q").value)p.set("q",$("#q").value);if(st.last)p.set("last",st.last);
 if(st.shuffle){p.set("shuffle","1");p.set("seed",st.seed)}
 p.set("page",st.page);p.set("size",50);
 const d=await j("/api/rows?"+p);st.rows=d.rows;
 $("#chip").innerHTML=st.last?`<span class="chip">last_turn = "${esc(st.last.slice(0,24))}" <a onclick="st.last='';st.page=1;load()">✕</a></span>`:"";
 $("#tbl").innerHTML="<thead><tr>"+COLS.map(([c,t])=>`<th>${c}<small>${t}</small></th>`).join("")+"</tr></thead><tbody>"+
  d.rows.map((r,i)=>`<tr class="r" onclick="open_(${i})"><td>${r.split}</td><td class="num">${r.length}</td><td>${r.script}</td><td>${r.label}</td><td>${r.subtype}</td><td>${esc(r.turns.slice(0,-1).join(" / "))}</td><td>${esc(r.last_turn)}</td></tr>`).join("")+"</tbody>";
 const pages=Math.max(1,Math.ceil(d.total/d.size));
 $("#sum").textContent=`${d.total.toLocaleString()} rows`;
 $("#pager").innerHTML=`<button class="btn" ${d.page<=1?"disabled":""} onclick="go(${d.page-1})">‹ Previous</button><span>Page ${d.page.toLocaleString()} of ${pages.toLocaleString()}</span><button class="btn" ${d.page>=pages?"disabled":""} onclick="go(${d.page+1})">Next ›</button>`;
 $(".tw").scrollTop=0;
}
function go(n){st.page=n;load()}
function open_(i){
 const r=st.rows[i];document.querySelectorAll("tr.r").forEach((e,k)=>e.classList.toggle("on",k===i));
 const f=(k,v)=>`<dt>${k}</dt><dd class="mono">${esc(v===""||v==null?"—":v)}</dd>`;
 $("#detail").innerHTML=`<h2>Row <button class="btn" onclick="closeD()">Close</button></h2>
 <dl>${f("thread_id",r.thread_id)}${f("split",r.split)}${f("length",r.length)}${f("n_context",r.n_context)}${f("script",r.script)}${f("label",r.label+" ("+r.label_id+")")}${f("subtype",r.subtype)}${f("topic",r.topic)}${f("audience",r.audience)}${f("target_gender",r.target_gender)}${f("child_age",r.child_age||"")}${f("pair_id",r.pair_id)}${f("source",r.source)}</dl>
 <p style="font-weight:600;margin:0 0 6px">turns</p>
 <ol class="t">${r.turns.map((t,k)=>`<li><span class="n">${k+1}</span><span class="s">${esc(r.speakers[k])}</span><span>${esc(t)}</span></li>`).join("")}</ol>
 <p style="margin-top:14px"><button class="btn" onclick="sameLast(${i})">Rows with the same last_turn</button></p>`;
 $("#detail").classList.add("on");
}
function closeD(){$("#detail").classList.remove("on");document.querySelectorAll("tr.r.on").forEach(e=>e.classList.remove("on"))}
function sameLast(i){st.last=st.rows[i].last_turn;st.page=1;["split","length","script","label","subtype","q"].forEach(k=>$("#"+k).value="");closeD();load()}
function stats(){
 const t=(title,obj)=>{const e=Object.entries(obj).sort((a,b)=>isNaN(a[0])?a[0].localeCompare(b[0]):a[0]-b[0]);const m=Math.max(...e.map(x=>x[1]));
  return `<div><h3>${title}</h3><table><tr><th>${title}</th><th>rows</th><th></th></tr>${e.map(([k,v])=>`<tr><td>${k}</td><td class="n">${v.toLocaleString()}</td><td><span class="sbar" style="width:${Math.round(140*v/m)}px"></span></td></tr>`).join("")}</table></div>`};
 const labels=Object.keys(S.label).sort();
 let rows="";for(const L of [3,5,7,10])for(const sc of Object.keys(S.script).sort()){
  rows+=`<tr><td>${L}</td><td>${sc}</td>`+labels.map(l=>`<td class="n">${S.cells.find(c=>c.length===L&&c.script===sc&&c.label===l).n.toLocaleString()}</td>`).join("")+"</tr>"}
 $("#stats").innerHTML=`<h3 style="margin-top:0">${S.total.toLocaleString()} rows</h3><div class="st">${t("split",S.split)}${t("length",S.length)}${t("script",S.script)}${t("label",S.label)}</div>
 <h3>Rows per length × script × label</h3><table><tr><th>length</th><th>script</th>${labels.map(l=>`<th>${l}</th>`).join("")}</tr>${rows}</table>
 <h3>Features</h3><table><tr><th>column</th><th>type</th></tr>${[["thread_id","string"],["split","string"],["length","int"],["n_context","int"],["script","string"],["label","string"],["label_id","int"],["subtype","string"],["topic","string"],["audience","string"],["target_gender","string"],["child_age","int"],["pair_id","string"],["speakers","list[string]"],["turns","list[string]"],["last_turn","string"],["source","string"]].map(x=>`<tr><td class="mono">${x[0]}</td><td class="mono">${x[1]}</td></tr>`).join("")}</table>`}
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tabs button").forEach(x=>x.classList.toggle("on",x===b));
 $("#samp").hidden=b.dataset.t!=="samp";$("#viewer").hidden=b.dataset.t!=="viewer";$("#stats").hidden=b.dataset.t!=="stats";closeD()});
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeD()});
init();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    load()
    print(f"serving http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
