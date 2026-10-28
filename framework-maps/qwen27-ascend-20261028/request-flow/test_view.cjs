// 通过最小 DOM 替身执行实际页面逻辑；不是浏览器渲染/交互验收。
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.style={};this.dataset={};this.className='';this._text='';this._value=undefined;this.classList={add:x=>{this.className+=' '+x},remove:x=>{this.className=this.className.split(' ').filter(y=>y!==x).join(' ')}}}
  append(...nodes){this.children.push(...nodes)}
  replaceChildren(...nodes){this.children=[];this._text='';this._value=undefined;this.append(...nodes)}
  set textContent(x){this._text=String(x);this.children=[]}
  get textContent(){return this._text+this.children.map(x=>x.textContent).join('')}
  set value(x){this._value=String(x)}
  get value(){return this.tag==='select'?(this._value??this.children[0]?.value??''):(this._value??'')}
}
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const ids=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element('div')]));
ids.get('flow-data').textContent=html.match(/<script id="flow-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const ctx=vm.createContext({Node:Element,document:{getElementById:id=>{assert(ids.has(id),id);return ids.get(id)},createElement:tag=>new Element(tag),createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e},querySelectorAll:()=>[]}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],ctx);
assert.equal(ids.get('vflow').children.length,13);assert.equal(ids.get('sflow').children.length,13);
assert.equal(ids.get('request-pair').children.length,2);assert.equal(ids.get('step-pairs').children.length,2);
vm.runInContext(`for(const x of D.components)component(x.id)`,ctx);
vm.runInContext(`let tested=0;for(const n of ns)for(const c of cs)for(const r of [1,2,3]){nsel.value=n;csel.value=c;rsel.value=r;requestOptions();for(let i=0;i<c;i++){isel.value=i;renderRequest();if(at('request-pair').textContent.includes('该请求未记录'))throw Error('missing request');tested++}}if(tested!==672)throw Error('pair count '+tested)`,ctx);
vm.runInContext(`let count=0;for(const n of [...new Set(D.profiles.map(x=>x.n))])for(const c of [...new Set(D.profiles.map(x=>x.c))]){pn.value=n;pc.value=c;profileOptions();for(const fw of ['vllm','sglang']){const sel=profileSelects[fw];for(const op of sel.children){sel.value=op.value;const p=D.profiles[+sel.value];if(p.framework!==fw||p.n!==n||p.c!==c)throw Error('wrong scope');renderOperators();count++}}}if(count!==500)throw Error('window count '+count);pn.value=-1;pc.value=-1;profileOptions();if(at('operator-table').textContent.includes('确切窗口'))throw Error('empty selection fell back to profile zero')`,ctx);
for(const article of ids.get('step-pairs').children){const label=article.children[1],select=label.children.find(x=>x.tag==='select');for(const option of select.children){select.value=option.value;select.onchange();assert(!article.textContent.includes('NaN'))}}
console.log(JSON.stringify({status:'passed',component_views:13,request_pairs:672,profile_windows:500,empty_selection:'passed',diagnostic_request_selectors:'passed',environment:'minimal DOM test double; not browser QA'}));
