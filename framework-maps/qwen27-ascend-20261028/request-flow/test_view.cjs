// 通过最小 DOM 替身执行实际页面逻辑；不是浏览器渲染/交互验收。
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element {
  setAttribute(key,value){this[key]=String(value)}
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
ids.get('report-data').textContent=html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const ctx=vm.createContext({Node:Element,document:{getElementById:id=>{assert(ids.has(id),id);return ids.get(id)},createElement:tag=>new Element(tag),createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e},querySelectorAll:()=>[]}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],ctx);
assert.equal(ids.get('request-readout').children.length,2);assert.equal(ids.get('batch-readout').children.length,2);
vm.runInContext(`let tested=0;for(const len of lengths)for(const concurrency of [1,5,10,16])for(const r of [1,2,3]){n.value=len;c.value=concurrency;rep.value=r;orders();for(let i=0;i<concurrency;i++){ord.value=i;drawRequest();if(at('request-readout').textContent.includes('该请求没有记录'))throw Error('missing request');tested++}}if(tested!==672)throw Error('pair count '+tested)`,ctx);
vm.runInContext(`let count=0;for(const len of [...new Set(D.profiles.map(x=>x.n))])for(const concurrency of [...new Set(D.profiles.map(x=>x.c))]){pn.value=len;pc.value=concurrency;refreshProfiles();for(const fw of ['vllm','sglang']){for(const op of picks[fw].children){picks[fw].value=op.value;const p=D.profiles[+picks[fw].value];if(p.framework!==fw||p.n!==len||p.c!==concurrency)throw Error('wrong scope');renderOperators();count++}}}if(count!==500)throw Error('window count '+count);pn.value=-1;pc.value=-1;refreshProfiles();if(!at('operator-explanation').textContent.includes('没有匹配窗口'))throw Error('empty selection fell back to profile zero')`,ctx);
for(const article of ids.get('batch-readout').children){const label=article.children[1],select=label.children.find(x=>x.tag==='select');for(const option of select.children){select.value=option.value;select.onchange();assert(!article.textContent.includes('NaN'));for(const token of ['actual_tokens','graph_key','ShapeKey(',':null','[object Object]'])assert(!article.textContent.includes(token))}}
console.log(JSON.stringify({status:'passed',request_pairs:672,profile_windows:500,empty_selection:'passed',diagnostic_request_selectors:'passed',raw_fields_visible:false,environment:'minimal DOM test double; browser rendering separately checked'}));
