"""本地离线报告的渲染检查；阻断网络，不访问公共网站或服务。"""
import json
import hashlib
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

U=Path(__file__).resolve().parent
OUT=Path(tempfile.mkdtemp(prefix='qwen27-report-qa-'))
FILE=U/'report.html'


def inspect(page):
    return page.evaluate('''() => {
      const text=document.body.innerText;
      const leaked=['actual_tokens','scheduled_tokens','padded_tokens','graph_key','ShapeKey(',':null','[object Object]','NaN'].filter(x=>text.includes(x));
      const overflow=[...document.querySelectorAll('p,h1,h2,h3,td,th,summary')].filter(e=>{
        const r=e.getBoundingClientRect();if(!r.width||!r.height)return false;
        if(e.closest('.table-scroll'))return false;
        return r.right>innerWidth+2 || r.left < -2;
      }).map(e=>e.textContent.slice(0,70));
      const clippedCells=[...document.querySelectorAll('td,th')].filter(cell=>{const box=cell.getBoundingClientRect();if(!box.width||!box.height)return false;const range=document.createRange();range.selectNodeContents(cell);return [...range.getClientRects()].some(r=>r.right>box.right+2||r.left<box.left-2)}).map(e=>e.textContent.slice(0,60));
      return {clippedCells,viewport:innerWidth,pageOverflow:document.documentElement.scrollWidth>innerWidth+2,leaked,overflow,stages:document.querySelectorAll('.stage').length,sections:document.querySelectorAll('main>section').length};
    }''')


def main():
    results=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for width in [1440,768,390]:
            page=browser.new_page(viewport={'width':width,'height':980},device_scale_factor=1)
            page.route('**/*',lambda route: route.continue_() if route.request.url.startswith('file:') else route.abort())
            page.on('pageerror',lambda error: errors.append(str(error)))
            page.goto(FILE.as_uri(),wait_until='load')
            assert page.locator('#request-readout article').count()==2
            initial=inspect(page);assert not initial['clippedCells'] and not initial['pageOverflow'] and not initial['leaked'] and not initial['overflow'],initial
            assert initial['stages']==13 and initial['sections']==11
            page.screenshot(path=str(OUT/f'preview-{width}.png'))
            # The screenshot's problematic C16 decode graph, now expressed as text.
            for details in page.locator('#batch-readout details').all():details.locator('summary').click()
            page.locator('#batch').scroll_into_view_if_needed()
            expanded=inspect(page);assert not expanded['clippedCells'] and not expanded['pageOverflow'] and not expanded['leaked'] and not expanded['overflow'],expanded
            assert '回放图的批大小为16个请求' in page.locator('#batch').inner_text()
            page.screenshot(path=str(OUT/f'batch-{width}.png'))
            if width==1440:
                # Every formal request pair and every profile record through actual page functions.
                counts=page.evaluate('''() => {
                  let pairs=0,profiles=0;
                  for(const len of lengths)for(const concurrency of [1,5,10,16])for(const repetition of [1,2,3]){
                    n.value=len;c.value=concurrency;rep.value=repetition;orders();
                    for(let i=0;i<concurrency;i++){ord.value=i;drawRequest();if(at('request-readout').innerText.includes('没有记录'))throw Error('missing request');pairs++;}
                  }
                  for(const len of [...new Set(D.profiles.map(x=>x.n))])for(const concurrency of [...new Set(D.profiles.map(x=>x.c))]){
                    pn.value=len;pc.value=concurrency;refreshProfiles();
                    for(const fw of ['vllm','sglang'])for(const option of picks[fw].options){picks[fw].value=option.value;renderOperators();profiles++;}
                  }
                  return {pairs,profiles};
                }''')
                assert counts=={'pairs':672,'profiles':500},counts
                page.get_by_label('输入token',exact=True).first.select_option('131072')
                page.get_by_label('并发请求',exact=True).select_option('16')
                page.get_by_label('重复轮次',exact=True).select_option('3')
                page.get_by_label('提交序号',exact=True).select_option('15')
                page.locator('#request').scroll_into_view_if_needed();page.screenshot(path=str(OUT/'requests-long.png'))
                page.get_by_label('输入token',exact=True).last.select_option('131072')
                page.get_by_label('并发',exact=True).select_option('16')
                page.locator('#device').scroll_into_view_if_needed();page.screenshot(path=str(OUT/'operators.png'))
                results.append(counts)
            page.locator('#components details').evaluate_all('(nodes)=>nodes.forEach(n=>n.open=true)')
            final=inspect(page);assert not final['clippedCells'] and not final['pageOverflow'] and not final['leaked'] and not final['overflow'],final
            results.append({'initial':initial,'expanded':expanded,'final':final})
            page.close()
        browser.close()
    assert not errors,errors
    receipt={'status':'passed','html_sha256':hashlib.sha256(FILE.read_bytes()).hexdigest(),'environment':'local Chromium / file URL / network blocked','viewports':[1440,768,390],'page_errors':errors,'results':results,'raw_json_visible':False,'new_performance_test':False}
    (OUT/'browser-qa.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
