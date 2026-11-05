"""Aggregate a clippy JSON stream by lint. usage: agg.py <name>.clippy.json"""
import json,sys,collections
import os; repo=os.path.basename(sys.argv[1]).removesuffix('.clippy.json')
c=collections.Counter(); per=collections.defaultdict(collections.Counter); seen=set()
for line in open(sys.argv[1]):
    try:d=json.loads(line)
    except:continue
    if d.get('reason')!='compiler-message':continue
    m=d['message']; code=(m.get('code') or {}).get('code')
    if not code or not code.startswith('clippy::'):continue
    sp=[s for s in m['spans'] if s['is_primary']]
    if not sp:continue
    s=sp[0]; key=(code,s['file_name'],s['line_start'])
    if key in seen:continue
    seen.add(key)
    if '/registry/' in s['file_name'] or s['file_name'].startswith('/'):continue
    c[code]+=1; per[code][s['file_name']]+=1
print(repo,'total',sum(c.values()))
for code,n in c.most_common(): 
    top=', '.join(f'{f}({k})' for f,k in per[code].most_common(3))
    print(f'  {n:5d} {code:55s} {top}')
