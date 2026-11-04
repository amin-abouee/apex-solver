"""Duplicate-window detector. usage: dup.py <repo-root> [window=8]"""
import os,re,sys,collections,hashlib
root=sys.argv[1]; K=int(sys.argv[2]) if len(sys.argv)>2 else 8
repo=os.path.basename(os.path.abspath(root))
files=[]
for dp,dn,fn in os.walk(root):
    dn[:]=[d for d in dn if d not in('target','.git','output','data','book','node_modules','.opencode','tests','benches','examples')]
    for f in fn:
        if f.endswith('.rs') and f!='tests.rs' and not f.endswith('_tests.rs'): files.append(os.path.join(dp,f))
def norm(l):
    l=l.strip()
    if not l or l.startswith('//') or l.startswith('#[') or l in('{','}','});','}','};',')','),'): return None
    l=re.sub(r'\b\d+(\.\d+)?(e-?\d+)?\b','N',l)
    l=re.sub(r'"[^"]*"','S',l)
    return l
seq={}
for p in files:
    lines=open(p,encoding='utf-8',errors='replace').read().split('\n')
    # drop test module
    for i,l in enumerate(lines):
        if l.strip().startswith('#[cfg(test)]'): lines=lines[:i];break
    seq[p]=[(i+1,norm(l)) for i,l in enumerate(lines) if norm(l) is not None]
def table_row(t): return bool(re.fullmatch(r'[\[\](),.\sNS_a-zA-Z:-]*',t)) and t.count('N')>=2 and 'fn ' not in t and '=' not in t
sh=collections.defaultdict(list)
for p,s in seq.items():
    for i in range(len(s)-K+1):
        if sum(1 for x in s[i:i+K] if table_row(x[1]))>=K-1: continue
        h=hashlib.md5('\n'.join(x[1] for x in s[i:i+K]).encode()).hexdigest()
        sh[h].append((p,s[i][0],i))
# group duplicates into maximal regions per file pair
cover=collections.defaultdict(set); pairs=collections.Counter()
for h,locs in sh.items():
    if len(locs)<2: continue
    for p,l,i in locs:
        for j in range(K): cover[p].add(i+j)
    fs=sorted({p for p,_,_ in locs})
    for a in range(len(fs)):
        for b in range(a+1,len(fs)): pairs[(fs[a],fs[b])]+=1
    if len({p for p,_,_ in locs})==1:
        pairs[(locs[0][0],locs[0][0])]+=1
tot=sum(len(s) for s in seq.values()); dup=sum(len(c) for c in cover.values())
print(f'{repo}: significant lines {tot}, in duplicated {K}-line windows {dup} ({100*dup/tot:.1f}%)')
print(' top file pairs by shared windows:')
for (a,b),n in pairs.most_common(14):
    ra,rb=os.path.relpath(a,root),os.path.relpath(b,root)
    print(f'  {n:4d}  {ra}  <->  {rb}')
