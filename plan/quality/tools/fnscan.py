import re,sys,os,json
def strip(src):
    out=[];i=0;n=len(src);st=0
    while i<n:
        c=src[i]
        if src.startswith('//',i):
            j=src.find('\n',i); j=n if j<0 else j
            out.append(' '*(j-i)); i=j; continue
        if src.startswith('/*',i):
            d=1;j=i+2
            while j<n and d:
                if src.startswith('/*',j): d+=1;j+=2
                elif src.startswith('*/',j): d-=1;j+=2
                else: j+=1
            out.append(re.sub(r'[^\n]',' ',src[i:j])); i=j; continue
        if c=='r' and re.match(r'r#*"',src[i:i+10]):
            m=re.match(r'r(#*)"',src[i:]); h=m.group(1); end='"'+h
            j=src.find(end,i+len(m.group(0))); j=n if j<0 else j+len(end)
            out.append(re.sub(r'[^\n]',' ',src[i:j])); i=j; continue
        if c=='"':
            j=i+1
            while j<n and src[j]!='"':
                j+=2 if src[j]=='\\' else 1
            j+=1
            out.append(re.sub(r'[^\n]',' ',src[i:j])); i=j; continue
        if c=="'":
            m=re.match(r"'(\\.[^']*|[^\\'])'",src[i:i+12])
            if m:
                out.append(' '*len(m.group(0))); i+=len(m.group(0)); continue
        out.append(c); i+=1
    return ''.join(out)
fnre=re.compile(r'\bfn\s+([A-Za-z_][A-Za-z0-9_]*)')
def scan(path):
    raw=open(path,encoding='utf-8',errors='replace').read()
    s=strip(raw)
    res=[]
    for m in fnre.finditer(s):
        i=m.end(); depth_par=0
        # find body start '{' or ';' at paren depth 0
        j=i;ang=0
        while j<len(s):
            ch=s[j]
            if ch in '([': depth_par+=1
            elif ch in ')]': depth_par-=1
            elif ch=='{' and depth_par==0: break
            elif ch==';' and depth_par==0: j=-1;break
            j+=1
        if j<0 or j>=len(s): continue
        d=0;k=j;maxd=0;branches=0
        while k<len(s):
            if s[k]=='{':
                d+=1;maxd=max(maxd,d)
            elif s[k]=='}':
                d-=1
                if d==0: break
            k+=1
        body=s[j:k+1]
        line0=s.count('\n',0,m.start())+1
        line1=s.count('\n',0,k)+1
        branches=len(re.findall(r'\b(if|match|for|while|loop)\b|\?\s*;|&&|\|\|',body))
        # params count
        sig=s[m.end():j]
        p0=sig.find('(')
        depth=0;q=p0;
        for q in range(p0,len(sig)):
            if sig[q]=='(':depth+=1
            elif sig[q]==')':
                depth-=1
                if depth==0:break
        params=sig[p0+1:q]
        nparams=0
        if params.strip():
            dd=0;cnt=1
            for ch in params:
                if ch in '<([': dd+=1
                elif ch in '>)]': dd-=1
                elif ch==',' and dd==0: cnt+=1
            nparams=cnt-(1 if params.strip().endswith(',') else 0)
        res.append((m.group(1),line0,line1-line0+1,maxd,branches,nparams))
    return res
if __name__=='__main__':
    root=sys.argv[1]; out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in('target','.git','node_modules','output','data','book')]
        for f in fn:
            if f.endswith('.rs'):
                p=os.path.join(dp,f)
                for r in scan(p):
                    out.append({'file':os.path.relpath(p,root),'name':r[0],'line':r[1],'len':r[2],'depth':r[3],'branches':r[4],'params':r[5]})
    json.dump(out,open(sys.argv[2],'w'))
    print(len(out),'functions')
