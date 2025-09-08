# file: sph_parser.py
import os, re, json, csv, math, sys

RANKS = "AKQJT98765432"

def load_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def get_nodes(doc):
    if isinstance(doc, dict) and "Nodes" in doc:
        return doc["Nodes"]
    raise ValueError("Unexpected JSON format")

def get_path(node):
    # SPH sometimes uses different keys for path/name
    for k in ("Path","Node","Name","Title"):
        if k in node and isinstance(node[k], str):
            return node[k]
    return "UNKNOWN"

def last_action(path):
    parts=[p.strip() for p in path.split(">") if p.strip()]
    return parts[-1].upper() if parts else path.upper()

def grid_from_hands(hands):
    def hand_class(c):
        r1,s1,r2,s2=c[0],c[1],c[2],c[3]
        if r1==r2: return r1+r2
        hi,lo=(r1,r2) if RANKS.index(r1)<RANKS.index(r2) else (r2,r1)
        return hi+lo+("s" if s1==s2 else "o")
    acc,cnt={},{}
    for h in hands:
        c=h.get("Cards","")
        if len(c)!=4: continue
        f=float(h.get("Abs") or h.get("Played") or 0.0)
        cls=hand_class(c)
        acc[cls]=acc.get(cls,0.0)+f
        cnt[cls]=cnt.get(cls,0)+1
    for k in acc: acc[k]/=cnt[k]
    mat=[]
    for r1 in RANKS:
        row=[]
        for r2 in RANKS:
            if r1==r2: key=r1+r2
            elif RANKS.index(r1)<RANKS.index(r2): key=r1+r2+"s"
            else: key=r2+r1+"o"
            row.append(f"{acc.get(key,0.0):.2f}")
        mat.append(row)
    return mat

def write_csv(role, mat, outdir):
    os.makedirs(outdir, exist_ok=True)
    fn=os.path.join(outdir,f"{role}.csv")
    with open(fn,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow([""]+list(RANKS))
        for r,row in zip(list(RANKS),mat):
            w.writerow([r]+row)
    return fn

def classify_roles(nodes):
    roles={}
    totals=[]
    for n in nodes:
        la=last_action(get_path(n))
        if la.startswith("BB RAISE"):
            nums=re.findall(r'(\d+(?:\.\d+)?)', la)
            if nums: totals.append(float(nums[-1]))
    totals=sorted(set(totals))
    s1=totals[0] if totals else None
    s2=totals[1] if len(totals)>1 else None

    for n in nodes:
        la=last_action(get_path(n))
        if la.startswith("BB RAISE"):
            if "ALL" in la: roles["oop_allin"]=n
            else:
                nums=re.findall(r'(\d+(?:\.\d+)?)', la)
                tot=float(nums[-1]) if nums else None
                if s1 and tot==s1: roles["oop_raise_s1"]=n
                elif s2 and tot==s2: roles["oop_raise_s2"]=n
        elif la.startswith("BB CALL"): roles["oop_call"]=n
        elif re.match(r'^(UTG|HJ|CO|BTN|SB)\s+CALL', la): roles["ip_call"]=n
        elif re.match(r'^(UTG|HJ|CO|BTN|SB)\s+BET', la): roles["ip_open"]=n
    return roles

def main(infile, outdir="export_ranges"):
    doc=load_json(infile)
    nodes=get_nodes(doc)
    roles=classify_roles(nodes)
    written=[]
    for role,node in roles.items():
        mat=grid_from_hands(node.get("Hands",[]))
        written.append(write_csv(role,mat,outdir))
    print("Wrote files:")
    for w in written: print(" -", w)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python sph_parser.py <file.json>")
    else:
        main(sys.argv[1])