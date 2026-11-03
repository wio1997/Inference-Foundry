"""Bounded scalar-control derivation, not an NPU simulator or physical-IO model.

Only the default-zero FLOAT32 branch is supported. DMA descriptor encodings are
retained as raw values. Host dictionaries stand for already-saved byte records;
no address from a packet is ever dereferenced. Vector/cache/pipeline effects and
timing are explicitly outside the model. Unknown instructions fail closed.
"""
from pathlib import Path
import hashlib,re,struct

MASK=(1<<64)-1
LINE=re.compile(r'^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)(.*)$')


def elf_sections(blob):
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',blob)
    assert h[0][:7]==b'\x7fELF\x02\x01\x01'and h[2]==4137 and h[8]==64 and h[11]==64
    assert h[12]>0 and 0<=h[13]<h[12]and h[6]+h[11]*h[12]<=len(blob)
    rows=[struct.unpack_from('<IIQQQQIIQQ',blob,h[6]+i*64)for i in range(h[12])]
    names=rows[h[13]];strings=blob[names[4]:names[4]+names[5]];out={}
    for row in rows:
        assert row[4]+row[5]<=len(blob)
        name=strings[row[0]:].split(b'\0',1)[0].decode()
        data=blob[row[4]:row[4]+row[5]]
        assert name not in out
        out[name]=dict(name=name,address=row[3],offset=row[4],bytes=row[5],data=data,
            sha256=hashlib.sha256(data).hexdigest(),type=row[1],flags=row[2])
    return out


def instructions(text):
    out={}
    for line_no,line in enumerate(text.splitlines(),1):
        m=LINE.match(line)
        if m:
            pc=int(m[1],16);assert pc not in out
            out[pc]=dict(pc=pc,word=int(m[2],16),op=m[3],args=[a.strip()for a in m[4].split('<')[0].strip().split(',')if a.strip()],line=line_no)
    return out


def scalar_trace(code,record,destination,flowtable,block):
    assert len(record)==96 and 0<=block<48
    assert struct.unpack_from('<II',record,88)==(0,0),'outside zero-FLOAT32 supported domain'
    # A deliberately artificial scalar/UB view. These are Python-dict keys.
    ubbase=0x10000000;parabase=0x40000000;regs={'CTRL':0};memory={};events=[];branches=[];path=[]
    for i,b in enumerate(struct.pack('<QQ',destination,flowtable)):memory[parabase+i]=b
    pc=0;flag=None
    def sign(v,bits=64):
        v&=(1<<bits)-1;return v-(1<<bits)if v&(1<<(bits-1))else v
    def get(s):
        if s.startswith('#'):return int(s[1:])
        if s=='SYS_VA_BASE':return ubbase-0x80000
        if s=='PARA_BASE':return parabase
        if s=='BLOCKID':return block
        if s=='COREID':return 0  # Only stack arithmetic uses this in supported paths.
        if s=='PC':return pc
        assert s in regs,('unbound register',pc,s)
        return regs[s]
    def number(s):
        v=get(s);assert isinstance(v,int),('encoded/noninteger operand',pc,s,v)
        return v
    def put(s,v):regs[s]=v&MASK if isinstance(v,int)else v
    def load(address,width):
        assert all(address+i in memory for i in range(width)),('unbound modeled bytes',pc,address,width)
        return int.from_bytes(bytes(memory[address+i]for i in range(width)),'little')
    def address(s):
        assert s.startswith('[')and s.endswith(']');return number(s[1:-1])
    def compare(condition,a,b):
        x,y=sign(a),sign(b)
        return {'EQ':x==y,'NE':x!=y,'LT':x<y,'LE':x<=y,'GE':x>=y,'GT':x>y}[condition]
    while len(path)<20000:
        assert pc in code,('missing instruction',pc)
        ins=code[pc];op=ins['op'];a=ins['args'];path.append(pc);next_pc=pc+4
        if op=='END':break
        if op=='MOV':put(a[0],get(a[1]))
        elif op=='MOVK':
            shift=number(a[2])*16;put(a[0],(number(a[0])&~(0xffff<<shift))|((number(a[1])&0xffff)<<shift))
        elif op=='MOVX8':put(a[0],('UNDECODED_MOVX8',number(a[1])))
        elif op=='NEG.s64':put(a[0],-number(a[1]))
        elif op=='SIGNEXT.s32':put(a[0],sign(number(a[1]),32))
        elif op=='ZEROEXT.u32':put(a[0],number(a[1])&0xffffffff)
        elif op in ['SHL.b64','SHR.u64','SHR.s64']:
            v=number(a[0]);shift=number(a[1]);assert 0<=shift<64
            put(a[0],v<<shift if op=='SHL.b64'else sign(v)>>shift if op=='SHR.s64'else v>>shift)
        elif op=='INSERT':
            src,msb,lsb=map(number,a[1:]);assert 0<=lsb<=msb<64
            bits=((1<<(msb-lsb+1))-1)<<lsb;put(a[0],(number(a[0])&~bits)|((src<<lsb)&bits))
        elif op in ['ADD.s64','SUB.s64','MUL.s64','AND.b64','OR.b64','MADD.s64']:
            x,y=number(a[1]),number(a[2])
            put(a[0],x+y if op=='ADD.s64'else x-y if op=='SUB.s64'else x*y if op=='MUL.s64'else x&y if op=='AND.b64'else x|y if op=='OR.b64'else number(a[0])+x*y)
        elif op.startswith('LD.')or op.startswith('LDP.'):
            pair=op.startswith('LDP.');width=int(op.split('.b')[1])//8;index=2 if pair else 1
            addr=address(a[index])+number(a[index+1]);assert len(a)==index+2 or a[index+2]=='#0'
            put(a[0],load(addr,width))
            if pair:put(a[1],load(addr+width,width))
        elif op.startswith('CMP.'):
            flag=compare(op.split('.')[1],number(a[0]),number(a[1]))
        elif op in ['JUMP','JUMPC']or op.startswith('JUMPCMP.'):
            target=number(a[0]);assert target in code
            if op=='JUMP':take=True
            elif op=='JUMPC':assert flag is not None;take=flag
            else:take=compare(op.split('.')[1],number(a[1]),number(a[2]))
            branches.append(dict(pc=pc,target=target,taken=take))
            if take:next_pc=target
        elif op=='MOV_OUT_TO_UB':
            dst,src=address(a[0]),address(a[1]);events.append(dict(kind='INPUT_TRANSFER_SITE',pc=pc,destination=dst,source=src,raw_control=get(a[2])))
            if src==flowtable:
                # Source-record binding is an explicit supported-domain premise,
                # not a general decoding of the MOVX8 transfer descriptor.
                for i,b in enumerate(record):memory[ubbase+dst+i]=b
            else:
                assert pc==0x14c,('unsupported extra source transfer',pc,src)
                # Constants feed only vectors after scalar fields are loaded.
        elif op.startswith('MOV_UB_TO_OUT'):
            events.append(dict(kind='OUTPUT_STORE_SITE',pc=pc,op=op,destination=address(a[0]),
                destination_relative=address(a[0])-destination,UB_source=address(a[1]),raw_control=number(a[2]),extra=[get(x)for x in a[3:]]))
        elif op in ['MOVEMASK','NOP','DCCI']or op.startswith(('SET_FLAG.','WAIT_FLAG.','BAR.','MOVEV.')):
            # These have no modeled scalar destination. Their vector, cache,
            # ordering and physical effects are NOT simulated or costed.
            pass
        else:raise AssertionError(('unsupported instruction',pc,op,a))
        pc=next_pc
    else:raise AssertionError('scalar trace budget exceeded')
    return dict(block=block,path=path,branches=branches,events=events,
        instruction_sites_visited=len(path),ended_at=pc,
        scalar_control_only=True,hardware_execution_or_timing_proved=False)
