"""Publish one private V12 CAL notebook pinned to the inspected repository HEAD."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .study import REPO,protocol,sha

def payload(commit,prereg,account,slug,shard,cfg):
    if any(not re.fullmatch('[0-9a-f]{40}',v) for v in (commit,prereg)):
        raise ValueError('full revision hashes required')
    if (not re.fullmatch('[a-z0-9]+',account) or not re.fullmatch('[a-z0-9][a-z0-9-]*',slug)
        or type(shard) is not int or shard not in range(4)
        or cfg['repository_url'] not in ('https://github.com/munnn01/preeee.git','https://github.com/munnn01/test_pre.git')):
        raise ValueError('invalid destination')
    shell = (REPO/'kaggle/v12_lowqp_cal_cell.sh').read_text(encoding='utf-8')
    for key,value in {'REF':commit,'PREREG':prereg,'SHARD':str(shard),'GITHUB':cfg['repository_url'],'DIRECTION':cfg['direction']}.items():
        shell=shell.replace(f'__{key}__',value)
    if not shell.startswith('%%bash\n') or re.search('__[A-Z_]+__',shell):
        raise ValueError('unsubstituted shell template')
    book={'cells':[{'id':f"v12-{cfg['direction']}-{shard}",'cell_type':'code','execution_count':None,
                   'metadata':{},'outputs':[],'source':shell.splitlines(keepends=True)}],
          'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
          'nbformat':4,'nbformat_minor':5}
    meta={'id':f'{account}/{slug}','title':slug,'code_file':'notebook.ipynb','language':'python',
          'kernel_type':'notebook','is_private':True,'enable_gpu':cfg['direction']=='semantic','enable_internet':True,
          'dataset_sources':['qktttttttttt/kineticscleaned'],'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    return book,meta

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('commit','prereg-commit','account','slug','payload-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--shard',type=int,choices=range(4),required=True)
    p.add_argument('--pool',type=Path,default=Path('D:/STUDY/LAB/pool.json'))
    p.add_argument('--write-only',action='store_true')
    args=p.parse_args()
    cfg,_,context=protocol(args.prereg_commit)
    if context['code_commit']!=args.commit:
        raise ValueError('push must pin inspected local HEAD')
    book,meta=payload(args.commit,args.prereg_commit,args.account,args.slug,args.shard,cfg)
    target=Path(args.payload_dir)
    if target.exists():
        raise ValueError('fresh payload directory required')
    target.mkdir(parents=True)
    hashes={}
    for name,value in (('notebook.ipynb',book),('kernel-metadata.json',meta)):
        raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
        (target/name).write_bytes(raw)
        hashes[name]=sha(raw)
    print(json.dumps({'generated':meta['id'],'payload_sha256':hashes},indent=2),flush=True)
    if args.write_only:
        return
    credentials=json.loads(args.pool.read_text(encoding='utf-8-sig'))
    token=credentials.get(args.account)
    if not isinstance(token,str) or not token.startswith('KGAT_'):
        raise ValueError('account credential missing')
    env=os.environ.copy()
    env.update(KAGGLE_API_TOKEN=token,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-c','from kaggle.cli import main; main()']
    def call(params):
        result=subprocess.run(command+params,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        output=result.stdout+result.stderr
        if 'KGAT_' in output or token in output:
            raise RuntimeError('refusing credential-bearing output')
        return result.returncode,output
    code,output=call(['kernels','status',meta['id']])
    if code==0 and any(v in output.upper() for v in ('RUNNING','QUEUED','PENDING')):
        raise ValueError('kernel already active; do not replace')
    if code!=0 and '404' not in output:
        raise RuntimeError('kernel status check failed; '+output)
    code,output=call(['kernels','push','-p',str(target)])
    print(output,flush=True)
    if code or 'successfully pushed' not in output.lower():
        raise SystemExit(code or 1)

if __name__=='__main__':
    main()
