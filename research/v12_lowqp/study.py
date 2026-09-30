"""Four-shard CAL-only V12 study with hash-locked registration; no holdout CLI."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import time
import cv2
import numpy as np
import torch
from src.codecs.standard import StandardCodec,ffmpeg_available
from src.data.video_dataset import VideoClipDataset
from src.models.codec_search import CANDIDATES,make_candidates,normalized_bpp
from .policy import QPS,TAUS,SLACKS,QP_MODES,finite,grid,choose,measured_names,spatial_distance
from .paired_metrics import compare,curves,summarize

REPO = Path(__file__).resolve().parents[2]
CONFIG = 'configs/v12_lowqp/protocol.json'
PRIMARY = ('r2plus1d_18','r3d_18')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={REPO.as_posix()}',*args],cwd=REPO)

def committed(path,ref='HEAD'):
    raw = git('show',f'{ref}:{path}')
    current = (REPO/path).read_bytes()
    if current!=raw and current.replace(b'\r\n',b'\n')!=raw:
        raise ValueError(f'uncommitted artifact: {path}')
    return raw

def write_json(path,value):
    if path.exists() or path.with_suffix('.sha256').exists():
        raise ValueError('fresh artifact required')
    path.parent.mkdir(parents=True,exist_ok=True)
    raw = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    path.write_bytes(raw)
    path.with_suffix('.sha256').write_bytes(f'{sha(raw)}  {path.name}\n'.encode())

def fingerprint(ids):
    return sha(json.dumps(ids,sort_keys=True,separators=(',',':')).encode())

def protocol(ref):
    if not re.fullmatch('[0-9a-f]{40}',ref):
        raise ValueError('full preregistration hash required')
    git('merge-base','--is-ancestor',ref,'HEAD')
    raw = committed(CONFIG)
    if committed(CONFIG,ref)!=raw:
        raise ValueError('registered config changed')
    cfg = json.loads(raw)
    doc = committed(cfg['preregistration_path'],ref)
    if committed(cfg['preregistration_path'])!=doc or b'\nPREREGISTRATION_LOCKED: true\n' not in doc or sha(raw).encode() not in doc:
        raise ValueError('registration not locked or changed')
    if (cfg['direction'] not in ('spatial','semantic') or cfg['n_sources']!=200 or cfg['shards']!=4
        or cfg['codec']!='h265' or cfg['preset']!='medium' or cfg['qps']!=list(QPS)
        or cfg['seed']!=20261009 or cfg['bootstrap_draws']!=2000
        or cfg['grid']!={'tau_relative':list(TAUS),'rate_slack':list(SLACKS),'qp_modes':{k:list(v) for k,v in QP_MODES.items()}}):
        raise ValueError('protocol constants changed')
    for p,expected in cfg['core_git_blob_sha256'].items():
        if sha(committed(p))!=expected:
            raise ValueError(f'upstream codec/metric changed: {p}')
    for p,key in ((cfg['cal_input_path'],'cal_input_sha256'),(cfg['primary_outcomes_path'],'primary_outcomes_sha256')):
        blob = committed(p)
        if sha(blob)!=cfg[key] or committed(p,ref)!=blob or (REPO/p).read_bytes()!=blob:
            raise ValueError('registered data bytes changed')
        if (REPO/p).with_suffix('.sha256').read_text().split()!=[sha(blob),Path(p).name]:
            raise ValueError('input sidecar mismatch')
    data = json.loads(committed(cfg['cal_input_path']))
    sources = data['sources']
    ids = [s['sequence_id'] for s in sources]
    hashes = [s['source_sha256'] for s in sources]
    if (data['stage']!='calibration' or len(sources)!=200 or len(set(Path(i).stem for i in ids))!=200
        or len(set(hashes))!=200 or fingerprint(ids)!=cfg['source_fingerprint']):
        raise ValueError('CAL cohort changed')
    for s in sources:
        if set(s)!={'sequence_id','source_sha256','measurements'} or not re.fullmatch('[0-9a-f]{64}',s['source_sha256']):
            raise ValueError('source schema or hash invalid')
        if [m['qp'] for m in s['measurements']]!=list(QPS):
            raise ValueError('source QPs changed')
        for m in s['measurements']:
            if (set(m)!={'qp','v2','v6','bpp'} or m['v2'] not in CANDIDATES or m['v6'] not in CANDIDATES
                or set(m['bpp'])!=set(CANDIDATES) or any(not finite(v) or v<=0 for v in m['bpp'].values())):
                raise ValueError('invalid frozen inputs')
    git('diff','--exit-code','HEAD','--','research/v12_lowqp','kaggle/v12_lowqp_cal_cell.sh')
    code = sha(git('ls-tree','-r','HEAD','--','research/v12_lowqp','kaggle/v12_lowqp_cal_cell.sh',*cfg['core_git_blob_sha256']))
    context = {k:cfg[k] for k in ('experiment','direction','repository_url','source_fingerprint','scope','bootstrap_unit')}
    context.update(code_commit=git('rev-parse','HEAD').decode().strip(),code_fingerprint=code,
        preregistration_commit=ref,preregistration_sha256=sha(doc),protocol_sha256=sha(raw),
        cal_input_sha256=cfg['cal_input_sha256'],primary_outcomes_sha256=cfg['primary_outcomes_sha256'],
        upstream_input_commit=cfg['upstream_input_commit'],bootstrap_seed=cfg['seed'],bootstrap_draws=2000,
        teacher=cfg['teacher'],metric_sha256=cfg['core_git_blob_sha256']['src/metrics/bd_rate.py'],
        paired_metrics_sha256=sha(committed('research/v12_lowqp/paired_metrics.py')))
    return cfg,data,context

def source_paths(root,sources):
    if not root.is_dir() or not ffmpeg_available():
        raise ValueError('dataset or FFmpeg unavailable')
    wanted = {Path(s['sequence_id']).stem:s for s in sources}
    found = {}
    for path in root.rglob('*.mp4'):
        if path.stem in wanted:
            if path.stem in found:
                raise ValueError('ambiguous source video ID')
            if path.parent.name!=Path(wanted[path.stem]['sequence_id']).parent.name:
                raise ValueError('source class path mismatch')
            found[path.stem] = path
    if set(found)!=set(wanted):
        raise ValueError('missing locked sources; no replacements')
    for stem,path in found.items():
        if sha(path.read_bytes())!=wanted[stem]['source_sha256']:
            raise ValueError('source bytes changed')
    return {s['sequence_id']:found[Path(s['sequence_id']).stem] for s in sources}

def read_clip(path):
    cap = cv2.VideoCapture(str(path))
    ok,_ = cap.read()
    cap.release()
    if not ok:
        raise ValueError('undecodable source')
    reader = VideoClipDataset.__new__(VideoClipDataset)
    reader.num_frames,reader.frame_size,reader.temporal_stride,reader.train = 16,128,2,False
    return reader._read_clip(str(path))

def validate_row(row,source,direction):
    if (set(row)!={'sequence_id','source_sha256','source_proxy_s','measurements'} or any(row[k]!=source[k] for k in ('sequence_id','source_sha256'))
        or not finite(row['source_proxy_s']) or row['source_proxy_s']<0 or len(row['measurements'])!=5):
        raise ValueError('raw source record invalid or outcome-contaminated')
    for point,fixed in zip(row['measurements'],source['measurements'],strict=True):
        if set(point)!={'qp','candidates'} or point['qp']!=fixed['qp'] or set(point['candidates'])!=measured_names(direction,fixed):
            raise ValueError('raw QPs/candidate set invalid')
        for name,v in point['candidates'].items():
            if (set(v)!={'bpp','coded_bytes','decoded_sha256','proxy_error','encode_decode_s','proxy_s'}
                or not finite(v['bpp']) or abs(v['bpp']-fixed['bpp'][name])>1e-9
                or type(v['coded_bytes']) is not int or v['coded_bytes']<=0
                or abs(v['coded_bytes']*8/(16*128*128)-v['bpp'])>1e-9
                or not re.fullmatch('[0-9a-f]{64}',v['decoded_sha256'])
                or not finite(v['proxy_error']) or not 0<=v['proxy_error']<=2
                or any(not finite(v[k]) or v[k]<0 for k in ('encode_decode_s','proxy_s'))):
                raise ValueError('raw candidate rate/proxy/hash/timing invalid')

def run_shard(root,shard,out,cfg,data,context):
    if type(shard) is not int or shard not in range(4) or out.exists():
        raise ValueError('invalid shard or nonfresh output')
    sources = data['sources'][shard::4]
    paths = source_paths(root,sources)
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    teacher = None
    if cfg['direction']=='semantic':
        from .teacher import Teacher
        teacher = Teacher(cfg['teacher'])
        if teacher.device.type!='cuda':
            raise ValueError('semantic GPU notebook has no CUDA; do not silently run CPU')
    import torchvision
    out.mkdir(parents=True)
    codec = StandardCodec('h265',preset='medium',strict_decode=True)
    raw = out/'shard_records.jsonl'
    trials = 0
    with raw.open('wb') as stream:
        for index,source in enumerate(sources,1):
            clip = read_clip(paths[source['sequence_id']])
            variants = make_candidates(clip)
            if teacher:
                teacher.synchronize()
            start = time.perf_counter()
            source_features = teacher.features(clip) if teacher else None
            if teacher:
                teacher.synchronize()
            row = {k:source[k] for k in ('sequence_id','source_sha256')}
            row.update(source_proxy_s=time.perf_counter()-start,measurements=[])
            for fixed in source['measurements']:
                values = {}
                for name in sorted(measured_names(cfg['direction'],fixed),key=CANDIDATES.index):
                    candidate = variants[name]
                    start = time.perf_counter()
                    decoded,native = codec._encode_decode_clip(candidate,qp=fixed['qp'])
                    seconds = time.perf_counter()-start
                    bpp = normalized_bpp(native,*candidate.shape[1:3])
                    if abs(bpp-fixed['bpp'][name])>1e-9:
                        raise ValueError('encoded bpp differs from pinned CAL stream')
                    if teacher:
                        teacher.synchronize()
                    start = time.perf_counter()
                    error = teacher.score(source_features,decoded) if teacher else spatial_distance(clip,decoded)
                    if teacher:
                        teacher.synchronize()
                    values[name] = {'bpp':bpp,'coded_bytes':round(bpp*16*128*128/8),
                        'decoded_sha256':sha(decoded.tobytes()),'proxy_error':error,
                        'encode_decode_s':seconds,'proxy_s':time.perf_counter()-start}
                    trials += 1
                row['measurements'].append({'qp':fixed['qp'],'candidates':values})
            validate_row(row,source,cfg['direction'])
            stream.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode())
            stream.flush()
            print(f"[V12 {cfg['direction']}] shard={shard} {index}/50 trials={trials}",flush=True)
    manifest = {'kind':'v12_cal_proxy_shard','provenance':context,'shard':shard,'shards':4,'n':len(sources),
        'source_ids':[s['sequence_id'] for s in sources],'records_sha256':sha(raw.read_bytes()),
        'trial_encode_decode_count':trials,'actual_bootstrap_draws':0,
        'device':str(teacher.device) if teacher else 'cpu; no model or labels',
        'versions':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                    'numpy':np.__version__,'opencv':cv2.__version__,
                    'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},
        'timing_scope':'all pilot trial encodes/decodes and proxy; not full frozen V6 selector runtime',
        'mc3_18':'CHƯA ĐO; excluded from CAL workers'}
    write_json(out/'manifest.json',manifest)
    return {'shard':shard,'n':len(sources),'trial_encode_decode_count':trials}

def load_shards(folders,cfg,data,context):
    if len(folders)!=4:
        raise ValueError('four complete shards required')
    by_id,metas,commits = {},{},set()
    for folder in folders:
        meta = json.loads((folder/'manifest.json').read_bytes())
        shard = meta.get('shard')
        if type(shard) is not int or shard not in range(4) or shard in metas:
            raise ValueError('duplicate or invalid shard')
        assigned = data['sources'][shard::4]
        prov = meta.get('provenance',{})
        if (meta.get('kind')!='v12_cal_proxy_shard' or meta.get('n')!=50 or meta.get('shards')!=4
            or meta.get('source_ids')!=[s['sequence_id'] for s in assigned]
            or any(prov.get(k)!=v for k,v in context.items() if k!='code_commit')
            or sha((folder/'manifest.json').read_bytes())!=(folder/'manifest.sha256').read_text().split()[0]
            or meta.get('records_sha256')!=sha((folder/'shard_records.jsonl').read_bytes())
            or meta.get('actual_bootstrap_draws')!=0
            or meta.get('device')!=('cuda' if cfg['direction']=='semantic' else 'cpu; no model or labels')):
            raise ValueError('shard hash/provenance/partition mismatch')
        worker = prov.get('code_commit','')
        if not re.fullmatch('[0-9a-f]{40}',worker):
            raise ValueError('missing worker commit')
        git('merge-base','--is-ancestor',worker,'HEAD')
        commits.add(worker)
        rows = [json.loads(line) for line in (folder/'shard_records.jsonl').read_bytes().splitlines()]
        if len(rows)!=50:
            raise ValueError('incomplete records')
        trials = 0
        for row,source in zip(rows,assigned,strict=True):
            validate_row(row,source,cfg['direction'])
            if row['sequence_id'] in by_id:
                raise ValueError('duplicate source')
            by_id[row['sequence_id']]=row
            trials += sum(len(p['candidates']) for p in row['measurements'])
        if meta['trial_encode_decode_count']!=trials:
            raise ValueError('trial count mismatch')
        metas[shard]={**meta,'manifest_sha256':sha((folder/'manifest.json').read_bytes())}
    if len(commits)!=1 or len(by_id)!=200 or any(m['versions']!=metas[0]['versions'] for m in metas.values()):
        raise ValueError('mixed workers/environments or source count')
    return [by_id[s['sequence_id']] for s in data['sources']],[metas[i] for i in range(4)]

def primary_rows(outcomes,pixels,sources,direction,p):
    rows,choices,proxy_delta = [],[],[]
    for source,pixel,old in zip(sources,pixels,outcomes['sources'],strict=True):
        if any(old[k]!=source[k] for k in ('sequence_id','source_sha256')):
            raise ValueError('primary outcome/source mismatch')
        points,decisions = [],[]
        for fixed,obs,truth in zip(source['measurements'],pixel['measurements'],old['measurements'],strict=True):
            if truth['qp']!=fixed['qp'] or set(truth['candidates'])!=set(CANDIDATES):
                raise ValueError('primary outcome QPs/candidates mismatch')
            values = {name:{k:v[k] for k in ('bpp','proxy_error')} for name,v in obs['candidates'].items()}
            name,reason = (fixed['v6'],'baseline') if p is None else choose(direction,fixed['qp'],fixed['v6'],values,p)
            arms = {}
            for arm,candidate in (('identity','identity128'),('v6',fixed['v6']),('area112','area112'),('policy',name)):
                correct = truth['candidates'][candidate]
                if set(correct)!=set(PRIMARY) or any(type(v) is not bool for v in correct.values()):
                    raise ValueError('outcome analyzer/schema contamination')
                arms[arm]={'bpp':fixed['bpp'][candidate],'analyzers':{m:{'correct':correct[m]} for m in PRIMARY}}
            points.append({'qp':fixed['qp'],'arms':arms})
            decisions.append({'qp':fixed['qp'],'candidate':name,'v6':fixed['v6'],'reason':reason,
                'bpp':fixed['bpp'][name],'proxy_error':values[name]['proxy_error'],
                'decoded_sha256':obs['candidates'][name]['decoded_sha256']})
            if fixed['qp'] in (30,35,40):
                proxy_delta.append(values[fixed['v6']]['proxy_error']-values[name]['proxy_error'])
        rows.append({'sequence_id':source['sequence_id'],'measurements':points})
        choices.append({k:source[k] for k in ('sequence_id','source_sha256')}|{'measurements':decisions})
    return rows,choices,float(np.mean(proxy_delta))

def points(rows):
    return {m:compare(curves(rows,'identity',m),curves(rows,'policy',m)) for m in PRIMARY}

def feasible(report,baseline,improvement):
    return (finite(improvement) and improvement>1e-6 and all(
        all(finite(report[m][k]) for k in ('bd_rate_top1_pct','bd_accuracy_top1_pp','min_same_qp_top1_gap_pp'))
        and finite(baseline[m]['bd_rate_top1_pct']) and report[m]['bd_rate_top1_pct']<-10
        and report[m]['bd_rate_top1_pct']<=baseline[m]['bd_rate_top1_pct']+2
        and report[m]['bd_accuracy_top1_pp']>0 and report[m]['min_same_qp_top1_gap_pp']>=-1-1e-9 for m in PRIMARY))

def rank_key(row):
    rates = [row['analyzers'][m]['bd_rate_top1_pct'] for m in PRIMARY]
    p = row['policy']
    return (-row['mean_lowqp_proxy_reduction'],max(rates),sum(rates),row['switch_count'],
            p['tau_relative'],p['rate_slack'],list(QP_MODES).index(p['qp_mode']))

def calibrate(folders,out,cfg,data,context):
    if out.exists() or out.with_suffix('.sha256').exists():
        raise ValueError('repeated CAL selection prohibited')
    pixels,manifests = load_shards(folders,cfg,data,context)
    outcomes = json.loads(committed(cfg['primary_outcomes_path']))
    if (outcomes.get('stage')!='calibration' or outcomes.get('analyzers')!=list(PRIMARY)
        or outcomes.get('source_fingerprint')!=cfg['source_fingerprint'] or len(outcomes.get('sources',[]))!=200):
        raise ValueError('primary CAL index mismatch')
    baseline,_,_ = primary_rows(outcomes,pixels,data['sources'],cfg['direction'],None)
    base_metrics = points(baseline)
    rows = []
    for p in grid():
        values,choices,reduction = primary_rows(outcomes,pixels,data['sources'],cfg['direction'],p)
        report = points(values)
        switches = Counter(str(m['qp']) for s in choices for m in s['measurements'] if m['candidate']!=m['v6'])
        rows.append({'policy':p,'analyzers':report,'mean_lowqp_proxy_reduction':reduction,
            'switch_count':sum(switches.values()),'switches_by_qp':dict(switches),'feasible':feasible(report,base_metrics,reduction)})
    valid = [row for row in rows if row['feasible']]
    chosen = min(valid,key=rank_key) if valid else None
    comparisons,choices,qualified = {},[],False
    if chosen:
        values,choices,_ = primary_rows(outcomes,pixels,data['sources'],cfg['direction'],chosen['policy'])
        draws = np.random.default_rng(cfg['seed']).integers(0,200,size=(2000,200))
        comparisons = {f'policy_vs_{arm}':{m:summarize(values,arm,'policy',m,draws) for m in PRIMARY} for arm in ('identity','v6','area112')}
        qualified = all(x['valid_draws']>=1900 and x['ci95'] is not None and all(finite(v) for v in x['ci95'])
            for m in comparisons['policy_vs_identity'].values() for x in m['bootstrap'].values())
    report = {'kind':'v12_calibration_result','provenance':context,'n':200,'shard_manifests':manifests,
        'sources':[{k:s[k] for k in ('sequence_id','source_sha256')} for s in data['sources']],
        'baseline_primary_metrics':base_metrics,'grid':rows,'feasible_count':len(valid),
        'selected_policy':chosen['policy'] if chosen else None,'selected_choices':choices,'comparisons':comparisons,
        'status':'FREEZE_BEFORE_NEW_DEV_PROTOCOL' if chosen and qualified else 'NO_GO_STOP_BEFORE_DEV',
        'actual_bootstrap_draws':2000 if chosen else 0,
        'bootstrap_interpretation':'descriptive selected-policy CAL intervals; selection uncertainty not accounted for',
        'mc3_18':'CHƯA ĐO; excluded from CAL','dev':'CHƯA ĐO','holdout':'CHƯA ĐO',
        'original_gate':'unchanged -15%; not confirmatory on reused CAL'}
    write_json(out,report)
    return {'status':report['status'],'selected_policy':report['selected_policy'],'feasible_count':len(valid),'result_sha256':sha(out.read_bytes())}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prereg-commit',required=True)
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('preflight')
    s = sub.add_parser('shard')
    s.add_argument('--source-root',type=Path,required=True)
    s.add_argument('--shard',type=int,choices=range(4),required=True)
    s.add_argument('--out-dir',type=Path,required=True)
    m = sub.add_parser('calibrate')
    m.add_argument('--shard-dir',type=Path,action='append',required=True)
    m.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    cfg,data,context = protocol(args.prereg_commit)
    if args.command=='preflight':
        result=context
    elif args.command=='shard':
        result=run_shard(args.source_root,args.shard,args.out_dir,cfg,data,context)
    else:
        result=calibrate(args.shard_dir,args.out,cfg,data,context)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)

if __name__=='__main__':
    main()
