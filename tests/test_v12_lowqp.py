import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from research.v12_lowqp import policy,study
from research.v12_lowqp import paired_metrics
from research.v12_lowqp.push import payload,require_new_destination
from types import SimpleNamespace

P={'tau_relative':0.1,'rate_slack':0.5,'qp_mode':'lowmid'}

def candidates(direction):
    names=policy.measured_names(direction,{'v6':'area112'})
    return {n:{'bpp':.75 if n=='area112' else 1.0,'proxy_error':.5 if n=='area112' else .1} for n in names}

def test_grid_and_lowqp_direct_identity_can_override_v6_without_v2():
    assert len(policy.grid())==24
    assert policy.choose('spatial',30,'area112',candidates('spatial'),P)[0]=='identity128'
    assert policy.choose('spatial',35,'area112',candidates('spatial'),P)[0]=='identity128'
    assert policy.choose('spatial',40,'area112',candidates('spatial'),P)[0]=='identity128'
    assert policy.choose('spatial',45,'area112',candidates('spatial'),P)[0]=='area112'
    assert policy.choose('spatial',50,'area112',candidates('spatial'),P)[0]=='area112'

def test_rate_budget_and_strict_proxy_gain():
    x=candidates('spatial')
    assert policy.choose('spatial',30,'area112',x,{**P,'rate_slack':.1})[0]=='area112'
    x['identity128']['proxy_error']=.25
    assert policy.choose('spatial',30,'area112',x,{**P,'tau_relative':.5})[0]=='area112'
    x['area112']['proxy_error']=0
    x['identity128']['proxy_error']=0
    assert policy.choose('spatial',30,'area112',x,P)[0]=='area112'

@pytest.mark.parametrize('bad',[{'label':3},{'correct':True},{'mc3_18':0.1},{'proxy_error':float('nan')}])
def test_selector_rejects_outcome_fields_and_nonfinite_values(bad):
    x=candidates('spatial')
    x['identity128'].update(bad)
    with pytest.raises(ValueError):
        policy.choose('spatial',30,'area112',x,P)

def test_semantic_rerank_can_choose_nonidentity_candidate_deterministically():
    x=candidates('semantic')
    x['area96']={'bpp':.6,'proxy_error':.05}
    x['area112_up128']={'bpp':.7,'proxy_error':.05}
    assert policy.choose('semantic',30,'area112',x,P)[0]=='area96'
    x['area96']['bpp']=1.1
    x['area112_up128']['bpp']=1.1
    assert policy.choose('semantic',30,'area112',x,P)[0]=='identity128'
    assert policy.choose('semantic',40,'area112',x,{**P,'qp_mode':'low'})[0]=='area112'

def test_spatial_and_dense_feature_distances():
    clip=np.random.default_rng(6).integers(0,256,size=(16,128,128,3),dtype=np.uint8)
    assert policy.spatial_distance(clip,clip)==0
    assert 0<policy.spatial_distance(clip,np.zeros_like(clip))<=2
    a=(torch.ones(2,3,4,4),torch.ones(2,5,2,2))
    assert policy.dense_feature_distance(a,a)==0
    assert policy.dense_feature_distance(a,tuple(-x for x in a))==pytest.approx(2)
    bad=(a[0].clone(),a[1].clone())
    bad[0][0,0,0,0]=float('nan')
    with pytest.raises(ValueError):
        policy.dense_feature_distance(a,bad)

def test_primary_gate_keeps_strict_thresholds_and_regression_budget():
    base={m:{'bd_rate_top1_pct':-14} for m in study.PRIMARY}
    trial={m:{'bd_rate_top1_pct':-12,'bd_accuracy_top1_pp':2,'min_same_qp_top1_gap_pp':-1} for m in study.PRIMARY}
    assert study.feasible(trial,base,.01)
    assert not study.feasible(trial,base,1e-6)
    trial['r3d_18']['bd_rate_top1_pct']=-11.99
    assert not study.feasible(trial,base,.01)
    trial['r3d_18']['bd_rate_top1_pct']=-10
    assert not study.feasible(trial,base,.01)

def fixture_source(i):
    fixed={'qp':30,'v2':'identity128','v6':'area112',
           'bpp':{n:(.375 if n=='area112' else .5) for n in policy.CANDIDATES}}
    source={'sequence_id':f'fixture/source{i}.mp4','source_sha256':f'{i+1:064x}',
            'measurements':[{**fixed,'qp':qp} for qp in policy.QPS]}
    row={k:source[k] for k in ('sequence_id','source_sha256')}
    row.update(source_proxy_s=0.0,measurements=[{'qp':qp,'candidates':{n:{'bpp':fixed['bpp'][n],
        'coded_bytes':int(fixed['bpp'][n]*32768),'decoded_sha256':'a'*64,'proxy_error':.1,
        'encode_decode_s':.01,'proxy_s':.001} for n in ('identity128','area112')}} for qp in policy.QPS])
    return source,row

def test_raw_records_reject_labels_swapped_qps_and_corrupt_bytes():
    source,row=fixture_source(0)
    study.validate_row(row,source,'spatial')
    for change in ('label','qp','bytes'):
        bad=deepcopy(row)
        if change=='label': bad['label']=0
        if change=='qp': bad['measurements'][0]['qp']=35
        if change=='bytes': bad['measurements'][0]['candidates']['area112']['coded_bytes']+=1
        with pytest.raises(ValueError): study.validate_row(bad,source,'spatial')

def test_four_shards_merge_by_locked_source_order_and_reject_tamper(tmp_path,monkeypatch):
    sources,rows=zip(*(fixture_source(i) for i in range(200)))
    context={'direction':'spatial','code_commit':'b'*40}
    folders=[]
    monkeypatch.setattr(study,'git',lambda *args:b'')
    for shard in range(4):
        folder=tmp_path/f'shard{shard}'
        folder.mkdir()
        raw=('\n'.join(json.dumps(row) for row in rows[shard::4])+'\n').encode()
        (folder/'shard_records.jsonl').write_bytes(raw)
        study.write_json(folder/'manifest.json',{'kind':'v12_cal_proxy_shard','provenance':context,'shard':shard,
            'shards':4,'n':50,'source_ids':[s['sequence_id'] for s in sources[shard::4]],
            'records_sha256':study.sha(raw),'actual_bootstrap_draws':0,'device':'cpu; no model or labels',
            'trial_encode_decode_count':500,'versions':{'fixture':1}})
        folders.append(folder)
    ordered,_=study.load_shards(list(reversed(folders)),{'direction':'spatial'},{'sources':list(sources)},context)
    assert [r['sequence_id'] for r in ordered]==[s['sequence_id'] for s in sources]
    with pytest.raises(ValueError): study.load_shards([folders[0]]*4,{'direction':'spatial'},{'sources':list(sources)},context)
    (folders[0]/'shard_records.jsonl').write_bytes(raw+b'\n')
    with pytest.raises(ValueError): study.load_shards(folders,{'direction':'spatial'},{'sources':list(sources)},context)

def test_bootstrap_functions_are_verbatim_original_and_keep_whole_source_pairing():
    path=Path(paired_metrics.__file__).with_name('bootstrap_reference.py.txt')
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='4c461954ed076f9a276c70f693cfcb3ce3ee054d1070529de5b1f379c497abc9'
    original={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
    current={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(paired_metrics.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    for name in ('curves','compare','summarize'): assert original[name]==current[name]
    rows=[]
    for i in range(4):
        points=[]
        for qp in policy.QPS:
            arms={a:{'bpp':(60-qp)/20*(1 if a=='identity' else .8),'analyzers':{'model':{'correct':i<(60-qp)/10}}} for a in ('identity','policy')}
            points.append({'qp':qp,'arms':arms})
        rows.append({'measurements':points})
    result=paired_metrics.summarize(rows,'identity','policy','model',np.array([[0,1,2,3],[3,2,1,0]]))
    assert result['metrics']['bd_rate_top1_pct']==pytest.approx(-20)
    assert result['bootstrap']['bd_rate_top1_pct']['requested_draws']==2
    assert result['bootstrap']['bd_rate_top1_pct']['valid_draws']==2
    assert result['bootstrap']['bd_rate_top1_pct']['ci95']==pytest.approx([-20,-20])

def test_notebooks_pin_correct_repo_revision_gpu_and_exclude_holdout():
    for direction,repo in (('spatial','preeee'),('semantic','test_pre')):
        book,meta=payload('a'*40,'b'*40,'example1','v12-new-s0',0,
            {'direction':direction,'repository_url':f'https://github.com/munnn01/{repo}.git'})
        shell=''.join(book['cells'][0]['source'])
        assert meta['is_private'] and meta['enable_gpu']==(direction=='semantic')
        assert f'github.com/munnn01/{repo}.git' in shell and 'a'*40 in shell and 'b'*40 in shell
        assert '__REF__' not in shell and 'holdout' not in shell and 'mc3_18' not in shell
        assert 'research.v12_lowqp.study' in shell and 'shard --source-root' in shell

def fake_listing(pages):
    calls=[]
    def listing(**kwargs):
        calls.append(kwargs)
        return pages[len(calls)-1]
    return SimpleNamespace(kernels_list_with_response=listing),calls

def listing_page(refs,cursor=None):
    return SimpleNamespace(kernels=[SimpleNamespace(ref=r) for r in refs],next_page_token=cursor)

def test_publisher_checks_every_owned_page_before_allowing_a_new_destination():
    api,calls=fake_listing([listing_page(['example1/old-private'],'page2'),listing_page(['example1/old2'])])
    proof=require_new_destination(api,'example1/new')
    assert proof['destination_absent'] and proof['owned_count']==2 and proof['pages']==2
    assert calls==[{'mine':True,'page_size':100,'page_token':None},
                   {'mine':True,'page_size':100,'page_token':'page2'}]

@pytest.mark.parametrize('pages',[
    [listing_page(['example1/old'],'page2'),listing_page(['example1/new'])],
    [listing_page(['different-owner/old'])],
    [listing_page([])],
    [None],
    [listing_page(['example1/old'],'same'),listing_page(['example1/other'],'same')],
])
def test_publisher_rejects_existing_names_wrong_owner_and_incomplete_listings(pages):
    api,_=fake_listing(pages)
    with pytest.raises(ValueError):
        require_new_destination(api,'example1/new')
