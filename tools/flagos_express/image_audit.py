#!/usr/bin/env python3
"""Validate complete frozen role trees and exactly the committed modifications."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/opt/flagos-express')
RELEASE=ROOT/'release-audit'


def digest(value):return hashlib.sha256(value).hexdigest()


def main():
    phase=sys.argv[1];assert phase in ('base','final')
    release=json.loads((RELEASE/'release.json').read_text())
    for name,record in release['files'].items():
        assert digest((RELEASE/name).read_bytes())==record['sha256'],name
    topology=json.loads((RELEASE/'topology.json').read_text())
    rank=json.loads((RELEASE/'rank-manifest.json').read_text())['files']
    sources={}
    for role in ('prefill','decode'):
        folder=Path('/sgl-workspace/roles')/role/'python/sglang'
        manifest_path=ROOT/'manifests'/(role+'-python.json')
        expected=json.loads(manifest_path.read_text())
        assert digest(json.dumps(expected,sort_keys=True).encode())==release['base_manifests'][role]
        if phase=='final':
            if role=='prefill':
                for record in topology['files']:
                    name=record['repo_path'].removeprefix('python/sglang/')
                    assert expected[name]==record['baseline_sha256']
                    expected[name]=record['candidate_sha256']
            change=rank['runtime-'+role+'/sglang/srt/disaggregation/decode.py']
            assert expected['srt/disaggregation/decode.py']==change['before_sha256']
            expected['srt/disaggregation/decode.py']=change['after_sha256']
        actual={str(p.relative_to(folder)):digest(p.read_bytes()) for p in folder.rglob('*.py')}
        assert actual==expected,(role,'unexpected source modifications')
        if phase=='final':manifest_path.write_text(json.dumps(actual,indent=2)+'\n')
        sources[role]=dict(path=str(folder.parent),python_files=len(actual),
            manifest_sha256=digest(json.dumps(actual,sort_keys=True).encode()))
    if phase=='final':
        adapter=ROOT/'deps/monitoring/sglang_pd_monitoring.py'
        exporter=ROOT/'deps/monitoring/exporter/exporter.py'
        assert digest(adapter.read_bytes())==release['files']['runtime/sglang_pd_monitoring.py']['sha256']
        assert digest(exporter.read_bytes())==release['files']['runtime/exporter/exporter.py']['sha256']
        ast.parse(adapter.read_text());ast.parse(exporter.read_text())
        for path in [adapter,exporter]:
            assert '10.0.0.' not in path.read_text() and '/public-flash/' not in path.read_text()
        versions=json.loads((ROOT/'versions.json').read_text())
        versions.update(tag=release['version'],sglang_source_commit=release['sglang_commit'],role_sources=sources)
        versions['monitoring']=dict(commit=release['sglang_commit'],adapter_sha256=digest(adapter.read_bytes()),
            exporter_sha256=digest(exporter.read_bytes()),token_lengths='D terminal actual lengths by initial P domain and D domain',
            interval_seconds=5,configuration='Runtime mount; no deployment addresses or private bindings packaged')
        versions['role_source_note']='Frozen base role sources plus committed topology/rank hooks; manifests identify exact runtime bytes.'
        (ROOT/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')
        entry=ROOT/'entrypoint';text=entry.read_text()
        needle='2p2d: two servers per P/D instance.'
        assert text.count(needle)==1
        entry.write_text(text.replace(needle,needle+' 4p2d: four P servers and two D servers.',1))
        result=dict(version=release['version'],sglang_commit=release['sglang_commit'],role_sources=sources,
            monitoring=versions['monitoring'],private_runtime_configuration_excluded=True)
        (RELEASE/'validated.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    else:print(json.dumps(dict(phase=phase,full_role_source_verification=True)))


if __name__=='__main__':main()
