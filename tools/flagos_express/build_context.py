#!/usr/bin/env python3
"""Export a public, committed overlay for the existing sanitized runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ASSETS = 'examples/monitoring/ascend_pd/4p2d'
BASE = 'harbor.baai.ac.cn/flagos-inner-models-release/glm-5.3-flash-w8a8-mtp-flagos-express:2026-10-28'
BASE_ID = 'sha256:7de5a8779d489607082c16554f6c73969c800495003c38753f3ac77fa37a0686'
EXPORTS = {
    'audit.py': 'tools/flagos_express/image_audit.py',
    'prefill-topology.patch': 'tools/flagos_express/runtime-overlays/glm53-prefill-4node.patch',
    'topology.json': 'tools/flagos_express/runtime-overlays/glm53-prefill-4node.json',
    'runtime-prefill-rank.patch': ASSETS+'/patches/runtime-prefill-rank.patch',
    'runtime-decode-rank.patch': ASSETS+'/patches/runtime-decode-rank.patch',
    'rank-manifest.json': ASSETS+'/patches/manifest.json',
    'runtime/sglang_pd_monitoring.py': 'python/sglang/srt/observability/pd_monitoring.py',
    'runtime/exporter/exporter.py': ASSETS+'/exporter/exporter.py',
}


def blob(repo, commit, path):
    return subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',args.tag):raise ValueError('invalid tag')
    commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse',args.commit+'^{commit}'],text=True).strip()
    data={name:blob(args.repo,commit,source) for name,source in EXPORTS.items()}
    assert data['runtime/sglang_pd_monitoring.py']==blob(args.repo,commit,ASSETS+'/native/sglang_pd_monitoring.py')
    # The base image is already privacy-audited. Do not copy a worktree, config,
    # tests, credentials, deployment addresses, build caches or node bindings.
    for name,raw in data.items():
        assert not any(s in name for s in ['.git','tests','config/','native-env'])
        assert not re.search(rb'(?:/home/iterator|/public-flash/liyan|10\.0\.0\.\d+|910b-\d+|BEGIN (?:OPENSSH|RSA) PRIVATE KEY)',raw),name
    assert not args.output.exists();args.output.mkdir(parents=True)
    for name,raw in data.items():
        path=args.output/'release'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    metadata=dict(schema=1,version=args.tag,sglang_commit=commit,base_image=BASE,base_image_id=BASE_ID,
        base_manifests={'prefill':'6714a5319c9854a088e9f390b1e520b9e3ffeb7ec1cbbb9f29c5807da6f8b74a',
                        'decode':'16fcd72a0068301d32d5a8f5454150388e6f8b770b602c303876f44d98b98f8b'},
        author='liyan <yan.l@pku.edu.cn>',private_runtime_configuration_excluded=True,
        files={name:dict(source=EXPORTS[name],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)) for name,raw in data.items()})
    (args.output/'release/release.json').write_text(json.dumps(metadata,indent=2)+'\n')
    dockerfile=f'''FROM {BASE}
COPY release/ /opt/flagos-express/release-audit/
COPY overlay/ /
LABEL org.opencontainers.image.version="{args.tag}" \\
      org.opencontainers.image.revision="{commit}" \\
      org.flagos.base-image="{BASE_ID}" \\
      org.flagos.prefill.topology="DP4TP8-EP32" \\
      org.flagos.decode.topology="DP8TP2-EP16"
'''
    (args.output/'Dockerfile').write_text(dockerfile)
    (args.output/'Dockerfile.source').write_text(f'FROM {BASE}\nCOPY release/ /opt/flagos-express/release-audit/\n')
    (args.output/'.dockerignore').write_text('*\n!Dockerfile\n!Dockerfile.source\n!release/\n!release/**\n!overlay/\n!overlay/**\n')
    print(json.dumps(dict(context=str(args.output),commit=commit,version=args.tag,files=len(data),
        context_bytes=sum(len(x) for x in data.values()),base_image_id=BASE_ID)))


if __name__=='__main__':main()
