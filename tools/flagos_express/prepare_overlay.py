#!/usr/bin/env python3
"""Prepare verified source files in a device-free container; copy only allowlist.

The final image inherits the clean base config and uses COPY only. Builder
container environment, hostname, mounts and writable runtime state never enter
the result. This also works on daemons whose legacy RUN seccomp profile fails.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

COMMAND = '''set -e
python3 -B /opt/flagos-express/release-audit/audit.py base
patch --fuzz=0 --no-backup-if-mismatch -p1 -d /sgl-workspace/roles/prefill < /opt/flagos-express/release-audit/prefill-topology.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d /sgl-workspace/roles/prefill/python < /opt/flagos-express/release-audit/runtime-prefill-rank.patch
patch --fuzz=0 --no-backup-if-mismatch -p1 -d /sgl-workspace/roles/decode/python < /opt/flagos-express/release-audit/runtime-decode-rank.patch
cp /opt/flagos-express/release-audit/runtime/sglang_pd_monitoring.py /opt/flagos-express/deps/monitoring/sglang_pd_monitoring.py
mkdir -p /opt/flagos-express/deps/monitoring/exporter
cp /opt/flagos-express/release-audit/runtime/exporter/exporter.py /opt/flagos-express/deps/monitoring/exporter/exporter.py
python3 -B /opt/flagos-express/release-audit/audit.py final
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=Path,required=True)
    parser.add_argument('--cpus',default='184,185')
    args=parser.parse_args();root=args.context.resolve()
    release=json.loads((root/'release/release.json').read_text())
    original=json.loads(subprocess.check_output(['docker','image','inspect',release['base_image']]))[0]
    assert original['Id']==release['base_image_id']
    assert not (root/'overlay').exists(),'never overwrite a prepared overlay'
    suffix=release['sglang_commit'][:12]
    image='glm53-public-source-stage:'+suffix
    name='glm53-public-source-stage-'+suffix
    env=dict(os.environ,DOCKER_BUILDKIT='0')
    with (root/'source-build.log').open('xb') as log:
        subprocess.run(['docker','build','--pull=false','--network=none','-f',str(root/'Dockerfile.source'),'-t',image,str(root)],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    create=['docker','create','--name',name,'--runtime','runc','--network','none',
        '--security-opt','seccomp=unconfined','--cpuset-cpus',args.cpus,'--memory','8g','--pids-limit','128',
        '--env','TORCH_DEVICE_BACKEND_AUTOLOAD=0','--env','ASCEND_RT_VISIBLE_DEVICES=',
        '--env','CUDA_VISIBLE_DEVICES=','--entrypoint','/bin/sh',image,'-ec',COMMAND]
    cid=subprocess.check_output(create,text=True).strip()
    with (root/'source-audit.log').open('xb') as log:
        subprocess.run(['docker','start','-a',cid],stdout=log,stderr=subprocess.STDOUT,check=True)
    state=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']
    assert state['ExitCode']==0 and not state['Running'],state
    paths=[record['target'] for record in json.loads((root/'release/topology.json').read_text())['files']]
    paths+=['/sgl-workspace/roles/'+role+'/python/sglang/srt/disaggregation/decode.py' for role in ('prefill','decode')]
    paths+=['/opt/flagos-express/deps/monitoring/sglang_pd_monitoring.py',
            '/opt/flagos-express/deps/monitoring/exporter/exporter.py','/opt/flagos-express/entrypoint',
            '/opt/flagos-express/versions.json','/opt/flagos-express/release-audit/validated.json']
    paths+=['/opt/flagos-express/manifests/'+role+'-python.json' for role in ('prefill','decode')]
    assert len(paths)==len(set(paths))
    inventory={}
    for name in paths:
        raw=subprocess.check_output(['docker','cp',cid+':'+name,'-'])
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            files=[m for m in tf if m.isfile()]
            assert len(files)==1 and not any(m.issym() or m.islnk() for m in tf.getmembers())
            member=files[0];data=tf.extractfile(member).read()
        target=root/'overlay'/name.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data);target.chmod(member.mode & 0o777)
        inventory[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),mode=oct(member.mode & 0o777))
    proof=json.loads((root/'overlay/opt/flagos-express/release-audit/validated.json').read_text())
    assert proof['sglang_commit']==release['sglang_commit']
    receipt=dict(commit=release['sglang_commit'],builder_container=cid,device_free=True,
        network_none=True,full_role_source_validation=True,only_allowlisted_files_copied=True,files=inventory)
    (root/'overlay-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(commit=receipt['commit'],files=len(inventory),builder_container=cid,validated=True)))


if __name__=='__main__':main()
