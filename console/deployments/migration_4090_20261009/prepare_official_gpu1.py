from pathlib import Path
import hashlib
import socket
assert socket.gethostname() == 'benyun-workstation'
p=Path('/home/claude/workspace/umi_cup_models_4090_20261009/lingbot/run_lingbot_official.sh')
s=p.read_text()
assert s.count('export CUDA_VISIBLE_DEVICES=0') == 1
backup=p.with_name(p.name+'.before_gpu1_20261009')
if not backup.exists(): backup.write_text(s)
p.write_text(s.replace('export CUDA_VISIBLE_DEVICES=0','export CUDA_VISIBLE_DEVICES=1'))
u=Path('/home/claude/.config/systemd/user/lingbot-official-squirrel.service')
assert not u.exists()
u.write_text('[Unit]\nDescription=Official unchanged LingBot VLA2 on dual4090 GPU1\n\n[Service]\nType=simple\nExecStart=/bin/bash /home/claude/workspace/umi_cup_models_4090_20261009/lingbot/run_lingbot_official.sh\nRestart=no\nWorkingDirectory=/home/claude/workspace/umi_cup_models_4090_20261009\n')
print('OFFICIAL_GPU1_UNIT_PREPARED',hashlib.sha256(p.read_bytes()).hexdigest())
