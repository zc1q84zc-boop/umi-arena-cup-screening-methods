"""Remux exactly three full-duration recordings for phone playback."""
import json
from pathlib import Path
import struct
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
verification = []
for view, source in (('head', 'video.mp4'), ('left_wrist', 'video_left_wrist.mp4'), ('right_wrist', 'video_right_wrist.mp4')):
    destination = root/f'{view}_full_mobile.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(root/source), '-map', '0:v:0', '-c', 'copy', '-movflags', '+faststart', str(destination)], check=True)
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames:format=duration,size',
        '-of', 'json', str(destination)]))
    stream = metadata['streams'][0]
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    atoms = []
    with destination.open('rb') as handle:
        while handle.tell() < destination.stat().st_size:
            offset = handle.tell()
            size, kind = struct.unpack('>I4s', handle.read(8))
            atoms.append(kind.decode())
            if size < 8:
                break
            handle.seek(offset+size)
    assert atoms.index('moov') < atoms.index('mdat')
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-sseof', '-1',
        '-i', str(destination), '-frames:v', '1', '-f', 'null', '-'], check=True)
    verification.append({'view': view, 'file': destination.name, 'metadata': metadata,
        'faststart': True, 'final_frame_decode_passed': True})
    print(json.dumps({'file': destination.name, **metadata['format']}), flush=True)
assert len({x['metadata']['streams'][0]['nb_frames'] for x in verification}) == 1
assert max(float(x['metadata']['format']['duration']) for x in verification) - min(float(x['metadata']['format']['duration']) for x in verification) < .05
(root/'mobile_media_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
