"""Export a read-only prefix of our open H.264 MP4 without stopping a trial.

An open MP4 lacks its final moov atom. Use parameter sets from a verified
recording made by the same writer, decode the existing NAL units, and omit
eight tail frames to exclude any incomplete/reordered tail. No new scene
frames are generated. The full original remains untouched.
"""
import argparse,hashlib,json,re,struct,subprocess
from pathlib import Path

def parameter_sets(template):
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(template),'-map','0:v:0',
        '-c:v','copy','-frames:v','1','-bsf:v','h264_mp4toannexb','-f','h264','pipe:1'])
    nals=re.split(rb'\x00\x00\x00?\x01',raw)
    found=[n for n in nals if n and (n[0]&31) in (7,8)]
    assert {n[0]&31 for n in found}=={7,8}
    return b''.join(b'\x00\x00\x00\x01'+n for n in found)

def extract(source,template,annexb):
    size=source.stat().st_size;copied=nals=0
    with source.open('rb') as stream,annexb.open('wb') as out:
        out.write(parameter_sets(template))
        offset=0
        while offset+8<=size:
            stream.seek(offset);length,kind=struct.unpack('>I4s',stream.read(8));header=8
            if length==1:length=struct.unpack('>Q',stream.read(8))[0];header=16
            if length==0:length=size-offset
            assert length>=header
            if kind==b'mdat':
                end=min(offset+length,size)
                while stream.tell()+4<=end:
                    count=struct.unpack('>I',stream.read(4))[0]
                    if not count or count>8*1024*1024 or stream.tell()+count>end:break
                    nal=stream.read(count);typ=nal[0]&31
                    assert typ in (1,5,6,7,8,9,12),f'Unexpected H.264 NAL type {typ}'
                    out.write(b'\x00\x00\x00\x01'+nal);copied+=count+4;nals+=1
                break
            offset+=length
    assert nals>1 and copied>100
    return {'source_snapshot_bytes':size,'complete_nal_bytes':copied,'complete_nal_units':nals,
        'parameter_sets_template':str(template),'parameter_sets_sha256':hashlib.sha256(parameter_sets(template)).hexdigest()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path);parser.add_argument('template',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--extract-only',action='store_true')
    parser.add_argument('--max-frames',type=int)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    annexb=args.output.with_suffix('.h264');audit=extract(args.source,args.template,annexb)
    if args.extract_only:print(json.dumps(audit));return
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-f','h264','-count_frames',
        '-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(annexb)],text=True))['streams'][0]
    assert (info['width'],info['height'])==(640,480)
    count=int(info['nb_read_frames'])-8
    if args.max_frames is not None:count=min(count,args.max_frames)
    assert count>0
    subprocess.run(['ffmpeg','-v','error','-y','-fflags','+genpts','-f','h264','-i',str(annexb),
        '-map','0:v:0','-an','-vf','setpts=N/(30*TB)','-frames:v',str(count),'-r','30',
        '-c:v','libx264','-preset','veryfast','-crf','24','-threads','2','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(args.output)],check=True)
    check=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,duration,r_frame_rate','-of','json',str(args.output)],text=True))['streams'][0]
    assert int(check['nb_frames'])==count and check['r_frame_rate']=='30/1'
    audit.update(recording_scope='in_progress_prefix_not_final_task_recording',frames=count,
        duration_s=count/30,discarded_tail_frames=8,source_recording_modified=False,
        synthetic_frames_added=False,output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(audit,indent=2)+'\n')
    annexb.unlink()
    print(json.dumps({k:audit[k] for k in ('recording_scope','frames','duration_s','synthetic_frames_added')}))

if __name__=='__main__':main()
