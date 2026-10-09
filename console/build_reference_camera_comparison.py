"""Build same-episode camera contact sheets; no production changes."""
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'sim_validation/reference259632_camera_motion_a'
for side in ['left','right']:
    canvas=Image.new('RGB',(960,6*380),'#17202b')
    draw=ImageDraw.Draw(canvas)
    for row,frame in enumerate([0,15,30,45,60,75]):
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(ROOT/f'videos/episode-259632-{side}.mp4'),
            '-vf',f'select=eq(n\\,{frame})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
        source=Image.frombytes('RGB',(640,480),raw)
        sim=Image.open(RUN/f'{side}_wrist_source_{frame:03d}.png')
        for col,img in enumerate([source,sim]):
            canvas.paste(img.resize((480,360)),(480*col,row*380+20))
        draw.text((5,row*380+3),f'{side} source frame {frame} | real (left) / Isaac candidate (right)',fill='white')
    canvas.save(RUN/f'{side}_comparison.jpg')

# An observation at policy step k is the recorded state at video frame 3*k.
# These samples follow the preceding source target (3*(k-1)), not source time.
canvas=Image.new('RGB',(1440,6*385),'#17202b')
draw=ImageDraw.Draw(canvas)
for row,frame in enumerate([0,15,30,45,60,75]):
    images=[]
    for path,index,size in [(ROOT/'videos/episode-259632-center.mp4',frame,(640,480)),
                            (RUN/'video.mp4',frame+3,(960,540))]:
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vf',
            f'select=eq(n\\,{index})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
        images.append(Image.frombytes('RGB',size,raw))
    panels=[images[0],Image.open(RUN/f'head_source_{frame:03d}.png'),images[1]]
    labels=[f'Real chest: source {frame}',f'Isaac head: target {frame}',
            f'Isaac overview: video {frame+3}, target {frame}']
    for col,(img,title) in enumerate(zip(panels,labels)):
        img.thumbnail((480,360))
        canvas.paste(img,(480*col+(480-img.width)//2,row*385+25+(360-img.height)//2))
        draw.text((480*col+5,row*385+4),title,fill='white')
canvas.save(RUN/'overview_comparison.jpg')
