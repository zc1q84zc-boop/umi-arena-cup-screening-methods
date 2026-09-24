const $ = id => document.getElementById(id);
const colors = ['#55d0d1', '#f2a06d'];
const videos = ['left', 'right', 'center'].map(view => $(`${view}-video`));
const state = {catalog: [], visible: [], data: null, frame: 0, playing: false, token: 0};

function visibleEpisodes() {
  const filter = $('filter').value;
  return state.catalog.filter(item => filter === 'all' ||
    (filter === 'human' && item.human_status === 'confirmed_anomaly') ||
    (filter === 'machine' && item.flags.length));
}

function refreshChoices(preferred) {
  state.visible = visibleEpisodes();
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.disabled = true;
  placeholder.textContent = '请选择 episode（不会预下载）';
  $('episode').replaceChildren(placeholder, ...state.visible.map(item => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.id} · ${item.frames} 帧 · ${item.human_status === 'confirmed_anomaly' ? '人工标记' : `${item.flags.length} 项机器标记`}`;
    return option;
  }));
  const valid = preferred && state.visible.some(item => item.id === Number(preferred));
  $('episode').value = valid ? preferred : '';
  if (!valid) {
    stopPlayback(); state.data = null;
    videos.forEach(video => { video.removeAttribute('src'); video.load(); });
    $('loading').textContent = '请选择一条 episode 后按需读取。';
  }
}

function motionAt(frame, hand, action = false) {
  if (!state.data || frame < 0 || frame >= state.data.frames) return 0;
  if (action) return Math.hypot(...state.data.action_poses[frame][hand].slice(0, 3));
  if (!frame) return 0;
  const a = state.data.poses[frame][hand], b = state.data.poses[frame - 1][hand];
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function seek(frame, moveVideo = true) {
  const data = state.data;
  if (!data) return;
  const i = Math.max(0, Math.min(data.frames - 1, Math.round(frame)));
  state.frame = i;
  $('timeline').value = i;
  $('frame-label').textContent = `frame ${i} / ${data.frames - 1}`;
  $('time').textContent = `${data.times[i].toFixed(2)} s`;
  for (const hand of [0, 1]) {
    const side = hand ? 'right' : 'left';
    $(`${side}-gripper`).textContent = `${data.grippers[i][hand].toFixed(3)} rad · 动作 ${data.action_grippers[i][hand].toFixed(3)}`;
    $(`${side}-motion`).textContent = `${motionAt(i, hand).toFixed(4)} m · 动作 ${motionAt(i, hand, true).toFixed(4)}`;
  }
  if (moveVideo) for (const video of videos) {
    if (video.readyState && Math.abs(video.currentTime - data.times[i]) > .025)
      video.currentTime = data.times[i];
  }
  document.querySelectorAll('#event-list button').forEach(button => {
    button.classList.toggle('active', Number(button.dataset.frame) === i);
  });
  drawCharts();
}

function stopPlayback() {
  state.playing = false;
  $('play').textContent = '▶';
  videos.forEach(video => video.pause());
}

function paintChart(canvas, series, unit) {
  const data = state.data;
  const box = canvas.getBoundingClientRect();
  if (!data || !box.width) return;
  const dpr = Math.min(devicePixelRatio, 2), w = box.width, h = box.height;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const p = {l: 43, r: 12, t: 15, b: 22};
  const innerW = w - p.l - p.r, innerH = h - p.t - p.b;
  const values = series.flatMap(s => s.values).filter(Number.isFinite);
  const low = Math.min(0, ...values), high = Math.max(...values, low + .001);
  const range = high - low;
  const x = i => p.l + innerW * i / Math.max(1, data.frames - 1);
  const y = v => p.t + innerH * (high - v) / range;
  ctx.fillStyle = '#83a0aa'; ctx.font = '10px ui-monospace,monospace';
  for (let n = 0; n <= 2; n++) {
    const yy = p.t + n * innerH / 2;
    ctx.strokeStyle = '#2c424c'; ctx.beginPath(); ctx.moveTo(p.l, yy); ctx.lineTo(w - p.r, yy); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText((high - n * range / 2).toFixed(unit === 'rad' ? 2 : 3), p.l - 5, yy + 3);
  }
  for (const event of data.events) {
    ctx.strokeStyle = '#a36265'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(event.frame_index), p.t); ctx.lineTo(x(event.frame_index), p.t + 7); ctx.stroke();
  }
  for (const item of series) {
    ctx.strokeStyle = colors[item.hand]; ctx.lineWidth = item.action ? 1.3 : 1.9;
    ctx.setLineDash(item.action ? [4, 3] : []); ctx.beginPath();
    item.values.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = '#ecf5ef'; ctx.beginPath(); ctx.moveTo(x(state.frame), p.t); ctx.lineTo(x(state.frame), p.t + innerH); ctx.stroke();
  ctx.fillStyle = '#83a0aa'; ctx.textAlign = 'left'; ctx.fillText('0 s', p.l, h - 3);
  ctx.textAlign = 'right'; ctx.fillText(`${(data.frames / data.fps).toFixed(1)} s`, w - p.r, h - 3);
}

function drawCharts() {
  const data = state.data;
  if (!data) return;
  const gripper = [], motion = [];
  for (const hand of [0, 1]) {
    gripper.push({hand, action: false, values: data.grippers.map(row => row[hand])});
    gripper.push({hand, action: true, values: data.action_grippers.map(row => row[hand])});
    motion.push({hand, action: false, values: data.poses.map((_, i) => motionAt(i, hand))});
    motion.push({hand, action: true, values: data.action_poses.map((_, i) => motionAt(i, hand, true))});
  }
  paintChart($('gripper-chart'), gripper, 'rad');
  paintChart($('motion-chart'), motion, 'm');
}

function renderEvents(data) {
  const list = $('event-list'); list.replaceChildren();
  const events = [
    ...data.human_frames.map(frame_index => ({frame_index, kind: '人工标记动作跳变'})),
    ...data.events
  ].sort((a, b) => a.frame_index - b.frame_index);
  $('event-count').textContent = `${events.length} 项`;
  for (const event of events) {
    const button = document.createElement('button');
    button.type = 'button'; button.dataset.frame = event.frame_index;
    button.textContent = `帧 ${event.frame_index} · ${event.kind}`;
    button.addEventListener('click', () => { stopPlayback(); seek(event.frame_index); });
    list.append(button);
  }
}

async function loadEpisode(id) {
  const token = ++state.token;
  stopPlayback(); $('loading').textContent = `正在从 A100 读取 episode ${id}（已有缓存时直接复用）…`;
  try {
    const response = await fetch(`/api/episode?id=${encodeURIComponent(id)}`, {cache: 'no-store'});
    if (!response.ok) throw Error(`轨迹读取失败：HTTP ${response.status}`);
    const data = await response.json();
    if (token !== state.token) return;
    state.data = data; $('timeline').max = data.frames - 1;
    $('episode-title').textContent = `Episode ${data.id}`;
    $('decision').textContent = `${data.frames} 帧 · ${data.human_status === 'confirmed_anomaly' ? '有人工作出的动作跳变标记' : '机器规则候选，待人工复核'}`;
    $('flags').replaceChildren(...data.flags.map(flag => { const tag = document.createElement('span'); tag.textContent = flag; return tag; }));
    renderEvents(data);
    for (const [index, view] of ['left', 'right', 'center'].entries()) {
      videos[index].src = `/video/${data.id}/${view}.mp4`;
      videos[index].load();
    }
    seek(0, false);
    $('cache-status').textContent = `只缓存 episode ${data.id} 的轨迹和请求播放的机位视频；不会下载其它 episode。`;
    $('loading').textContent = '轨迹已加载；视频首次打开时按需传输。';
  } catch (error) {
    if (token === state.token) $('loading').textContent = error.message;
  }
}

$('filter').addEventListener('change', () => refreshChoices($('episode').value));
$('episode').addEventListener('change', event => loadEpisode(event.target.value));
for (const [id, offset] of [['previous', -1], ['next', 1]]) {
  $(id).addEventListener('click', () => {
    const index = state.visible.findIndex(item => item.id === Number($('episode').value));
    const next = state.visible[index < 0 ? (offset > 0 ? 0 : -1) : index + offset];
    if (next) { $('episode').value = next.id; loadEpisode(next.id); }
  });
}
$('timeline').addEventListener('input', event => { stopPlayback(); seek(Number(event.target.value)); });
$('play').addEventListener('click', async () => {
  if (!state.data) return;
  if (state.playing) { stopPlayback(); return; }
  if (state.frame >= state.data.frames - 1) seek(0);
  state.playing = true; $('play').textContent = 'Ⅱ';
  try { await Promise.all(videos.map(video => video.play())); }
  catch (_error) { stopPlayback(); $('loading').textContent = '视频未能播放，请检查 A100 连接。'; }
});
videos[2].addEventListener('timeupdate', () => {
  if (!state.playing || !state.data) return;
  const time = videos[2].currentTime;
  seek(Math.round(time * state.data.fps), false);
  for (const video of videos.slice(0, 2)) if (Math.abs(video.currentTime - time) > .08) video.currentTime = time;
});
videos[2].addEventListener('ended', stopPlayback);
new ResizeObserver(drawCharts).observe($('gripper-chart'));

fetch('/api/catalog', {cache: 'no-store'}).then(response => {
  if (!response.ok) throw Error(`目录读取失败：HTTP ${response.status}`);
  return response.json();
}).then(catalog => {
  state.catalog = catalog.episodes;
  $('connection').textContent = `A100 已连接 · ${state.catalog.length} episodes / ${catalog.total_frames} 帧`;
  refreshChoices(null);
}).catch(error => { $('connection').textContent = 'A100 连接失败'; $('loading').textContent = error.message; });
