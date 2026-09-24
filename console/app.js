import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const colors = {left: '#55d0d1', right: '#f2a06d'};
const state = {catalog: null, episode: null, prediction: null, checkpoint: null, frame: 0, playing: false,
  timelineMode: 'whole', chunkIndex: 0, primaryView: 2,
  signals: null, videoReady: false, videoLoadToken: 0, syncToken: 0};
const videos = [$('leftVideo'), $('rightVideo'), $('centerVideo')];
const frameCanvases = [$('leftFrame'), $('rightFrame'), $('centerFrame')];
const videoTiles = [$('leftTile'), $('rightTile'), $('centerTile')];
const sampleCanvas = document.createElement('canvas');
sampleCanvas.width = 12; sampleCanvas.height = 9;
const sampleContext = sampleCanvas.getContext('2d', {willReadFrequently: true});

function paintVideo(view) {
  const video = videos[view];
  if (video.readyState < 2 || !video.videoWidth) return;
  try {
    sampleContext.drawImage(video, 0, 0, 12, 9);
    const pixels = sampleContext.getImageData(0, 0, 12, 9).data;
    let brightness = 0;
    for (let i = 0; i < pixels.length; i += 4) brightness += pixels[i] + pixels[i + 1] + pixels[i + 2];
    if (brightness / (pixels.length * .75) < 8 && videoTiles[view].classList.contains('has-frame')) return;
    const canvas = frameCanvases[view];
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }
    canvas.getContext('2d').drawImage(video, 0, 0);
    videoTiles[view].classList.add('has-frame');
  } catch (error) {
    // The existing rendered frame remains visible while a seek completes.
  }
}
function setPrimaryView(view) {
  const smaller = videoTiles.map((_, index) => index).filter(index => index !== view);
  videoTiles.forEach((tile, index) => {
    tile.classList.toggle('primary', index === view);
    tile.classList.toggle('secondary', index === smaller[0]);
    tile.classList.toggle('tertiary', index === smaller[1]);
  });
  state.primaryView = view;
}
videos.forEach((video, view) => {
  video.addEventListener('loadeddata', () => paintVideo(view));
  video.addEventListener('seeked', () => paintVideo(view));
});
const scene = new THREE.Scene();
scene.background = new THREE.Color('#101b23');
const camera = new THREE.PerspectiveCamera(44, 1, .01, 100);
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
$('scene').prepend(renderer.domElement);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = .08;
orbit.screenSpacePanning = true;
const grid = new THREE.GridHelper(1, 10, 0x38505a, 0x29404b);
scene.add(grid);
const axes = new THREE.AxesHelper(.15);
scene.add(axes);
const motionGroup = new THREE.Group();
scene.add(motionGroup);
let home = {target: new THREE.Vector3(), position: new THREE.Vector3(.65, .55, .7)};
let actualGrippers = [];
let predictedGrippers = [];
let traversed = [];
let actualTipTrails = [];
let predictedTipTrails = [];
let predictedFrames = new Map();
const emptyTrajectoryLayers = () => ({actualRoot: [], actualTips: [], predictedRoot: [], predictedTips: []});
let trajectoryLayers = emptyTrajectoryLayers();

function vec(p) { return new THREE.Vector3(p[0], p[2], p[1]); }
const GRIPPER_DISPLAY_SCALE = 1.7;
// The display swaps source Y and Z. A quaternion's vector part also changes sign
// under this handedness-changing axis swap.
function poseRotation(p) { return new THREE.Quaternion(-p[3], -p[5], -p[4], p[6]).normalize(); }
// The replay exposes joint angle, but not calibrated fingertip spacing. The gap
// below is a deliberately enlarged visual mapping; the HUD retains the raw radians.
function visualGap(angle) { return .008 + THREE.MathUtils.clamp(angle / .8, 0, 1) * .048; }
function tipAt(pose, angle, side) {
  return new THREE.Vector3(side * visualGap(angle) / 2, 0, .059).multiplyScalar(GRIPPER_DISPLAY_SCALE)
    .applyQuaternion(poseRotation(pose)).add(vec(pose));
}
function makeGripper(color, ghost = false) {
  const group = new THREE.Group();
  group.scale.setScalar(GRIPPER_DISPLAY_SCALE);
  const material = new THREE.MeshBasicMaterial({color, transparent: ghost, opacity: ghost ? .76 : 1,
    wireframe: ghost, depthWrite: !ghost});
  const tipMaterial = new THREE.MeshBasicMaterial({color: ghost ? '#ffffff' : color,
    transparent: ghost, opacity: ghost ? .86 : 1, depthWrite: !ghost});
  const body = new THREE.Mesh(new THREE.BoxGeometry(.034, .018, .022), material);
  body.position.z = .006;
  group.add(body);
  const jaws = [-1, 1].map(() => {
    const pivot = new THREE.Group();
    const finger = new THREE.Mesh(new THREE.BoxGeometry(.007, .009, .043), material);
    finger.position.z = .034;
    const tip = new THREE.Mesh(new THREE.SphereGeometry(.005, 10, 8), tipMaterial);
    tip.position.z = .059;
    pivot.add(finger, tip); group.add(pivot);
    return pivot;
  });
  motionGroup.add(group);
  return {group, jaws};
}
function positionGripper(visual, pose, angle) {
  visual.group.position.copy(vec(pose));
  visual.group.quaternion.copy(poseRotation(pose));
  const gap = visualGap(angle);
  visual.jaws[0].position.x = -gap / 2;
  visual.jaws[1].position.x = gap / 2;
  visual.group.visible = true;
}
function line(points, color, opacity, dashed = false, layer = null) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = dashed ? new THREE.LineDashedMaterial({color, transparent: true, opacity, dashSize: .012, gapSize: .008}) :
    new THREE.LineBasicMaterial({color, transparent: true, opacity});
  const object = new THREE.Line(geometry, material);
  if (dashed) object.computeLineDistances();
  motionGroup.add(object);
  if (layer) trajectoryLayers[layer].push(object);
  return object;
}
function updateTrajectoryVisibility() {
  for (const [layer, objects] of Object.entries(trajectoryLayers)) {
    const visible = $(`${layer}Toggle`).checked;
    for (const object of objects) object.visible = visible;
  }
}
function clearScene() {
  for (const child of [...motionGroup.children]) {
    motionGroup.remove(child);
    child.traverse(object => {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
      else object.material?.dispose();
    });
  }
  actualGrippers = []; predictedGrippers = []; traversed = [];
  actualTipTrails = []; predictedTipTrails = []; predictedFrames = new Map();
  trajectoryLayers = emptyTrajectoryLayers();
}
function selectedWindow() {
  return state.timelineMode === 'chunk' && state.prediction?.available ?
    state.prediction.windows[state.chunkIndex] : null;
}
function frameRange() {
  const window = selectedWindow();
  return window?.indices?.length ?
    [window.indices[0], window.indices.at(-1)] : [0, (state.episode?.frames ?? 1) - 1];
}
function selectChunk(index) {
  if (!state.prediction?.available) return;
  stopPlayback();
  state.chunkIndex = Math.max(0, Math.min(state.prediction.windows.length - 1, index));
  $('chunkSelect').value = String(state.chunkIndex);
  const [start, end] = frameRange();
  $('scrubber').min = start; $('scrubber').max = end;
  $('previousChunk').disabled = state.chunkIndex === 0;
  $('nextChunk').disabled = state.chunkIndex === state.prediction.windows.length - 1;
  $('duration').textContent = `${state.episode.times[end].toFixed(2)} s`;
  const window = selectedWindow();
  const avg = hand => {
    const values = window.position_cm?.map(row => row[hand]).filter(Number.isFinite) ?? [];
    return values.length ? `${(values.reduce((a, b) => a + b, 0) / values.length).toFixed(2)} cm` : '—';
  };
  const gripError = hand => {
    const values = window.indices.map((frame, i) => Math.abs(window.grippers[i][hand] - state.episode.grippers[frame][hand]));
    return `${(values.reduce((a, b) => a + b, 0) / values.length).toFixed(3)} rad`;
  };
  $('chunkMetrics').textContent = `Chunk ${state.chunkIndex + 1}/${state.prediction.windows.length} · 帧 ${start + 1}–${end + 1} · 平均末端位置误差：左 ${avg(0)} / 右 ${avg(1)} · 平均夹爪角误差：左 ${gripError(0)} / 右 ${gripError(1)}`;
  buildScene(); setFrame(start);
}
function updateTimelineMode() {
  stopPlayback();
  if (!state.episode) return;
  if (state.timelineMode === 'chunk' && !state.prediction?.available) state.timelineMode = 'whole';
  $('timelineMode').value = state.timelineMode;
  $('chunkPicker').hidden = state.timelineMode !== 'chunk';
  $('chunkMetrics').hidden = state.timelineMode !== 'chunk';
  if (state.timelineMode === 'chunk') selectChunk(state.chunkIndex);
  else {
    $('scrubber').min = 0; $('scrubber').max = state.episode.frames - 1;
    $('duration').textContent = `${state.episode.duration_s.toFixed(2)} s`;
    buildScene(); setFrame(state.frame);
  }
}
function buildScene() {
  clearScene();
  const data = state.episode;
  if (!data) return;
  const all = [];
  const [start, end] = frameRange();
  for (let hand = 0; hand < 2; hand++) {
    const color = hand === 0 ? colors.left : colors.right;
    const points = data.poses.slice(start, end + 1).map(row => vec(row[hand]));
    all.push(...points);
    line(points, color, .55, false, 'actualRoot');
    const active = line(points, color, 1, false, 'actualRoot');
    active.geometry.setDrawRange(0, 1);
    traversed.push(active);
    for (const side of [-1, 1]) {
      const tips = data.poses.slice(start, end + 1).map((row, i) => tipAt(row[hand], data.grippers[start + i][hand], side));
      line(tips, color, .18, false, 'actualTips');
      const trail = line(tips, color, .65, false, 'actualTips');
      trail.geometry.setDrawRange(0, 1);
      actualTipTrails.push(trail);
    }
    actualGrippers.push(makeGripper(color));
    predictedGrippers.push(makeGripper(color, true));
    predictedGrippers[hand].group.visible = false;
  }
  if (state.prediction?.available) {
    for (const window of selectedWindow() ? [selectedWindow()] : state.prediction.windows) {
      window.indices.forEach((frame, k) => {
        if (window.poses[k] && window.grippers[k])
          predictedFrames.set(frame, {poses: window.poses[k], grippers: window.grippers[k]});
      });
      for (let hand = 0; hand < 2; hand++) {
        const points = window.poses.map(row => vec(row[hand]));
        all.push(...points);
        if (points.length) line(points, hand === 0 ? colors.left : colors.right, .9, true, 'predictedRoot');
        for (const side of [-1, 1]) {
          const tips = window.poses.map((row, k) => tipAt(row[hand], window.grippers[k][hand], side));
          line(tips, hand === 0 ? colors.left : colors.right, .24, true, 'predictedTips');
          const trail = line(tips, hand === 0 ? colors.left : colors.right, .82, true, 'predictedTips');
          trail.geometry.setDrawRange(0, 0);
          predictedTipTrails.push({indices: window.indices, trail});
        }
      }
    }
  }
  updateTrajectoryVisibility();
  const bounds = new THREE.Box3().setFromPoints(all);
  const center = bounds.getCenter(new THREE.Vector3());
  const span = Math.max(selectedWindow() ? .10 : .2, bounds.getSize(new THREE.Vector3()).length());
  grid.position.set(center.x, bounds.min.y - .015, center.z);
  grid.scale.setScalar(Math.max(.5, span));
  axes.position.set(center.x - span * .36, bounds.min.y, center.z - span * .36);
  axes.scale.setScalar(Math.max(.6, span));
  home = {target: center, position: center.clone().add(new THREE.Vector3(span * 1.15, span * .95, span * 1.35))};
  resetCamera();
}
function resetCamera() { orbit.target.copy(home.target); camera.position.copy(home.position); camera.near = .001; camera.far = 100; camera.updateProjectionMatrix(); orbit.update(); }
function fitCamera() { resetCamera(); }
function resizeScene() {
  const element = $('scene');
  const w = element.clientWidth, h = element.clientHeight;
  if (w && h) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
}
new ResizeObserver(resizeScene).observe($('scene'));
function renderLoop() { requestAnimationFrame(renderLoop); orbit.update(); renderer.render(scene, camera); }
renderLoop();

function deltaAngle(a, b) {
  const qa = a.slice(3), qb = b.slice(3);
  const na = Math.hypot(...qa), nb = Math.hypot(...qb);
  const dot = Math.min(1, Math.abs(qa.reduce((sum, x, i) => sum + x * qb[i], 0) / (na * nb)));
  return 2 * Math.acos(dot) * 180 / Math.PI;
}
function signalsFor(data) {
  const n = data.frames, linear = [[], []], angular = [[], []], joint = [[], []], error = [[], []];
  const modelLinear = [Array(n).fill(null), Array(n).fill(null)];
  const modelAngular = [Array(n).fill(null), Array(n).fill(null)];
  const modelJoint = [Array(n).fill(null), Array(n).fill(null)];
  const modelWindowStarts = new Set();
  for (let i = 0; i < n; i++) for (let hand = 0; hand < 2; hand++) {
    const dt = i ? data.times[i] - data.times[i - 1] : 1 / data.fps;
    const current = data.poses[i][hand];
    const prior = i ? data.poses[i - 1][hand] : current;
    linear[hand].push(Math.hypot(...current.slice(0, 3).map((v, axis) => v - prior[axis])) / dt);
    angular[hand].push(deltaAngle(prior, current) / dt);
    joint[hand].push(data.grippers[i][hand]);
    error[hand].push(null);
  }
  if (state.prediction?.available) for (const window of state.prediction.windows) {
    if (window.indices.length) modelWindowStarts.add(window.indices[0]);
    window.indices.forEach((frame, i) => {
      if (frame >= n) return;
      for (let hand = 0; hand < 2; hand++) {
        error[hand][frame] = window.position_cm?.[i]?.[hand] ?? null;
        modelJoint[hand][frame] = window.grippers?.[i]?.[hand] ?? null;
        if (i > 0) {
          const current = window.poses[i][hand], prior = window.poses[i - 1][hand];
          const dt = data.times[frame] - data.times[window.indices[i - 1]];
          modelLinear[hand][frame] = Math.hypot(...current.slice(0, 3).map((v, axis) => v - prior[axis])) / dt;
          modelAngular[hand][frame] = deltaAngle(prior, current) / dt;
        }
      }
    });
  }
  return {linear, angular, joint, error, modelLinear, modelAngular, modelJoint, modelWindowStarts};
}
function setFrame(frame, seekVideos = true) {
  const data = state.episode;
  if (!data) return;
  const [start, end] = frameRange();
  state.frame = Math.max(start, Math.min(end, Math.round(frame)));
  const i = state.frame;
  if (seekVideos && state.videoReady) for (const video of videos)
    if (Math.abs(video.currentTime - data.times[i]) > .008) video.currentTime = data.times[i];
  $('scrubber').value = i;
  $('currentTime').textContent = `${data.times[i].toFixed(2)} s`;
  $('frameLabel').textContent = `${String(i + 1).padStart(4, '0')} / ${data.frames}`;
  const predicted = predictedFrames.get(i);
  $('gripperSummary').textContent = `遥操 左 ${data.grippers[i][0].toFixed(2)} / 右 ${data.grippers[i][1].toFixed(2)} rad` +
    (predicted ? ` · 预测 左 ${predicted.grippers[0].toFixed(2)} / 右 ${predicted.grippers[1].toFixed(2)} rad` : ' · 无预测');
  $('predictionGapNotice').hidden = !!predicted;
  if (!predicted) $('predictionGapNotice').textContent = state.prediction?.available ?
    '当前帧不在模型 action chunk 中' : '该片段没有模型预测，只显示遥操夹爪';
  for (let hand = 0; hand < 2; hand++) {
    positionGripper(actualGrippers[hand], data.poses[i][hand], data.grippers[i][hand]);
    if (predicted) positionGripper(predictedGrippers[hand], predicted.poses[hand], predicted.grippers[hand]);
    else predictedGrippers[hand].group.visible = false;
    traversed[hand]?.geometry.setDrawRange(0, i - start + 1);
    const side = hand === 0 ? 'left' : 'right';
    updateGripperHud(`actual${hand}`, data.grippers[i][hand]);
    updateGripperHud(`predicted${hand}`, predicted?.grippers[hand] ?? null);
    $(`${side}Linear`).textContent = state.signals.linear[hand][i].toFixed(3);
    $(`${side}Angular`).textContent = state.signals.angular[hand][i].toFixed(1);
    $(`${side}Joint`).textContent = state.signals.joint[hand][i].toFixed(3);
  }
  for (const trail of actualTipTrails) trail.geometry.setDrawRange(0, i - start + 1);
  for (const {indices, trail} of predictedTipTrails)
    trail.geometry.setDrawRange(0, indices.findLastIndex(frame => frame <= i) + 1);
  const errors = state.signals.error.map(values => values[i]);
  const modelCovered = state.signals.modelJoint.every(values => values[i] !== null);
  $('modelPanel').hidden = !modelCovered;
  if (modelCovered) for (let hand = 0; hand < 2; hand++) {
    const side = hand === 0 ? 'Left' : 'Right';
    const lin = state.signals.modelLinear[hand][i];
    const ang = state.signals.modelAngular[hand][i];
    $(`model${side}Motion`).textContent = lin === null ? 'chunk 首帧' : `${lin.toFixed(3)} m/s · ${ang.toFixed(1)} °/s`;
    $(`model${side}Joint`).textContent = `${state.signals.modelJoint[hand][i].toFixed(3)} rad · ${errors[hand]?.toFixed(2) ?? '—'} cm`;
  }
  $('modelReadout').textContent = errors.every(v => v !== null) ?
    '预测速度按 chunk 内相邻帧计算；首帧不跨 chunk 外推。' :
    state.prediction?.available ? '当前帧未被模型 action chunk 覆盖。' : '模型预测到位后，此处显示当前位置误差。';
  drawCharts();
}
function updateGripperHud(id, angle) {
  const row = $(`${id}Gripper`);
  row.hidden = angle === null;
  if (angle === null) return;
  $(`${id}Angle`).textContent = `${angle.toFixed(3)} rad`;
  row.style.setProperty('--jaw-half-gap', `${Math.round(7 + THREE.MathUtils.clamp(angle / .8, 0, 1) * 22)}px`);
}
function drawChart(canvas, left, right, unit, available = true, models = null) {
  const rect = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
  if (!rect.width) return;
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = {l: 40, r: 12, t: 11, b: 22};
  ctx.clearRect(0, 0, w, h);
  if (!available) { ctx.fillStyle = '#718b97'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center'; ctx.fillText('等待模型 replay 数据', w / 2, h / 2); return; }
  const [start, end] = frameRange();
  const all = [...left.slice(start, end + 1), ...right.slice(start, end + 1),
    ...(models?.[0]?.slice(start, end + 1) || []), ...(models?.[1]?.slice(start, end + 1) || [])].filter(Number.isFinite);
  let min = Math.min(0, ...all), max = Math.max(0, ...all);
  if (max - min < 1e-6) max = min + 1;
  const span = max - min, innerW = w - pad.l - pad.r, innerH = h - pad.t - pad.b;
  ctx.strokeStyle = '#263b44'; ctx.lineWidth = 1; ctx.fillStyle = '#78909c'; ctx.font = '10px ui-monospace,monospace';
  for (let k = 0; k <= 2; k++) {
    const y = pad.t + k * innerH / 2;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText((max - span * k / 2).toFixed(unit === '°/s' ? 0 : 2), pad.l - 5, y + 3);
  }
  const x = i => pad.l + innerW * (i - start) / Math.max(1, end - start), y = v => pad.t + innerH * (max - v) / span;
  [[left, colors.left], [right, colors.right]].forEach(([series, color]) => {
    ctx.beginPath(); let started = false;
    for (let i = start; i <= end; i++) { const value = series[i]; if (!Number.isFinite(value)) { started = false; continue; } if (!started) { ctx.moveTo(x(i), y(value)); started = true; } else ctx.lineTo(x(i), y(value)); }
    ctx.strokeStyle = color; ctx.globalAlpha = .9; ctx.lineWidth = 1.5; ctx.stroke(); ctx.globalAlpha = 1;
  });
  if (models) [[models[0], colors.left], [models[1], colors.right]].forEach(([series, color]) => {
    ctx.beginPath(); ctx.setLineDash([4, 3]); let started = false;
    for (let i = start; i <= end; i++) { const value = series[i]; if (!Number.isFinite(value)) { started = false; continue; } if (!started || state.signals.modelWindowStarts.has(i)) { ctx.moveTo(x(i), y(value)); started = true; } else ctx.lineTo(x(i), y(value)); }
    ctx.strokeStyle = color; ctx.globalAlpha = .8; ctx.lineWidth = 1.6; ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
  });
  ctx.strokeStyle = '#d9eeee8c'; ctx.beginPath(); ctx.moveTo(x(state.frame), pad.t); ctx.lineTo(x(state.frame), pad.t + innerH); ctx.stroke();
  ctx.textAlign = 'left'; ctx.fillText(`${state.episode.times[start].toFixed(1)} s`, pad.l, h - 3); ctx.textAlign = 'right'; ctx.fillText(`${state.episode.times[end].toFixed(1)} s`, w - pad.r, h - 3);
}
function drawCharts() {
  if (!state.signals) return;
  for (const [name, unit] of [['linear', 'm/s'], ['angular', '°/s'], ['joint', 'rad'], ['error', 'cm']]) {
    const models = name === 'error' ? null : state.signals[`model${name[0].toUpperCase()}${name.slice(1)}`];
    drawChart($(`${name}Chart`), ...state.signals[name], unit, name !== 'error' || !!state.prediction?.available, models);
  }
}
new ResizeObserver(drawCharts).observe($('linearChart'));

function updatePredictionStatus() {
  const pred = state.prediction;
  if (pred?.available) {
    $('modelStatus').textContent = 'π0.5 预测已接入';
    $('modelDetail').textContent = `${pred.windows.length} 个 action chunk · A100 离线预测${pred.training_overlap ? ' · 训练集片段，非独立评测' : ''}`;
    $('checkpoint').textContent = `CHECKPOINT ${pred.checkpoint_id}`;
    $('sourceBadge').textContent = '遥操 + 模型预测';
    $('windowStatus').textContent = '预测末端与双侧指尖轨迹按 action chunk 分段；实体夹爪为遥操，线框夹爪为当前预测。';
  } else {
    $('modelStatus').textContent = '等待模型结果';
    $('modelDetail').textContent = pred?.reason || '尚无模型 replay 结果。';
    $('checkpoint').textContent = state.checkpoint ? `CHECKPOINT ${state.checkpoint.step} · ${state.checkpoint.completed_episodes} 个片段已生成` : 'CHECKPOINT —';
    $('sourceBadge').textContent = '真实遥操已加载 · 模型待接入';
    $('windowStatus').textContent = '灰色区间表示尚无模型预测。';
  }
}
async function loadEpisode(id) {
  stopPlayback();
  state.videoReady = false;
  $('playButton').disabled = true;
  const loadToken = ++state.videoLoadToken;
  const response = await fetch(`/api/episode?id=${encodeURIComponent(id)}&checkpoint=${encodeURIComponent($('checkpointSelect').value)}`);
  const result = await response.json();
  if (loadToken !== state.videoLoadToken) return;
  if (!response.ok) throw Error(result.error || '读取片段失败');
  state.episode = result.recording; state.prediction = result.prediction; state.checkpoint = result.checkpoint;
  state.signals = signalsFor(state.episode);
  setPrimaryView(state.primaryView);
  const windows = state.prediction?.available ? state.prediction.windows : [];
  $('timelineMode').querySelector('[value="chunk"]').disabled = !windows.length;
  $('chunkSelect').replaceChildren(...windows.map((window, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `${String(index + 1).padStart(2, '0')} · ${state.episode.times[window.indices[0]].toFixed(2)}–${state.episode.times[window.indices.at(-1)].toFixed(2)} s · ${window.indices.length} 帧`;
    return option;
  }));
  state.chunkIndex = Math.min(state.chunkIndex, Math.max(0, windows.length - 1));
  updatePredictionStatus(); updateTimelineMode(); loadVideos(state.episode, loadToken);
}
function stopPlayback() {
  state.playing = false;
  state.syncToken++;
  $('playButton').textContent = '▶';
  for (const video of videos) video.pause();
}
async function loadVideos(data, token) {
  state.videoReady = false;
  $('playButton').disabled = true;
  $('videoStatus').textContent = '视频加载中';
  videoTiles.forEach(tile => tile.classList.remove('has-frame'));
  const sources = [data.videos?.left, data.videos?.right, data.videos?.center];
  if (sources.some(source => !source)) {
    $('videoStatus').textContent = '视频尚未导出';
    return;
  }
  const results = await Promise.all(videos.map((video, view) => new Promise(resolve => {
    video.onloadedmetadata = () => resolve(true);
    video.onerror = () => resolve(false);
    video.src = sources[view];
    video.load();
  })));
  if (token !== state.videoLoadToken) return;
  state.videoReady = results.every(Boolean);
  $('playButton').disabled = !state.videoReady;
  $('videoStatus').textContent = state.videoReady ? '三路视频已同步 · 左腕 / 右腕 / 中心' : '视频读取失败';
  if (state.videoReady) { setFrame(state.frame); videos.forEach((_, view) => paintVideo(view)); }
}
function fillEpisodes() {
  const taskId = Number($('taskSelect').value);
  const items = state.catalog.episodes.filter(e => e.task_id === taskId);
  $('episodeSelect').replaceChildren(...items.map(e => {
    const option = document.createElement('option');
    option.value = e.episode_index;
    option.textContent = `${e.episode_index} · 第 ${e.step} 步 · ${e.step_name} (${e.duration_s}s)`;
    return option;
  }));
  if (items.length) loadEpisode(items[0].episode_index).catch(showError);
}
function showError(error) { $('modelStatus').textContent = '读取失败'; $('modelDetail').textContent = error.message; console.error(error); }
function fillCheckpoints(checkpoints, selected = null) {
  $('checkpointSelect').replaceChildren(...checkpoints.map(item => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.label} · ${item.completed_episodes} 段预测`;
    return option;
  }));
  $('checkpointSelect').value = checkpoints.some(item => item.id === selected) ? selected : checkpoints.at(-1).id;
}
async function init() {
  const response = await fetch('/api/catalog'); state.catalog = await response.json();
  $('suiteLabel').textContent = `${state.catalog.episodes.length} episodes · ${state.catalog.fps} Hz`;
  fillCheckpoints(state.catalog.checkpoints);
  const tasks = [...new Map(state.catalog.episodes.map(e => [e.task_id, e.task_name])).entries()];
  $('taskSelect').replaceChildren(...tasks.map(([id, name]) => { const option = document.createElement('option'); option.value = id; option.textContent = `Task ${id} · ${name}`; return option; }));
  fillEpisodes();
}
$('taskSelect').addEventListener('change', fillEpisodes);
$('episodeSelect').addEventListener('change', e => loadEpisode(e.target.value).catch(showError));
$('checkpointSelect').addEventListener('change', () => loadEpisode($('episodeSelect').value).catch(showError));
$('refreshButton').addEventListener('click', async () => {
  try {
    const selected = $('checkpointSelect').value;
    const response = await fetch('/api/catalog');
    state.catalog = await response.json();
    fillCheckpoints(state.catalog.checkpoints, selected);
    await loadEpisode($('episodeSelect').value);
  } catch (error) { showError(error); }
});
$('fitButton').addEventListener('click', fitCamera);
$('resetButton').addEventListener('click', resetCamera);
for (const layer of Object.keys(trajectoryLayers)) $(`${layer}Toggle`).addEventListener('change', updateTrajectoryVisibility);
$('swapCamera').addEventListener('click', () => setPrimaryView((state.primaryView + 1) % videoTiles.length));
videoTiles.forEach((tile, view) => tile.addEventListener('click', () => setPrimaryView(view)));
$('timelineMode').addEventListener('change', e => { state.timelineMode = e.target.value; updateTimelineMode(); });
$('chunkSelect').addEventListener('change', e => selectChunk(Number(e.target.value)));
$('previousChunk').addEventListener('click', () => selectChunk(state.chunkIndex - 1));
$('nextChunk').addEventListener('click', () => selectChunk(state.chunkIndex + 1));
$('exportChunk').addEventListener('click', () => {
  const window = selectedWindow();
  if (!window) return;
  const frames = window.indices.map((frame, index) => ({
    frame, time_s: state.episode.times[frame],
    teleop: {poses: state.episode.poses[frame], grippers_rad: state.episode.grippers[frame]},
    prediction: {poses: window.poses[index], grippers_rad: window.grippers[index]},
    position_error_cm: window.position_cm?.[index] ?? null,
    rotation_error_deg: window.rotation_deg?.[index] ?? null,
  }));
  const payload = {episode_index: state.episode.episode_index, checkpoint_id: state.prediction.checkpoint_id,
    chunk_number: state.chunkIndex + 1, fps: state.episode.fps, frames};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url;
  link.download = `episode-${state.episode.episode_index}-${state.prediction.checkpoint_id}-chunk-${state.chunkIndex + 1}.json`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
});
$('scrubber').addEventListener('input', e => { stopPlayback(); setFrame(e.target.value); });
$('speedSelect').addEventListener('change', e => { for (const video of videos) video.playbackRate = Number(e.target.value); });
$('playButton').addEventListener('click', async () => {
  if (!state.episode || !state.videoReady) return;
  if (state.playing) { stopPlayback(); return; }
  const [start, end] = frameRange();
  if (state.frame >= end) setFrame(start);
  const rate = Number($('speedSelect').value);
  for (const video of videos) video.playbackRate = rate;
  state.playing = true;
  $('playButton').textContent = 'Ⅱ';
  const token = ++state.syncToken;
  try { await Promise.all(videos.map(video => video.play())); }
  catch (error) { stopPlayback(); $('videoStatus').textContent = '浏览器未能播放视频'; return; }
  if (token !== state.syncToken) return;
  if (videos[0].requestVideoFrameCallback) {
    const onFrame = (_now, metadata) => {
      if (token !== state.syncToken || !state.playing) return;
      paintVideo(0);
      const frame = Math.round(metadata.mediaTime * state.episode.fps);
      if (frame >= end) { stopPlayback(); setFrame(end); return; }
      setFrame(frame, false);
      for (const video of videos.slice(1))
        if (Math.abs(video.currentTime - metadata.mediaTime) > .07) video.currentTime = metadata.mediaTime;
      videos[0].requestVideoFrameCallback(onFrame);
    };
    videos[0].requestVideoFrameCallback(onFrame);
    for (let view = 1; view < videos.length; view++) {
      const onOtherFrame = () => {
        if (token !== state.syncToken || !state.playing) return;
        paintVideo(view);
        videos[view].requestVideoFrameCallback(onOtherFrame);
      };
      videos[view].requestVideoFrameCallback(onOtherFrame);
    }
  }
});
videos[0].addEventListener('ended', () => { if (state.episode) setFrame(frameRange()[1]); stopPlayback(); });
function fallbackVideoTick() {
  requestAnimationFrame(fallbackVideoTick);
  if (state.playing && state.videoReady && !videos[0].requestVideoFrameCallback) {
    videos.forEach((_, view) => paintVideo(view));
    const frame = Math.round(videos[0].currentTime * state.episode.fps);
    if (frame >= frameRange()[1]) { stopPlayback(); setFrame(frameRange()[1]); }
    else {
      setFrame(frame, false);
      for (const video of videos.slice(1))
        if (Math.abs(video.currentTime - videos[0].currentTime) > .07) video.currentTime = videos[0].currentTime;
    }
  }
}
requestAnimationFrame(fallbackVideoTick);
init().catch(showError);
