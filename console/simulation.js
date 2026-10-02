const $ = id => document.getElementById(id);
let catalog = null;
let selectedRunId = /^[0-9a-f]{12}$/.test(new URLSearchParams(location.search).get('run') || '')
  ? new URLSearchParams(location.search).get('run') : null;
let runListSignature = '';
let activeRunId = null;
let stoppableRunId = null;
let activeStopRequested = false;

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'HTTP ' + response.status);
  return body;
}

function renderPolicies() {
  // Populate from the server registry so newly validated model backends are
  // selectable without maintaining a second, stale list in the HTML.
  const backendSelect = $('inferenceBackend');
  for (const option of [...backendSelect.options]) {
    if (!(option.value in (catalog.inference_backends || {}))) option.remove();
  }
  for (const [id, backend] of Object.entries(catalog.inference_backends || {})) {
    if (![...backendSelect.options].some(option => option.value === id)) {
      backendSelect.add(new Option(backend.label || id, id));
    }
  }
  const select = $('policySelect');
  const list = $('policyList');
  const previous = select.value;
  select.replaceChildren();
  list.replaceChildren();
  for (const policy of catalog.policies) {
    const option = new Option(policy.label, policy.id);
    option.disabled = !policy.ready;
    select.add(option);
    const item = document.createElement('div');
    item.className = 'policy-item ' + (policy.ready ? 'ready' : 'pending');
    const title = document.createElement('strong');
    title.textContent = (policy.ready ? '可执行 · ' : '待适配 · ') + policy.label;
    const note = document.createElement('small');
    note.textContent = policy.description || policy.reason || '';
    item.append(title, note);
    list.append(item);
  }
  const selected = catalog.policies.find(policy => policy.id === previous && policy.ready)
    || catalog.policies.find(policy => policy.ready && policy.online_inference)
    || catalog.policies.find(policy => policy.ready);
  if (selected) select.value = selected.id;
  updateSelectedPolicy();
}

function updateSelectedPolicy() {
  const policy = catalog?.policies.find(item => item.id === $('policySelect').value);
  $('policyNote').textContent = policy?.description || policy?.reason || '选择一个策略。';
  const online = Boolean(policy?.online_inference);
  if (!online) $('runMode').value = 'fixed';
  $('runMode').querySelector('[value="until_success"]').disabled = !online;
  $('stepsWrap').hidden = $('runMode').value === 'until_success';
  $('steps').disabled = $('stepsWrap').hidden;
  $('inferenceBackendWrap').hidden = !online;
  $('inferenceControls').hidden = !online;
  const allowed = (policy?.supported_backends || ['rtx5090'])
    .filter(id => id in (catalog?.inference_backends || {}));
  for (const option of $('inferenceBackend').options) {
    option.disabled = online && !allowed.includes(option.value);
  }
  if (online && !allowed.includes($('inferenceBackend').value)) {
    $('inferenceBackend').value = allowed.find(id => id === 'rtx5090' || id.endsWith('_rtx5090')) || allowed[0];
  }
  const backendId = $('inferenceBackend').value;
  const backend = catalog?.inference_backends?.[backendId];
  $('inferenceStatus').textContent = online
    ? (backend?.label || backendId) + '：' + (backend?.reason || '状态未知')
    : '';
  $('startInferenceButton').disabled = !online || !backend?.deploy_ready || Boolean(backend?.ready) || Boolean(activeRunId);
  $('startInferenceButton').textContent = backend?.active ? '重连推理通道' : '启动推理服务';
  $('stopInferenceButton').disabled = !online || !backend?.active || Boolean(activeRunId);
  const localInferenceReady = online && backend?.ready && (backendId === 'rtx5090' || backendId.endsWith('_rtx5090'));
  $('runButton').disabled = !policy?.ready || (!catalog.gpu_status.available && !localInferenceReady) || Boolean(activeRunId) || (online && !backend?.deploy_ready);
  $('stopRunButton').disabled = !stoppableRunId || activeStopRequested;
}

async function refreshCatalog() {
  try {
    catalog = await request('/api/catalog');
    const status = catalog.gpu_status;
    $('runtimeBadge').textContent = status.available
      ? 'squirrel · RTX 5090 GPU 0 空闲'
      : 'squirrel GPU 0 暂不可用 · ' + status.reason;
    renderPolicies();
  } catch (error) {
    $('runtimeBadge').textContent = '仿真服务不可用 · ' + error.message;
    $('runButton').disabled = true;
    $('startInferenceButton').disabled = true;
    $('stopInferenceButton').disabled = true;
  }
}

function runLabel(run) {
  const time = new Date(run.created_at).toLocaleString();
  return run.policy_label + ' · random:' + run.setup_index + ' · ' + time;
}

function renderRun(run) {
  selectedRunId = run.id;
  const result = run.result?.episodes?.[0];
  const objective = run.task_objective || run.result?.task_objective || 'plate';
  $('runSummary').textContent = [
    runLabel(run),
    '状态：' + run.status + (run.error ? ' · ' + run.error : ''),
    result ? (objective === 'plate_return' ? '完整任务成功：' : '仅放上盘子成功：') +
      (result.success ? '是' : '否') +
      (result.plate_placed === true && objective === 'plate_return' ? ' · 已完成放盘阶段' : '') +
      ' · 控制步数：' + result.policy_steps +
      (result.stop_reason === 'user_stop' ? ' · 手动停止' : '') +
      ' · 视频帧：' + result.video_frames +
      ' · 关节采样：' + result.joint_samples :
      (run.result?.error ? '仿真中止：' + run.result.error +
        (run.observed_policy_steps != null ? ' · 已完成 ' + run.observed_policy_steps + ' 个在线控制步' : '')
        : '等待运行结果。'),
    '策略输入：' + run.policy + ' · 录制相机：' + run.camera + ' · GPU ' + run.gpu,
    '双臂命名：' + (run.canonical_hand_sides === true
      ? '机器人自身左右已校正（右腕为历史 USD LeftMount）'
      : '旧版 USD 左右命名；历史右腕可能是机器人自身左腕，不能视为右腕标定'),
    run.inference_backend ? '模型推理位置：' + (catalog?.inference_backends?.[run.inference_backend]?.label || run.inference_backend) : '',
    run.calibration_result ? '夹爪校准：' + run.calibration_result.label +
      ' · 杯子最高离桌 ' + run.calibration_result.max_cup_lift_mm + ' mm' +
      ' · 最终倾角 ' + run.calibration_result.final_cup_tilt_deg + '°' +
      ' · ' + run.calibration_result.scene_tuning : '',
  ].join('\n');
  const video = $('simVideo');
  const preview = $('simPreview');
  const mediaStatus = $('simMediaStatus');
  const previewPath = '';
  const finished = !['starting','running'].includes(run.status);
  $('threeViewResult').hidden = !(finished && run.three_views_available);
  for (const [id, side] of [['simLeftVideo','left'],['simRightVideo','right']]) {
    const other = $(id);
    if (finished && run.three_views_available) {
      const url = '/runs/' + run.id + '/video_' + side + '_wrist.mp4';
      if (other.getAttribute('src') !== url) other.src = url;
    } else { other.pause(); other.removeAttribute('src'); }
  }
  if (!video.dataset.syncBound) {
    for (const event of ['play','pause','seeked','ratechange']) video.addEventListener(event, () => {
      if ($('threeViewResult').hidden) return;
      for (const id of ['simLeftVideo','simRightVideo']) {
        const v=$(id); v.playbackRate=video.playbackRate;
        if (event==='seeked' || event==='play') v.currentTime=video.currentTime;
        if (event==='play') v.play().catch(()=>{});
        if (event==='pause') v.pause();
      }
    });
    video.dataset.syncBound='1';
  }
  if (run.video_available && finished) {
    const path = '/runs/' + run.id + '/video.mp4';
    if (video.getAttribute('src') !== path) {
      video.src = path;
      video.load();
    }
    if (previewPath) video.poster = previewPath;
    preview.hidden = true;
    video.hidden = false;
    mediaStatus.textContent = run.three_views_available ? '三路同步视频已就绪：胸口、左腕、右腕。' : '旧运行仅有一路录制视频。';
  } else {
    video.removeAttribute('src');
    video.removeAttribute('poster');
    video.hidden = true;
    preview.hidden = !previewPath;
    if (previewPath && preview.getAttribute('src') !== previewPath) preview.src = previewPath;
    mediaStatus.textContent = run.status === 'starting' || run.status === 'running'
      ? '仿真运行中，实时预览已关闭；结束后显示三路视频。'
      : '本次运行没有可播放视频；请选择下方一条已完成记录。';
  }
  const audit = run.online_audit;
  const auditPanel = $('onlineAudit');
  const wristPanel = $('onlineWristImages');
  auditPanel.hidden = !audit;
  wristPanel.hidden = !audit?.live_wrist_images && !audit?.live_head_image;
  if (audit) {
    const modelViews = audit.model_input_views || ['left_wrist', 'right_wrist'];
    auditPanel.textContent = [
      '在线闭环审计：' + (audit.validated ? '通过' : '未通过'),
      '实时模型输入视角：' + modelViews.join('、') + '；' + audit.online_observation_steps + '/' + audit.expected_policy_steps + ' 步',
      '右腕相机：' + (run.canonical_hand_sides === true
        ? '自身右臂映射已核对；相机外参仍未实测标定'
        : '历史运行未启用自身左右映射，右腕输入存在错臂风险'),
      '坐标映射：' + (audit.world_pose_transform || '未知') + '（仿真假设，非实测标定）',
      run.online_control?.continuous_targets
        ? '连续执行：模型 ' + (audit.model_request_hz || 10) + ' Hz，动作 ' +
          (audit.action_execution_hz || 10) + ' Hz，物理子步更新；指令速度≤' +
          run.online_control.velocity_rad_s + ' rad/s，加速度≤' +
          run.online_control.acceleration_rad_s2 + ' rad/s²；坐标配置：' +
          (run.online_control.calibration || '历史配置')
        : '连续执行：历史运行未启用本次指令限速方案',
      '夹爪映射：源 ' + audit.gripper_calibration?.source_closed_rad + '～' + audit.gripper_calibration?.source_open_rad +
        ' rad → 仿真 ' + audit.gripper_calibration?.sim_closed_rad + '～' + audit.gripper_calibration?.sim_open_rad + ' rad',
      '双臂关节最大变化：' + Number(audit.max_arm_joint_motion_rad).toFixed(4) + ' rad',
      audit.measured_gripper_joint_rad?.left ?
        '夹爪实测：左 ' + Number(audit.measured_gripper_joint_rad.left.start_rad).toFixed(3) + '→' +
          Number(audit.measured_gripper_joint_rad.left.end_rad).toFixed(3) + ' rad，右 ' +
          Number(audit.measured_gripper_joint_rad.right.start_rad).toFixed(3) + '→' +
          Number(audit.measured_gripper_joint_rad.right.end_rad).toFixed(3) + ' rad' :
        '夹爪实测：无记录',
      '平均模型推理：' + Number(audit.mean_model_latency_ms).toFixed(0) + ' ms',
    ].join('\n');
  }
  $('onlineHeadFigure').hidden = !audit?.live_head_image;
  if (audit?.live_head_image) $('onlineHeadImage').src = '/runs/' + run.id + '/input_head_0000.jpg';
  if (audit?.live_wrist_images) {
    $('onlineLeftImage').src = '/runs/' + run.id + '/input_left_0000.jpg';
    $('onlineRightImage').src = '/runs/' + run.id + '/input_right_0000.jpg';
    const modelViews = audit.model_input_views || ['left_wrist', 'right_wrist'];
    $('onlineLeftCaption').textContent = '左腕 · ' + (modelViews.includes('left_wrist') ? '在线模型输入' : '仅审计，未输入模型');
    $('onlineRightCaption').textContent = (run.canonical_hand_sides === true
      ? '机器人自身右腕' : '历史 USD 右腕（机器人自身左侧）') + ' · ' +
      (modelViews.includes('right_wrist') ? '在线模型输入' : '仅审计，未输入模型');
  }
  const links = $('resultLinks');
  links.replaceChildren();
  if (run.joints_available) {
    const link = document.createElement('a');
    link.href = '/runs/' + run.id + '/joints.csv';
    link.textContent = '下载关节 CSV';
    links.append(link);
  }
  if (audit) {
    const link = document.createElement('a');
    link.href = '/runs/' + run.id + '/online_adapter.jsonl';
    link.textContent = '下载逐步在线适配审计';
    links.append(link);
  }
  if (run.calibration_result) {
    for (const [filename, label] of [
      ['calibration_summary.json', '查看校准汇总'],
      ['grasp_calibration.jsonl', '下载逐步夹爪轨迹'],
    ]) {
      const link = document.createElement('a');
      link.href = '/runs/' + run.id + '/' + filename;
      link.textContent = label;
      links.append(link);
    }
  }
}

async function refreshRuns() {
  try {
    const runs = await request('/api/runs');
    const activeRun = runs.find(run => run.status === 'starting' || run.status === 'running');
    activeRunId = activeRun?.id || null;
    const stoppableRun = activeRun || runs.find(run => run.status === 'unverified' && !run.finished_at);
    stoppableRunId = stoppableRun?.id || null;
    activeStopRequested = Boolean(stoppableRun?.stop_requested);
    if (catalog) updateSelectedPolicy();
    const list = $('runList');
    const signature = JSON.stringify(runs.map(run => [run.id, run.status, run.video_available]));
    if (signature !== runListSignature) {
      runListSignature = signature;
      const comparison = $('modelComparison');
      comparison.replaceChildren();
      const table = document.createElement('table');
      const header = table.insertRow();
      for (const label of ['模型 / checkpoint', '场景 / seed', '步数上限', '已执行', '完整任务', '推理均耗时']) {
        const cell = document.createElement('th');
        cell.textContent = label;
        header.append(cell);
      }
      for (const run of runs.filter(item => item.inference_backend)) {
        const episode = run.result?.episodes?.[0];
        const latency = run.online_audit?.mean_model_latency_ms;
        const row = table.insertRow();
        const values = [run.policy, `${run.setup_index} / ${run.seed}`,
          run.run_until_success ? '无上限' : run.steps,
          episode?.policy_steps ?? run.observed_policy_steps ?? '—',
          run.task_objective !== 'plate_return' ? '非完整任务评测' :
            episode?.full_task_success === true ? '成功' :
            run.status === 'failed' ? '运行失败' : episode ? '未完成' : '运行中',
          Number.isFinite(latency) ? `${latency.toFixed(0)} ms` : '—'];
        for (const value of values) row.insertCell().textContent = String(value);
      }
      comparison.append(table);
      list.replaceChildren();
      if (!runs.length) list.textContent = '暂无运行。';
      for (const run of runs) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'run-item ' + run.status + (run.id === selectedRunId ? ' selected' : '');
      const title = document.createElement('span');
      title.textContent = runLabel(run);
      const detail = document.createElement('small');
      detail.textContent = run.status + ' · ' + (run.run_until_success ? '直到成功' : run.steps + ' 步') + ' · ' + run.camera;
      button.append(title, detail);
      button.addEventListener('click', () => {
        renderRun(run);
        runListSignature = '';
        refreshRuns();
      });
      list.append(button);
      }
    }
    const selected = runs.find(run => run.id === selectedRunId)
      || runs.find(run => run.id === activeRunId)
      || runs.find(run => run.policy?.startsWith('pi05-') && run.status === 'completed' && run.video_available)
      || runs[0];
    if (selected) renderRun(selected);
  } catch (error) {
    $('runList').textContent = '读取运行记录失败：' + error.message;
  }
}

$('policySelect').addEventListener('change', () => {
  $('runMode').value = 'fixed';
  updateSelectedPolicy();
});
$('inferenceBackend').addEventListener('change', updateSelectedPolicy);
$('runMode').addEventListener('change', updateSelectedPolicy);
$('stopRunButton').addEventListener('click', async () => {
  if (!stoppableRunId) return;
  $('stopRunButton').disabled = true;
  $('formStatus').textContent = '停止请求已发送，正在结束当前步并收尾录像…';
  try {
    await request('/api/runs/' + stoppableRunId + '/stop', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
    });
  } catch (error) {
    $('formStatus').textContent = error.message;
  }
  await refreshRuns();
});
for (const [id, action] of [['startInferenceButton', 'start'], ['stopInferenceButton', 'stop']]) {
  $(id).addEventListener('click', async () => {
    const backend = $('inferenceBackend').value;
    $('startInferenceButton').disabled = true;
    $('stopInferenceButton').disabled = true;
    $('inferenceStatus').textContent = action === 'start' ? '正在启动模型服务；加载权重可能需要约一分钟…' : '正在退出选中的模型服务…';
    try {
      await request('/api/inference/' + action, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({backend}),
      });
    } catch (error) {
      $('inferenceStatus').textContent = error.message;
    }
    await refreshCatalog();
  });
}
$('refreshButton').addEventListener('click', async () => {
  await Promise.all([refreshCatalog(), refreshRuns()]);
});
$('runForm').addEventListener('submit', async event => {
  event.preventDefault();
  const button = $('runButton');
  button.disabled = true;
  $('formStatus').textContent = '正在自动切换并加载所选模型，确认就绪后启动仿真…';
  const requestBody = {
    policy: $('policySelect').value,
    setup_index: Number($('setupIndex').value),
    seed: Number($('seed').value),
    ...( $('runMode').value === 'fixed' ? {steps: Number($('steps').value)} : {} ),
    run_until_success: $('runMode').value === 'until_success',
    task_objective: $('taskObjective').value,
    camera: $('camera').value,
    inference_backend: $('inferenceBackend').value,
  };
  try {
    const run = await request('/api/runs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(requestBody),
    });
    selectedRunId = run.id;
    $('formStatus').textContent = '运行 ' + run.id + ' 已启动。';
    await refreshRuns();
  } catch (error) {
    $('formStatus').textContent = error.message;
  } finally {
    await refreshCatalog();
  }
});
Promise.all([refreshCatalog(), refreshRuns()]);
setInterval(refreshRuns, 5000);
setInterval(refreshCatalog, 120000);
