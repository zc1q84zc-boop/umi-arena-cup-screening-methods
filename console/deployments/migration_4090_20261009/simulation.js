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
  const contactSelect = $('contactProfile');
  for (const profile of catalog.stiffness_profiles || []) {
    let option = [...contactSelect.options].find(item => item.value === profile.id);
    if (!option) { option = new Option(profile.label, profile.id); contactSelect.add(option); }
    option.disabled = !profile.ready;
    option.textContent = profile.label + (profile.ready ? ' · 物理检查通过' : ' · 待物理检查');
  }
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
    title.textContent = (policy.ready ? '可执行 · ' : `${policy.pending_label || '待适配'} · `) + policy.label;
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
  const tuned = policy?.simulator_profile === 'tuned_v1';
  const tunedOnline = policy?.simulator_profile === 'tuned_online_v1';
  $('contactProfile').disabled = !tunedOnline || $('runMode').value === 'until_success';
  if ($('contactProfile').disabled) $('contactProfile').value = 'baseline';
  $('onlineSetupNote').hidden=!online;
  $('onlineSetupNote').textContent = tunedOnline
    ? 'tuned_online_v1 固定初始拟合与杯盘布局（场景0、seed42），保留近似手根／工具端坐标；CAD开度双向映射和当前GPU基座随动腕相机已启用。示范专用平移、帧依赖闭爪和杯位辅助均未迁入。运行后输出胸口、左腕、右腕，旧视频不更新。'
    : '历史辅助路径使用 reference_259632_v1 的近似开度／相机配置，与 tuned_online_v1 分开；使用杯位或接触反馈的诊断不能当作纯模型成绩。';
  if(tuned){$('setupIndex').value='0';$('seed').value='42';$('steps').value='215';$('camera').value='overview';}
  if(tunedOnline){$('setupIndex').value='0';$('seed').value='42';$('camera').value='head';}
  for(const id of ['setupIndex','seed','camera','runMode','taskObjective'])$(id).disabled=tuned;
  if(tunedOnline)for(const id of ['setupIndex','seed','camera'])$(id).disabled=true;
  if(tuned)$('taskObjective').value='plate';
  $('steps').disabled=$('stepsWrap').hidden||tuned;
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
      ? 'squirrel_4090_2 · 仿真 GPU 0 空闲'
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
    result ? (run.simulator_profile==='tuned_v1'
      ? '示范重放首次放盘：'+(result.ever_success ? '是 · 步 '+result.first_success_policy_step : '未核验')+'；末帧放盘：'
      : (run.left_return_diagnostic ? '左臂取杯放回诊断成功：' : objective === 'plate_return' ? '完整任务成功：' : '仅放上盘子成功：')) +
      ((run.left_return_diagnostic ? result.left_return_diagnostic?.diagnostic_success : result.success) ? '是' : '否') +
      (run.left_return_diagnostic ? ' · 杯子由初始化放在盘上；不计完整任务成绩' : result.plate_placed === true && objective === 'plate_return' ? ' · 已完成放盘阶段' : '') +
      ' · 控制步数：' + result.policy_steps +
      (result.stop_reason === 'user_stop' ? ' · 手动停止' : '') +
      ' · 视频帧：' + result.video_frames +
      ' · 关节采样：' + result.joint_samples :
      (run.result?.error ? '仿真中止：' + run.result.error +
        (run.observed_policy_steps != null ? ' · 已完成 ' + run.observed_policy_steps + ' 个在线控制步' : '')
        : '等待运行结果。'),
    '策略输入：' + run.policy + ' · 录制相机：' + run.camera + ' · GPU ' + run.gpu,
    '接触配置：' + ((run.contact_profile || 'baseline') === 'baseline' ? '原始刚性杯基线' : run.contact_profile + ' · 未实测物理对照，非原环境模型成绩'),
    run.language_instructions?.protocol === 'umi_arena_cup_primitives_20261008'
      ? '语言指令：UMI Arena 官网杯子原语；右手放盘并释放后切换左手归位。LingBot 原生模型输出，无额外闭合或杯位动作辅助。' : '',
    run.imported_verified_run ? '已导入同一次核验运行：'+run.imported_verified_run+'；没有重复执行仿真。' : '',
    '双臂命名：' + (['tuned_v1','tuned_online_v1'].includes(run.simulator_profile) ? '机器人自身左=LeftMount，右=RightMount（调优资产）' : run.canonical_hand_sides === true
      ? '机器人自身左右已校正（右腕为历史 USD LeftMount）'
      : '旧版 USD 左右命名；历史右腕可能是机器人自身左腕，不能视为右腕标定'),
    run.inference_backend ? '模型推理位置：' + (catalog?.inference_backends?.[run.inference_backend]?.label || run.inference_backend) : '',
    run.simulator_profile==='tuned_online_v1'
      ? '已启用 tuned_online_v1：双指显式驱动／CAD双向开度／详细接触层／逐帧跟随腕相机；不含示范专用偏移。'
      : '',
    run.cup_physics_profile ? '杯壁：PhysX 弹性薄壳 FEM；网格可形变，无隐藏刚性杯。参数未实测，不含塑性压痕/破裂；刚体接触传感器不适用于此模式，不能把缺失力数据解释为零力。' : '',
    run.cup_physics_profile ? '弹性模量：' + (run.cup_physics_profile.youngs_modulus_Pa/1e9).toFixed(2) +
      ' GPa · 固定厚度：' + (1000*run.cup_physics_profile.thickness_m).toFixed(1) + ' mm（未实测）' : '',
    run.cup_physics_profile ? 'FEM 求解：' + run.cup_physics_profile.solver_position_iterations +
      ' 次位置迭代 · 物理 ' + run.cup_physics_profile.physics_hz +
      ' Hz；动作执行仍为 30 Hz。高精度对照不代表材料已标定或完全数值收敛。' : '',
    run.cup_lift_summary ? '杯底最高离桌：' + (1000*run.cup_lift_summary.max_bottom_clearance_m).toFixed(1) +
      ' mm · 直立稳定 ≥50 mm / ≥10 动作步：' + (run.cup_lift_summary.stable_grasp_verified ? '已核验，需视频复核' : '未达到') +
      ' · 最长保持：' + run.cup_lift_summary.longest_stable_action_steps + ' 步；不等于完整任务成功。' : '',
    result?.left_return_diagnostic ?
      '左臂分阶段诊断：最高离盘抬升 ' + (1000*result.left_return_diagnostic.max_cup_lift_m).toFixed(1) +
      ' mm；稳定抬升≥50 mm：' + (result.left_return_diagnostic.stable_lift_confirmed ? '是' : '否') +
      '。右臂指令保持；原模型提示词与权重未改。' : '',
    run.left_extra_closure_fraction > 0 ?
      '额外闭合对照：最多减少 ' + (100*run.left_extra_closure_fraction).toFixed(0) +
      '% 行程的目标开度；张开/释放、驱动力与摩擦不变。属于动作辅助，不计纯模型成绩。' : '',
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
  const mainViewLabel = ({overview:'总览',head:'胸口',left_wrist:'左腕',right_wrist:'右腕'})[run.camera] || run.camera;
  $('threeViewDescription').textContent=mainViewLabel+'视频见上方；下方为同次仿真的机器人自身左腕和右腕。主视频播放／暂停／跳转会同步双腕。';
  $('onlineResultNote').hidden=!run.inference_backend;
  $('onlineResultNote').textContent = run.simulator_profile==='tuned_online_v1'
    ? '本次采用 tuned_online_v1：近似互逆坐标／工具端变换，固定CAD开度映射、双指显式驱动、详细接触层和每帧随动腕相机。10 Hz模型请求、30 Hz动作、'+(run.cup_physics_profile?.physics_hz || 60)+' Hz物理子步，指令限速0.8 rad/s、加速度1.5 rad/s²。无示范专用偏移或杯位辅助；模型权重不变，尚非真机标定。'
    : '本次为历史 reference_259632_v1 路径，使用原运行的近似坐标／开度／相机配置；没有被新 tuned_online_v1 静默替换。';
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
    mediaStatus.textContent = run.three_views_available ? '三路同步视频已就绪：'+mainViewLabel+'、左腕、右腕。' : '旧运行仅有一路录制视频。';
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
      audit.tuned_profile_audit?.validated
        ? '调优配置核验：实时腕相机输入与录制位姿一致、双指镜像命令、连续速度／加速度及三路帧数通过。' : '',
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
  if (run.cup_physics_profile) {
    const link = document.createElement('a');
    link.href = '/runs/' + run.id + '/cup_deformation.jsonl';
    link.textContent = '下载杯壁 FEM 形变审计（非力传感器）';
    links.append(link);
  } else if (run.physics_contact_trial && finished) {
    const link = document.createElement('a');
    link.href = '/runs/' + run.id + '/gripper_contact_audit.jsonl';
    link.textContent = '下载双指接触审计（力值未标定）';
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
    const signature = JSON.stringify(runs.map(run => [run.id, run.status, run.video_available, Boolean(run.cup_lift_summary)]));
    if (signature !== runListSignature) {
      runListSignature = signature;
      const comparison = $('modelComparison');
      comparison.replaceChildren();
      const table = document.createElement('table');
      const header = table.insertRow();
      for (const label of ['模型 / checkpoint', '物理配置', '场景 / seed', '步数上限', '已执行', '杯底稳定抬升', '完整任务', '推理均耗时']) {
        const cell = document.createElement('th');
        cell.textContent = label;
        header.append(cell);
      }
      for (const run of runs.filter(item => item.inference_backend)) {
        const episode = run.result?.episodes?.[0];
        const latency = run.online_audit?.mean_model_latency_ms;
        const row = table.insertRow();
        const values = [run.policy, run.contact_profile || 'baseline', `${run.setup_index} / ${run.seed}`,
          run.run_until_success ? '无上限' : run.steps,
          episode?.policy_steps ?? run.observed_policy_steps ?? '—',
          run.cup_lift_summary ? (run.cup_lift_summary.stable_grasp_verified ? '达标，需视频复核' : '未达到') : '—',
          run.left_return_diagnostic ? '左臂分阶段诊断（非完整任务）' : run.task_objective !== 'plate_return' ? '非完整任务评测' :
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
    contact_profile: $('contactProfile').value,
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
