/* Validate before replacing a loaded dashboard. Also usable by the Node checks. */
function validateDashboard(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || !Array.isArray(value.subjects))
    throw new Error('Expected a dashboard object with a subjects array.');
  const ids = new Set();
  for (const subject of value.subjects) {
    if (!subject || typeof subject.subject !== 'string' || ids.has(subject.subject) || !Array.isArray(subject.probes))
      throw new Error('Each subject needs a unique identifier and a probes array.');
    ids.add(subject.subject);
    for (const key of ['mean_ratio','mean_pac','mean_concentration','mean_mind_wandering'])
      if (subject[key] != null && !Number.isFinite(subject[key])) throw new Error('Invalid subject mean.');
    for (const probe of subject.probes) {
      if (!probe || typeof probe !== 'object' || !Number.isFinite(probe.probe_index))
        throw new Error('Each probe needs a numeric probe_index.');
      for (const key of ['probe_onset_s','minimal_ratio','minimal_pac','concentration_rating','depth_rating','mind_wandering_rating','tiredness_rating'])
        if (probe[key] != null && !Number.isFinite(probe[key])) throw new Error('Invalid probe value: ' + key);
    }
  }
  function table(rows, kind) {
    if (!Array.isArray(rows) || rows.some(row => !row || typeof row !== 'object' || Array.isArray(row)))
      throw new Error('Result tables must contain objects.');
    const fields = {model_comparison:['model','score'],electrode_minimalism:['electrode_set','score'],shap_importance:['feature','importance']}[kind];
    if (fields) for (const row of rows) {
      if (typeof row[fields[0]] !== 'string' || (row[fields[1]] != null && !Number.isFinite(row[fields[1]])))
        throw new Error('Invalid result row: ' + kind);
    }
  }
  for (const key of ['model_comparison','electrode_minimalism','shap_importance'])
    if (value[key] != null) table(value[key],key);
  if (value.tasks != null) {
    if (typeof value.tasks !== 'object' || Array.isArray(value.tasks)) throw new Error('Invalid tasks.');
    for (const task of Object.values(value.tasks)) {
      if (!task || typeof task !== 'object') throw new Error('Invalid task.');
      for (const key of ['model_comparison','electrode_minimalism','shap_importance']) if (task[key] != null) table(task[key],key);
    }
  }
  if (value.statistics != null) {
    if (typeof value.statistics !== 'object' || Array.isArray(value.statistics)) throw new Error('Invalid statistics.');
    Object.values(value.statistics).forEach(rows=>table(rows));
  }
  if (value.group_summary != null) {
    if (typeof value.group_summary !== 'object' || Array.isArray(value.group_summary)) throw new Error('Invalid group summary.');
    for (const group of Object.values(value.group_summary)) {
      if (!group || typeof group !== 'object') throw new Error('Invalid group.');
      for (const item of Object.values(group)) if (item != null && !Number.isFinite(item)) throw new Error('Invalid group value.');
    }
  }
  function finiteNumbers(obj) {
    if (typeof obj === 'number' && !Number.isFinite(obj)) throw new Error('Nonfinite number in dashboard.');
    if (obj && typeof obj === 'object') Object.values(obj).forEach(finiteNumbers);
  }
  finiteNumbers(value);
  return {...value, n_subjects: value.subjects.length,
    n_probes: value.subjects.reduce((n, subject) => n + subject.probes.length, 0)};
}
if (typeof module !== 'undefined') module.exports = {validateDashboard};
