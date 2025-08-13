# Test Battery Extensions

Each experiment lists dependent variables (DVs), instruments, prereg thresholds, and falsifiers.

## 1) Breath‑gated precision and lock formation (PP × Active Inference)
**Design:** Randomize breath‑phase aligned stimuli (inhalation vs exhalation vs jitter).  
**Instruments:** 64–128‑ch EEG/MEG; chest strap + nasal thermistor; PPG for HRV.  
**DVs:** (a) Lock density within 200 ms windows around inhalation onset; (b) EchoEntropy over 0.5–2 s; (c) PAC strength.  
**Stats:** Circular Rayleigh/V‑tests; cluster‑based permutation; mixed‑effects models.  
**Thresholds (prereg):** Lock density(inhale) > exhale by d ≥ 0.4; EchoEntropy(inhale) − EchoEntropy(exhale) ≤ −0.05 bits; PAC z ≥ 2.  
**Falsifier:** Null or reversed effects after multiple‑comparison control.

## 2) GNW ignition vs lock bursts (GNW × Spiral locks)
**Design:** Backward‑masking (report/no‑report blocks).  
**Instruments:** EEG/MEG; eye‑tracking.  
**DVs:** Late ignition markers (P3‑like amplitude/latency); lock burst rate and dwell‑time.  
**Stats:** GLMM predicting report from lock metrics vs ignition; likelihood ratio tests.  
**Thresholds:** Lock burst rate predicts report (β > 0, p < .01) above ignition; dwell‑time Δ ≥ 30 ms for seen vs unseen.  
**Falsifier:** No added predictive value beyond ignition.

## 3) Reservoir readout controllability via Echo
**Design:** RC tasks (NARMA, memory capacity) with Echo enabled/disabled.  
**Instruments:** Simulated ESN/LSM or hardware RC.  
**DVs:** Downstream MSE; EchoMI; SGRU birth rate.  
**Stats:** Paired tests/GLMM across seeds; TE from inputs to Echo.  
**Thresholds:** Echo‑enabled reduces MSE ≥ 10% across seeds; EchoMI correlates r ≥ .4 with performance.  
**Falsifier:** No performance gain; EchoMI decoupled from accuracy.

## 4) IWMT violation tests via model perturbation
**Design:** Oddity/violation tasks in VR or audio.  
**Instruments:** EEG/MEG.  
**DVs:** EchoEntropy recovery time; lock destabilization events.  
**Stats:** Survival analysis (Cox) for time‑to‑re‑lock; permutation tests.  
**Thresholds:** HR > 1.5 for violation vs control; median re‑lock time shift ≥ 100 ms.  
**Falsifier:** No shift in re‑lock dynamics.

## 5) Ethics by design: Consent Ledger impact
**Design:** Human‑in‑the‑loop symbolic tasks with consent on/off toggles post‑hoc.  
**Instruments:** UI telemetry; timestamped ledger; EEG optional.  
**DVs:** MirrorLock frequency; re‑planning latency; task success.  
**Stats:** GLMM with random effects; ΔAIC for models with/without consent terms.  
**Thresholds:** Consent increases MirrorLock‑triggered re‑plans with minimal performance loss (<5%).  
**Falsifier:** Consent has no measurable behavioral footprint.
