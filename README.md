# Common Oscillator Models

Code repository for all of the methods described and the analysis performed in *Switching Functional Network Models of Oscillatory Brain Dynamics*.

The detailed explantion of the code is at the top of each Matlab script file. Below is a quick glance of all scripts in this repo.

* core_functions
  * skf.m -- switching Kalman filter
  * smoother.m -- switching RTS smoother
  * em_B.m -- EM on B matrices
  
* single_rhythm_model
  * single_rhythm_model.m -- implement the common oscillator model on the propofol data (single rhythm - alpha)
  
* multiple_rhythms_model
  * multi_rhythms_model.m -- implement the common oscillator model on the propofol data (multiple rhythm - alpha + slow wave)
  * std_kf.m -- standard Kalman filter (used for EM on B for awake & unconscious periods separately)
  * std_smth.m -- standard RTS smoother (used for EM on B for awake & unconscious periods separately)
  * kf_em_B.m -- EM on B matrices without switching components
 
* plotting
  * plt_B_single.m -- plot the estimated B matrices for the single rhythm model
  * plt_B_multi.m -- plot the estimated B matrices for the multiple rhythms model
  * plt_dosage_bhvr_SW.m -- plot the propofol dosage, behavioral responses, estimated switching states
  * helper_functions 
   * plt_funcs.m -- create the layout of the electrodes
   * adjust_sw.m -- adjust the estimated switching states for removed segments
