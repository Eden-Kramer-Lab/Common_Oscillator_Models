# Common Oscillator Models

Code repository for all of the methods described and the analysis performed in Switching Functional Network Models of Oscillatory Brain Dynamics,*The 56th Asilomar Conference on Signals, Systems & Computer, IEEE*, 2022.

The detailed explantion of the code is at the top of each Matlab script file. Below is a brief description of all scripts in this repo.

* core_functions
  * skf.m -- switching Kalman filter
  * smoother.m -- switching RTS smoother
  * em_B.m -- EM on B matrices
  
* single_rhythm_model
  * single_rhythm_model.m -- propofol example of implementing the common oscillator model with a single rhythm (alpha)
  
* multiple_rhythms_model
  * multi_rhythms_model.m -- propofol example of implementing the common oscillator model with multiple rhythms (alpha + slow wave)
  * std_kf.m -- standard Kalman filter (used for EM on B during awake & unconscious periods)
  * std_smth.m -- standard RTS smoother (used for EM on B during awake & unconscious periods)
  * kf_em_B.m -- EM on B matrices without switching components
 
* plotting
  * plt_B_single.m -- plot the estimated B matrices for the single rhythm model
  * plt_B_multi.m -- plot the estimated B matrices for the multiple rhythms model
  * plt_dosage_bhvr_SW.m -- plot the propofol dosage, behavioral responses, estimated switching states
  * helper_functions 
    * plt_funcs.m -- create the scalp layout of the electrodes
    * adjust_sw.m -- adjust the estimated switching states for removed segments
