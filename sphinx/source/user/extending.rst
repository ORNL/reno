Extending Reno
##############


Custom operations
=================

(show an example of what a current op class looks like)


PyMC transpiling
================

(maybe this just gets covered in the bayes page?)


Simulation timestep control
===========================

Default way to run is via __call__()

models under the hood use a simulator generator,
can manually step through step by step
(allows doing extra steps/reporting/analysis between timesteps, or having
multiple models going that operate on different timescales etc.)
