# zag
For when zigging is not quite enough

A framework for deep learning operations that are set up to be as fast as possible while being bounded in memory. Any memory gotchas are to be surfaced at 
comptime - e.g. no hidden allocations so one shouldn't run into random out of memory errors at runtime. This project structure is inspired by the tigerstyle
rules - so lots of asserts and minimal heap allocations. The goal is to separate the control plane from the data plane as much as possible such that the
matrix and activation function definitions are symbolically represented and map to a specific choreography of extremely fast pure functions that manipulate
the data. An extension of this goal, and the primary motivation for this framework, is to support gpu and/or cpu data planes while maintaining the same
control plane
