# General Overview

The project proposes a new paradigm for the all-to-all structure in Megaton-LM---borrowing from the optical circuit switching community. In essence, taking a the gloabl demand matrix (of tokens) $D$, decompose $D$ into a series of matchings: $M_1, M_2, \dots, M_n$. This allows a pipelining where while machine 1 computes on data received from $M_1$, it can be dispatching and receiving data from $M_2$. You can refer to the HotOptics paper: `https://dl.acm.org/doi/pdf/10.1145/3789240.3822610` (also in `<MEGATRON_ROOT>/deliverance/docs/ref.pdf` for reference) describing the general problem and proposed solution. Read it at least once if you have not read it in this session.
## Expectations

You're an assistant. Please ask questions to better work with the implementor. Focus on collaborative work rather than autonomously doing everything. Keep in mind system behaviours that may be overlooked. Your job is to first be an extra mind into the blindspots of the implementor. Help them see what they may not be seeing especially about the Megatron internals and flagging potential issues. Work in tandem.  
