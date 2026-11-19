# V4.1 DSpark terminal prefill

Terminal prefill prepares the first proposed tokens and the draft KV state
used by the next decode step. Serving now binds the completed main prefill
outputs to the Lib bootstrap composite. Each rank submits one L2 program for
this handoff. The main prefill and the recurrent decode have separate entries.

The Lib update also changes the draft projection to packed MXFP8 and divides
compressed C2A pages into 64 storage rows for 128 logical tokens. The serving
adapter binds the projection scales, drafter state and compact page views.
Unit tests cover crossing the C2A page boundary and sharing the packed pool
with C1A pages.

The four-rank synthetic bootstrap comparison on Lib base 873c2452 produced
identical request state, draft IDs, confidence values and draft KV. Median
bootstrap time across four warmed measurements was 19.92 ms for the five
submissions and 12.93 ms for the fused submission. Checkpoint streaming
validation on the rebased Lib is pending. Ordinary K0 decode requires an
output head alongside the new backbone entry and remains unresolved.
