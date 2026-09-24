# ci-69 serving submodule recovery (2026-09-24)

Base: 2830741823687489eeb7a9e02dcee0a1481fb4f4.
The only original dirty change was the pypto-lib gitlink:
e94a0bfbc5b75a8a913e36ce31c37b193bbc4bb2 ->
5a96d70d6fe549c590421adcd6cea734027c6ac3.
The target commit is retrievable from the lib fork.

No serving Python source was dirty. This archives the selected lib version,
not a newly tested compatible serving/lib combination. The issue1275
standalone logs concern individual boundary/transport experiments and do
not establish end-to-end serving correctness for this gitlink.
No new device run was performed during recovery.

Local recovery checks: all pre-commit checks passed. No end-to-end
serving/device validation was performed.
