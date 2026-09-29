"""Codegen-only check of the V4.1 final-state output adapter."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--build-dir", default="build_output/v41-serving-output")
    args = parser.parse_args()

    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig

    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.final_output import compile_final_output
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology

    topology = SegmentTopology.for_decode()
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=list(range(topology.world))))
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    _, arguments = compile_final_output(compiler, args.lib_root, topology)
    assert len(arguments) == 9
    print("CODEGEN PASS: final residual/pre_mix -> HC head -> RMSNorm -> LM head -> greedy")


if __name__ == "__main__":
    main()
