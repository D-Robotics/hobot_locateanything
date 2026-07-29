# LocateAnything Project Layout

The product has one implementation tree. Compiler source and compiler-side
tools are isolated from the S600 deployment source; there is no compatibility
wrapper or second `main/` tree.

```text
LocateAnything/
├── compiler/                  # OELLM adapters and compiler-side scripts
│   ├── quantize.py            # public prepare/calibrate/build/verify CLI
│   ├── config.yaml            # fixed release profile and workspace paths
│   ├── leap_llm/
│   └── scripts/
│       ├── calibration/       # calibration tensor preparation and activation statistics
│       ├── build/             # internal Vision and Language build wrappers
│       ├── validate/          # contract, pipeline, and task validation
│       └── common/            # shared compiler-side utilities
├── deploy/                    # S600 C++ runners, Python CLI, tokenizer
├── src/oe_locateanything/     # shared path and project helpers
├── tests/                     # host-side regression tests
├── docs/                      # active technical documentation
├── assets/                    # README media
└── workspace/                 # ignored models, artifacts, data, logs, samples
```

Only these seven ownership directories are kept. `compiler/` contains the
shared OELLM operators, the two model implementations used here, and the tools
that produce or validate compiler artifacts. `deploy/` contains only the S600
runtime, CLI, tokenizer, and deployment entrypoints. `workspace/` has no
pre-created child tree: build, calibration, evaluation, and runtime commands
create only the directories they actually write.

The public compiler workflow is intentionally limited to four commands:

```bash
python compiler/quantize.py prepare
python compiler/quantize.py calibrate
python compiler/quantize.py build
python compiler/quantize.py verify
```

Files below `compiler/scripts/` are implementation details used by this
orchestrator. Dataset curation and focused diagnostic tools may be invoked by
maintainers, but they are not separate release build entrypoints.

## Path contract

Every executable derives the product root from its own file location. Generated
state defaults to `workspace/`; deployment environments may override locations
with `LA_WORKSPACE`, `LA_MODEL_ROOT`, `LA_BUILD_ROOT`, `LA_ARTIFACT_ROOT`,
`LA_CALIBRATION_ROOT`, `LA_EVALUATION_ROOT`, and `LA_RUN_ROOT`. A packaged
compiler or deployment tree may be selected with `LA_COMPILER_ROOT` or
`LA_DEPLOY_ROOT`; `LA_COMPILER_SCRIPTS_ROOT` is available only when the compiler
tools are installed separately. `LA_MODEL_PATH` and `LA_UPSTREAM_SOURCE` may
select host-specific checkpoint and upstream source locations. Source paths
remain product-relative and do not depend on `$HOME`, a user name, or the name
of the outer checkout.

Compiler candidates are written to `workspace/builds/` and may be relocated
with `LA_BUILD_ROOT`. Only artifacts that
have passed the release checks are promoted to `workspace/artifacts/release/`
for deployment; `LA_ARTIFACT_ROOT` relocates that published-artifact tree.
Keeping these roots separate prevents an interrupted HBDK build from replacing
the model used by the board runtime.
