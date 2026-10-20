# tools/

The operator scripts that used to live in `tools/ops/` have moved to
`typography_engine/ops/` on `pet-engine-drape-tone-tiers`.

They were split across two branches and two directories, so reaching them on
the VPS meant a hand-copy rather than a pull. All five deployments track
`pet-engine-drape-tone-tiers`, so that is where they belong.

    git fetch github pet-engine-drape-tone-tiers
    git -C /root/<tree> pull github pet-engine-drape-tone-tiers
    ls typography_engine/ops/

See `typography_engine/ops/README.md` for what each script does.
