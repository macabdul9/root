# Harbor smoke task

The smallest task that exercises the whole chain: Harbor's orchestration, a
Singularity environment, `root.harbor:RootAgent` driving commands into it, and a
verifier writing a reward. Use it before spending on real benchmark images.

```bash
cd containers/harbor-smoke
apptainer build --fakeroot env.sif env.def
sed -i "s|REPLACED_BY_README_STEP|$PWD/env.sif|" task.toml
cd ../.. && uv run harbor run -p containers/harbor-smoke \
  --agent root.harbor:RootAgent --model lfm2-350m --env singularity -k 1
```

`--agent nop` should score 0 and `--agent oracle` is not provided here, so a 0
from `nop` and a populated `agent/root-transcript.txt` from root together mean
the chain works. On LFM2.5-350M the task is not reliably solved: it reaches for
`echo 42 >> /app/answer.txt`, appending where the task says replace, and then
reports success anyway.
