# Lab 5: Frameworks and automatic differentiation

Vladyslav Ryzhyk, 72348

A scalar reverse-mode autograd engine (`Value`), verified against PyTorch and numerical gradients, used to train a 2→8→1 model on make_moons, plus a PyTorch study of graphs, eager vs compiled execution and tensor contiguity.

## Run

```bash
pip install -r requirements.txt
cd src
python run_all.py        # ~3 min on CPU, writes everything to ../results
```

Individual steps: `python gradcheck.py`, `python train_moons.py`, `python torch_inspect.py`.

## Layout

```
lab05/
├── results/
│   ├── gradcheck.md / .json     task 4: engine vs PyTorch vs numerical
│   ├── moons_timing.json        task 5: timings (median of 5), loss, accuracy
│   ├── moons_training.png       task 5: loss curves + decision boundary
│   ├── torch_graph.txt          task 6: autograd / FX / TorchScript graphs, tensor metadata
│   ├── compile_timing.json      task 6: eager vs jit.trace vs torch.compile
│   ├── contiguity.json          task 6: contiguous vs non-contiguous op timings
│   ├── summary.json
│   └── run.log
├── src/
│   ├── config.yaml              all hyperparameters, seed 42
│   ├── engine.py                tasks 1-3: Value node, ops, topological backward
│   ├── nn.py                    MLP on Value + BCE
│   ├── gradcheck.py             task 4
│   ├── train_moons.py           task 5
│   ├── torch_inspect.py         task 6
│   ├── common.py                config, logging, seeding helpers
│   ├── run_all.py               one-command entry point
│   └── report.md                full write-up
├── README.md
└── requirements.txt
```

See `src/report.md` for results and the note on when compilation helps.
