# space-model

## Experiments

```
epochs = 5
batch_size = 256
max_seq_len = 256
learning_rate = 2e-4
max_grad_norm = 1000
n_latent = 3
```

### Results Table:
| Metric           | Bert-base  | Space-model |
|------------------|------------|-------------|
| Train Params     | 1538       | 4622        |
| Loss             | 0.5793     | **0.8092**  |
| Accuracy         | 0.724      | **0.8175**  |
| F1-score (macro) | 0.7213     | **0.8175**  |
| Precision        | 0.6841     | **0.8162**  |
| Recall           | **0.8349** | 0.8195      |
