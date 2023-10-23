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

### Bert Results:
![loss](https://github.com/StepanTita/space-model/assets/44279105/a1dc0fa1-d115-4313-ae8f-1c6546fe7db1)
![accuracy](https://github.com/StepanTita/space-model/assets/44279105/e1185270-fb45-4deb-a97b-5df9517b420b)
![f1_macro](https://github.com/StepanTita/space-model/assets/44279105/d5510dd0-00bf-4f58-806c-8185a9b47ada)
![recall](https://github.com/StepanTita/space-model/assets/44279105/c345d0d4-97b1-4968-b420-df4642525490)
![precision](https://github.com/StepanTita/space-model/assets/44279105/6cc6bc2c-9a8f-457d-b23d-8ae87bb6c4fd)

### Space-model Results:
![loss](https://github.com/StepanTita/space-model/assets/44279105/89e6a322-d7b4-4f86-8d4f-178e11723b0f)
![accuracy](https://github.com/StepanTita/space-model/assets/44279105/35e96447-8400-454b-9f83-76602cd2586a)
![f1_macro](https://github.com/StepanTita/space-model/assets/44279105/c4f43845-b74f-4612-b6ea-6096cfdfc807)
![recall](https://github.com/StepanTita/space-model/assets/44279105/b183f72c-4090-43b2-baf5-d6ff8554ba2a)
![precision](https://github.com/StepanTita/space-model/assets/44279105/15304368-7de2-43ce-8143-7bf171fdb364)

