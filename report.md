模型部署命令：
```shell
CUDA_LAUNCH_BLOCKING=1 vllm serve /root/autodl-tmp/LLaMA-Factory/saves/qwen3-4b-finance-merged/ \
  --served-model-name qwen3-4b-finance \
  --max-model-len 4096 \
  --max-num-seqs 85 \
  --port 8000 \
  --enforce-eager \
  --enable-prefix-caching
```

微调后的模型评估结果

general_mcq report table:

┌──────────────────┬─────────────┬──────────┬──────────────┬───────┬─────────┬─────────┐
│ Model            │ Dataset     │ Metric   │ Subset       │   Num │   Score │ Cat.0   │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ mmlu_test    │   500 │   0.738 │ default │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ ceval_test   │   500 │   0.722 │ default │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ finance_test │  1000 │   0.722 │ default │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ OVERALL      │  2000 │   0.726 │ -       │
└──────────────────┴─────────────┴──────────┴──────────────┴───────┴─────────┴─────────┘

2026-05-15 16:23:16 - evalscope - INFO: Skipping report analysis (`analysis_report=False`).
2026-05-15 16:23:16 - evalscope - INFO: Dump report to: ./outputs/20260515_154801/reports/qwen3-4b-finance/general_mcq.json

2026-05-15 16:23:16 - evalscope - INFO:
general_mcq perf table:
Model             Dataset        Num    Avg Lat  Avg TTFT    Avg TPOT      Avg Thpt    Avg In    Avg Out
                                            (s)  (ms)        (ms)           (tok/s)       Tok        Tok
----------------  -----------  -----  ---------  ----------  ----------  ----------  --------  ---------
qwen3-4b-finance  general_mcq   2000    81.1528  -           -                 14.1   143.329    1143.98

2026-05-15 16:23:16 - evalscope - INFO: Benchmark general_mcq evaluation finished.
2026-05-15 16:23:16 - evalscope - INFO: Running[eval] 100%| 1/1 [Elapsed: 35:15 < Remaining: 00:00, 2115.36s/benchmark]
Running[eval]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [35:15<00:00, 2115.36s/benchmark]
2026-05-15 16:23:16 - evalscope - INFO: Overall report table:
┌──────────────────┬─────────────┬──────────┬──────────────┬───────┬─────────┬─────────┐
│ Model            │ Dataset     │ Metric   │ Subset       │   Num │   Score │ Cat.0   │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ mmlu_test    │   500 │   0.738 │ default │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ ceval_test   │   500 │   0.722 │ default │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ finance_test │  1000 │   0.722 │ default │
├──────────────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b-finance │ general_mcq │ mean_acc │ OVERALL      │  2000 │   0.726 │ -       │
└──────────────────┴─────────────┴──────────┴──────────────┴───────┴─────────┴─────────┘

2026-05-15 16:23:16 - evalscope - INFO: Overall perf table:
Model             Dataset        Num    Avg Lat  Avg TTFT    Avg TPOT      Avg Thpt    Avg In    Avg Out
                                            (s)  (ms)        (ms)           (tok/s)       Tok        Tok
----------------  -----------  -----  ---------  ----------  ----------  ----------  --------  ---------
qwen3-4b-finance  general_mcq   2000    81.1528  -           -                 14.1   143.329    1143.98

2026-05-15 16:23:16 - evalscope - INFO: HTML report generated: /root/autodl-tmp/data/outputs/20260515_154801/reports/report.html
2026-05-15 16:23:17 - evalscope - INFO: Finished evaluation for qwen3-4b-finance on ['general_mcq']
2026-05-15 16:23:17 - evalscope - INFO: Output directory: ./outputs/20260515_154801

微调前模型的评估结果

模型部署命令：
```shell
CUDA_LAUNCH_BLOCKING=1 vllm serve /root/autodl-tmp/models/   --served-model-name qwen3-4b   --max-model-len 4096   --max-num-seqs 85   --port 8000   --enforce-eager   --enable-prefix-caching

```



general_mcq report table:
┌──────────┬─────────────┬──────────┬──────────────┬───────┬─────────┬─────────┐
│ Model    │ Dataset     │ Metric   │ Subset       │   Num │   Score │ Cat.0   │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ mmlu_test    │   500 │  0.744  │ default │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ ceval_test   │   500 │  0.732  │ default │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ finance_test │  1000 │  0.749  │ default │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ OVERALL      │  2000 │  0.7435 │ -       │
└──────────┴─────────────┴──────────┴──────────────┴───────┴─────────┴─────────┘

2026-05-15 17:11:07 - evalscope - INFO: Skipping report analysis (`analysis_report=False`).
2026-05-15 17:11:07 - evalscope - INFO: Dump report to: ./outputs/20260515_163218/reports/qwen3-4b/general_mcq.json

2026-05-15 17:11:07 - evalscope - INFO:
general_mcq perf table:
Model     Dataset        Num    Avg Lat  Avg TTFT    Avg TPOT      Avg Thpt    Avg In    Avg Out
                                    (s)  (ms)        (ms)           (tok/s)       Tok        Tok
--------  -----------  -----  ---------  ----------  ----------  ----------  --------  ---------
qwen3-4b  general_mcq   2000    90.6602  -           -                14.07   143.329    1275.88

2026-05-15 17:11:07 - evalscope - INFO: Benchmark general_mcq evaluation finished.
2026-05-15 17:11:07 - evalscope - INFO: Running[eval] 100%| 1/1 [Elapsed: 38:48 < Remaining: 00:00, 2328.60s/benchmark]
Running[eval]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [38:48<00:00, 2328.60s/benchmark]
2026-05-15 17:11:07 - evalscope - INFO: Overall report table:
┌──────────┬─────────────┬──────────┬──────────────┬───────┬─────────┬─────────┐
│ Model    │ Dataset     │ Metric   │ Subset       │   Num │   Score │ Cat.0   │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ mmlu_test    │   500 │  0.744  │ default │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ ceval_test   │   500 │  0.732  │ default │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ finance_test │  1000 │  0.749  │ default │
├──────────┼─────────────┼──────────┼──────────────┼───────┼─────────┼─────────┤
│ qwen3-4b │ general_mcq │ mean_acc │ OVERALL      │  2000 │  0.7435 │ -       │
└──────────┴─────────────┴──────────┴──────────────┴───────┴─────────┴─────────┘

2026-05-15 17:11:07 - evalscope - INFO: Overall perf table:
Model     Dataset        Num    Avg Lat  Avg TTFT    Avg TPOT      Avg Thpt    Avg In    Avg Out
                                    (s)  (ms)        (ms)           (tok/s)       Tok        Tok
--------  -----------  -----  ---------  ----------  ----------  ----------  --------  ---------
qwen3-4b  general_mcq   2000    90.6602  -           -                14.07   143.329    1275.88

2026-05-15 17:11:07 - evalscope - INFO: HTML report generated: /root/autodl-tmp/data/outputs/20260515_163218/reports/report.html
2026-05-15 17:11:07 - evalscope - INFO: Finished evaluation for qwen3-4b on ['general_mcq']
2026-05-15 17:11:07 - evalscope - INFO: Output directory: ./outputs/20260515_163218


微调又翻车了。