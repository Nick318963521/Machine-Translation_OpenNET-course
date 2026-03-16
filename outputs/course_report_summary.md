# Course Project Result Summary

## BLEU
- Reference: E:\a_DA\Machine-Translation_OpenNET\data\processed\test.tgt
- Baseline prediction: E:\a_DA\Machine-Translation_OpenNET\outputs\baseline_test_pred.txt
- Finetuned prediction: E:\a_DA\Machine-Translation_OpenNET\outputs\finetuned_test_pred.txt
- Baseline BLEU: 0.0000
- Finetuned BLEU: 0.0000
- BLEU improvement: 0.00%

## Benchmark Highlights
- Fastest model_compare run: model_type=finetuned, sent_per_sec=1.4304
- Best throughput in batch_compare: batch_size=8, sent_per_sec=1.4058
- Fastest beam setting in beam_compare: beam_size=5, sent_per_sec=1.4428
- Highest cache hit in cache_compare: cache_enabled=True, cache_hits=18
- Throughput change (finetuned vs baseline, model_compare): 0.47%
- Cache speedup (cache_on vs cache_off): 55449681.41%

## Tips for Course Report
- Explain whether finetuning improved BLEU and by how much.
- Discuss trade-offs between beam size and speed.
- Discuss why cache improves repeated-sentence throughput.
- Use metrics table from: E:\a_DA\Machine-Translation_OpenNET\outputs\course_report_metrics.csv
