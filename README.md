# MMI_Algorithmen_WS25_26

This repository contains the code, notebooks, result tables, and paper for a project on the relationship between **annotation agreement** and **dialogue coherence** in the **MultiWOZ** corpus.

The project asks whether disagreement between annotation layers can be interpreted as an indirect signal of pragmatic incoherence. To address this question, the repository combines two analytical branches:

1. **Annotation-overlap analysis**  
   MultiWOZ annotations are compared across annotation layers using overlap metrics such as **BLEU-1**, **BLEU-4**, and **Jaccard**, under different matching regimes.

2. **Grice-inspired coherence analysis**  
   Dialogue turns are evaluated with an LLM-based coherence judge inspired by the conversational maxims of **Quality**, **Quantity**, **Relation**, and **Manner**.

The resulting overlap and coherence signals are then related at both **dialogue level** and **turn level**.

---

## Repository structure

```text
MMI_Algorithmen_WS25_26/
├── data/                         # local data directory (large files may be excluded from Git tracking)
├── notebooks/                    # analysis notebooks, scripts, and result tables
├── results/                      # local result directory (large files may be excluded from Git tracking)
├── paper/                        # project paper / draft PDF
├── MultiWOZ_pipeline.png         # pipeline overview figure
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Main files

### Notebooks
The `notebooks/` directory contains the main analysis pipeline, including:

- `01_prepare_common_dialogs.ipynb`  
  Preparation and alignment of the shared dialogue set.

- `annotate_multiwoz_weiß.ipynb`  
  Re-annotation workflow for MultiWOZ.

- `02_agreement_bleu_jaccard.ipynb`  
  Computation of annotation-overlap metrics such as BLEU and Jaccard.

- `03_grice_tests.ipynb`  
  Grice-based coherence evaluation and related testing.

- `turn_level_bleu_jaccard_vs_grice.ipynb`  
  Turn-level comparison of overlap metrics and coherence scores.

- `multiwoz_result_plots_from_csv.ipynb`  
  Visualization of the final result tables used in the paper.

- `merged_final_df.ipynb`  
  Merging and preparation of final analysis tables.

- `Grice_vs_bleu_jacc.ipynb`  
  Additional comparison notebook for overlap and coherence analyses.

### Scripts
- `notebooks/grice_judge_corpus.py`  
  Python script for running the Grice-inspired coherence judging pipeline.

### Final result tables
The repository includes result files that are directly relevant to the paper, including:

- `notebooks/grice_dialog_level.csv`
- `notebooks/grice_turn_level.csv`
- `notebooks/merged_dialog_analysis.csv`
- `notebooks/spearman_results.csv`
- `notebooks/grice_slim.csv`

These files contain the final exported tables used for reporting and plotting the main findings.

---

## Workflow overview

The overall workflow is shown in `MultiWOZ_pipeline.png`.

In simplified form, the pipeline is:

1. prepare a shared set of MultiWOZ dialogues  
2. generate or collect re-annotations  
3. compute overlap scores between annotation layers  
4. evaluate coherence with a Grice-inspired judge  
5. aggregate coherence scores to turn and dialogue level  
6. merge overlap and coherence results  
7. compute correlations, grouped comparisons, and plots

---

## Paper

The current paper draft is available here:

- [`paper/Annotation_Agreement_to_Coherence.pdf`](paper/Annotation_Agreement_to_Coherence.pdf)

---

## Notes on data and reproducibility

Some raw data files, intermediate outputs, and large result folders are intentionally not tracked in the public repository in order to keep the repository lightweight and manageable.

The repository is therefore designed to provide:

- the main analysis notebooks
- the judging script
- the final result tables cited in the paper
- the pipeline figure
- the paper PDF

If additional raw or intermediate files are required for full reproduction, they can be regenerated through the notebook pipeline, provided the necessary local input data are available.

---

## Research focus

The project investigates whether symbolic agreement and pragmatic coherence align in a stable way. More specifically, it asks:

- whether lower annotation overlap corresponds to lower dialogue coherence
- whether this relationship depends on the overlap metric used
- whether turn-level and dialogue-level analyses produce different conclusions

The results suggest that annotation agreement and pragmatic coherence should **not** be treated as interchangeable indicators of dialogue quality.

---

## License / usage

This repository is intended for academic and research use.  
Please cite the paper or repository appropriately if you reuse parts of the analysis or structure.

---

## Author

Repository maintained by **Benedikt Weiß**.
