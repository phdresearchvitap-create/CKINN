# Dated literature search for the novelty statement (28-step protocol, step 25)

Searched 2026-11-21 in PubMed only (Scite was unavailable: no active plan). This log records what was found; it does not establish novelty. A systematic search in Embase, Scopus, IEEE Xplore, ACM Digital Library and arXiv is still needed before any novelty claim, because work on constrained or knowledge-informed machine learning is often published outside PubMed.

| Query (PubMed syntax) | Hits |
| --- | --- |
| `(knowledge-informed OR knowledge-guided OR physics-informed) neural network diabetes` | 8 |
| `monotonic constraints machine learning diabetes risk prediction` | 1 |
| `clinical knowledge constraint deep learning type 2 diabetes prediction external validation` | 0 |

## Hits relevant to the CKINN claim

- **Thieu V. Knowledge-guided counterfactual explanations for diabetes risk decision support: a directional intervention taxonomy. *Int J Med Inform* 2026;219:106555.** PMID 42349269. DOI [10.1016/j.ijmedinf.2026.106555](https://doi.org/10.1016/j.ijmedinf.2026.106555). Encodes per-feature clinical directions (immutable, monotonic up, monotonic down, bidirectional, conditional) as constraints on counterfactual generation for an XGBoost diabetes-risk model, with external validation on BRFSS. This is the closest work found: the same idea of signed clinical directions, applied to explanations rather than to training a network. A CKINN paper must cite and distinguish it.
- **Monshizadeh M, Ye Y. MicroKPNN, a prior-knowledge guided interpretable neural network for microbiome-based host phenotype prediction. *Gut Microbes* 2024;16:2302076.** PMID 38214657. DOI [10.1080/19490976.2024.2302076](https://doi.org/10.1080/19490976.2024.2302076). Knowledge built into the network structure (microbial taxonomy), type 2 diabetes among the phenotypes; different data type and mechanism.

## Hits not relevant (recorded for completeness)

Physics-informed models of retinal vessels (PMID 39127778), carotid flow (41687411) and microaneurysm flow (33762307); knowledge-guided NLP for ICU mortality in diabetes (33380338); prior-knowledge attention for nocturnal hypoglycaemia in type 1 diabetes (39696373); a physics-informed network for gestational diabetes in 80 women (42783742); knowledge-guided food-image recognition (40072861).

## What this means for the claim

In this narrow PubMed search, no study trained a neural network for type 2 diabetes risk with a penalty on input derivatives that contradict a clinical sign table, compared it with unconstrained networks on several cohorts, and tested transport. That is a statement about this search, not about the literature. The real-data results of this study (CKINN non-inferior to M2 and M1, not superior; more stable attributions than M2) limit the claim to non-inferiority with clinically consistent behaviour, not to better accuracy.
