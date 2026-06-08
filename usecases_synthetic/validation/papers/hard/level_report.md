# Validation report - papers / hard

_Generated at 2026-06-08T10:37:10.978941+00:00_

- domain: `papers`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_papers.yaml@a3dd1a8b0e71`, fusion=`fusion_committee_papers.yaml@c898dae65e10`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.4945 | 0.8340 | -0.3395 |
| norm | macro_f1 | 0.5418 | 0.7608 | -0.2189 |
| fusion | overall_accuracy | 0.4351 | 0.6104 | -0.1752 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.1860 | 0.9714 | -0.7854 |
| duplicate_majority | 0.7143 | 0.9697 | -0.2554 |
| embedding_sbert | 0.5000 | 0.7778 | -0.2778 |
| instance_tf_cosine | 0.4060 | 0.4151 | -0.0091 |
| label_jw | 0.2617 | 0.8608 | -0.5991 |
| llm_openai | 0.9286 | 0.9714 | -0.0429 |
| magneto_slm_llm | 0.4648 | 0.8718 | -0.4070 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5681 | 0.8161 | -0.2479 |
| passthrough | 0.5682 | 0.8257 | -0.2575 |
| rule_per_attribute_optimal | 0.4892 | 0.6405 | -0.1513 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.4222 | 0.6240 | -0.2018 |
| casefusion_only | 0.3612 | 0.4833 | -0.1220 |
| fusionquery_only | 0.3888 | 0.4833 | -0.0945 |
| llm_only | 0.4469 | 0.5896 | -0.1427 |
| ltm_only | 0.3760 | 0.5364 | -0.1604 |
| prefer_higher_trust_only | 0.4144 | 0.6181 | -0.2037 |
| pydi_per_attribute_optimal | 0.3632 | 0.5404 | -0.1772 |
| truthfinder_only | 0.3848 | 0.5610 | -0.1762 |
| voting_only | 0.3996 | 0.6102 | -0.2106 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| authors | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| first_page | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| issue | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| journal | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keywords | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| last_page | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| publication_year | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| referenced_works_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| title | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| volume | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
