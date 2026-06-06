# Validation report - games / easy

_Generated at 2026-06-05T22:45:23.045423+00:00_

- domain: `games`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@25731d7e9982`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`, fusion=`fusion_committee_games.yaml@fd979439ba3d`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.8709 | 0.7478 | 0.1231 |
| norm | macro_f1 | 0.8854 | 0.8971 | -0.0118 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.6275 | 0.6093 | 0.0182 |
| fusion | overall_accuracy | 0.7114 | 0.7202 | -0.0089 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.9811 | 0.7442 | 0.2369 |
| duplicate_majority | 0.8000 | 0.8444 | -0.0444 |
| embedding_sbert | 0.7547 | 0.6400 | 0.1147 |
| instance_tf_cosine | 0.6531 | 0.7200 | -0.0669 |
| label_jw | 0.9811 | 0.3429 | 0.6383 |
| llm_openai | 0.9811 | 1.0000 | -0.0189 |
| magneto_slm_llm | 0.9455 | 0.9434 | 0.0021 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8476 | 0.8547 | -0.0070 |
| passthrough | 0.9608 | 0.9809 | -0.0200 |
| rule_per_attribute_optimal | 0.8476 | 0.8559 | -0.0083 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9429 | 0.9528 | -0.0100 | 0.9973 | 0.9963 |
| embedding_blocker | 0.9429 | 0.9387 | 0.0042 | 0.9973 | 0.9963 |
| sc_block | 0.9571 | 0.9434 | 0.0137 | 0.9973 | 0.9963 |
| sorted_neighbourhood_blocker | 0.9194 | 0.9155 | 0.0039 | 0.9990 | 0.9988 |
| standard_blocker | 0.9905 | 0.9858 | 0.0046 | 0.9987 | 0.9988 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9633 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.5693 | 0.6071 | -0.0378 | 0.5693 | 0.6071 | 0.5693 | 0.6071 |
| ditto_plm | 0.7104 | 0.7155 | -0.0051 | 0.7184 | 0.7155 | 0.7184 | 0.7155 |
| llm_matcher | 0.5755 | 0.6054 | -0.0298 | 0.5755 | 0.6054 | 0.5755 | 0.6054 |
| magellan | 0.6547 | 0.5092 | 0.1455 | 0.6675 | 0.5092 | 0.6675 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.6782 | 0.7143 | -0.0361 |
| dbpedia_sales | ditto_plm | 0.8402 | 0.8122 | 0.0280 |
| dbpedia_sales | llm_matcher | 0.6971 | 0.7143 | -0.0171 |
| dbpedia_sales | magellan | 0.8430 | 0.5934 | 0.2496 |
| metacritic_dbpedia | comem | 0.4604 | 0.5000 | -0.0396 |
| metacritic_dbpedia | ditto_plm | 0.5806 | 0.6188 | -0.0381 |
| metacritic_dbpedia | llm_matcher | 0.4539 | 0.4965 | -0.0426 |
| metacritic_dbpedia | magellan | 0.4663 | 0.4250 | 0.0413 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6363 | 0.6485 | -0.0122 |
| casefusion_only | 0.6919 | 0.6986 | -0.0067 |
| fusionquery_only | 0.7063 | 0.7208 | -0.0145 |
| llm_only | 0.6073 | 0.6140 | -0.0067 |
| ltm_only | 0.6607 | 0.6796 | -0.0189 |
| prefer_higher_trust_only | 0.7119 | 0.7186 | -0.0067 |
| pydi_per_attribute_optimal | 0.6874 | 0.6986 | -0.0111 |
| truthfinder_only | 0.7075 | 0.7208 | -0.0133 |
| voting_only | 0.6307 | 0.6440 | -0.0133 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| ESRB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| criticScore | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| developer | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| genres | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| platform | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| publisher | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| releaseYear | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| userScore | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
