# Validation report - games / hard

_Generated at 2026-06-05T05:11:11.210896+00:00_

- domain: `games`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@25731d7e9982`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`, fusion=`fusion_committee_games.yaml@fd979439ba3d`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6930 | 0.7478 | -0.0548 |
| norm | macro_f1 | 0.8398 | 0.8971 | -0.0574 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.5563 | 0.6093 | -0.0530 |
| fusion | overall_accuracy | 0.5847 | 0.7202 | -0.1356 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.5641 | 0.7442 | -0.1801 |
| duplicate_majority | 0.8261 | 0.8444 | -0.0184 |
| embedding_sbert | 0.6809 | 0.6400 | 0.0409 |
| instance_tf_cosine | 0.6667 | 0.7200 | -0.0533 |
| label_jw | 0.3030 | 0.3429 | -0.0398 |
| llm_openai | 0.9630 | 1.0000 | -0.0370 |
| magneto_slm_llm | 0.8475 | 0.9434 | -0.0959 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8339 | 0.8547 | -0.0208 |
| passthrough | 0.8516 | 0.9809 | -0.1293 |
| rule_per_attribute_optimal | 0.8338 | 0.8559 | -0.0221 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.7358 | 0.9528 | -0.2170 | 0.9959 | 0.9963 |
| embedding_blocker | 0.7406 | 0.9387 | -0.1981 | 0.9959 | 0.9963 |
| sc_block | 0.7358 | 0.9434 | -0.2075 | 0.9959 | 0.9963 |
| sorted_neighbourhood_blocker | 0.6821 | 0.9155 | -0.2334 | 0.9988 | 0.9988 |
| standard_blocker | 0.7371 | 0.9858 | -0.2488 | 0.9981 | 0.9988 |
| token_blocker | 0.7736 | 1.0000 | -0.2264 | 0.9563 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.4669 | 0.6071 | -0.1402 | 0.4093 | 0.6071 | 0.4669 | 0.6071 |
| ditto_plm | 0.6147 | 0.7155 | -0.1008 | 0.4505 | 0.7155 | 0.4866 | 0.7155 |
| llm_matcher | 0.4851 | 0.6054 | -0.1202 | 0.4353 | 0.6054 | 0.4851 | 0.6054 |
| magellan | 0.6585 | 0.5092 | 0.1493 | 0.5883 | 0.5092 | 0.6297 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.6919 | 0.7143 | -0.0224 |
| dbpedia_sales | ditto_plm | 0.7876 | 0.8122 | -0.0246 |
| dbpedia_sales | llm_matcher | 0.7047 | 0.7143 | -0.0096 |
| dbpedia_sales | magellan | 0.8140 | 0.5934 | 0.2205 |
| metacritic_dbpedia | comem | 0.2419 | 0.5000 | -0.2581 |
| metacritic_dbpedia | ditto_plm | 0.4417 | 0.6188 | -0.1771 |
| metacritic_dbpedia | llm_matcher | 0.2656 | 0.4965 | -0.2308 |
| metacritic_dbpedia | magellan | 0.5030 | 0.4250 | 0.0780 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5648 | 0.6485 | -0.0837 |
| casefusion_only | 0.5648 | 0.6986 | -0.1337 |
| fusionquery_only | 0.5847 | 0.7208 | -0.1361 |
| llm_only | 0.5291 | 0.6140 | -0.0849 |
| ltm_only | 0.5807 | 0.6796 | -0.0990 |
| prefer_higher_trust_only | 0.5728 | 0.7186 | -0.1458 |
| pydi_per_attribute_optimal | 0.5542 | 0.6986 | -0.1443 |
| truthfinder_only | 0.5820 | 0.7208 | -0.1388 |
| voting_only | 0.5503 | 0.6440 | -0.0938 |

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
