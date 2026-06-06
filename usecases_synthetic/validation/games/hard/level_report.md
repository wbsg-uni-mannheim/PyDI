# Validation report - games / hard

_Generated at 2026-06-05T23:59:56.926022+00:00_

- domain: `games`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_games.yaml@25731d7e9982`, em_blocking=`em_blocking_committee_games.yaml@c07cf762a7d5`, em_matching=`em_matching_committee_games.yaml@ead92fa3a2a8`, fusion=`fusion_committee_games.yaml@fd979439ba3d`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7092 | 0.7478 | -0.0386 |
| norm | macro_f1 | 0.8333 | 0.8971 | -0.0638 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.7263 | 0.6093 | 0.1170 |
| fusion | overall_accuracy | 0.5860 | 0.7202 | -0.1342 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.5641 | 0.7442 | -0.1801 |
| duplicate_majority | 0.8511 | 0.8444 | 0.0066 |
| embedding_sbert | 0.6531 | 0.6400 | 0.0131 |
| instance_tf_cosine | 0.6667 | 0.7200 | -0.0533 |
| label_jw | 0.3030 | 0.3429 | -0.0398 |
| llm_openai | 0.9811 | 1.0000 | -0.0189 |
| magneto_slm_llm | 0.9455 | 0.9434 | 0.0021 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8086 | 0.8547 | -0.0461 |
| passthrough | 0.8576 | 0.9809 | -0.1233 |
| rule_per_attribute_optimal | 0.8338 | 0.8559 | -0.0221 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9567 | 0.9528 | 0.0039 | 0.9959 | 0.9963 |
| embedding_blocker | 0.9663 | 0.9387 | 0.0277 | 0.9959 | 0.9963 |
| sc_block | 0.9567 | 0.9434 | 0.0133 | 0.9959 | 0.9963 |
| sorted_neighbourhood_blocker | 0.8694 | 0.9155 | -0.0461 | 0.9988 | 0.9988 |
| standard_blocker | 0.9443 | 0.9858 | -0.0416 | 0.9981 | 0.9988 |
| token_blocker | 0.9909 | 1.0000 | -0.0091 | 0.9564 | 0.9530 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.6264 | 0.6071 | 0.0192 | 0.4680 | 0.6071 | 0.6264 | 0.6071 |
| ditto_plm | 0.8227 | 0.7155 | 0.1072 | 0.5256 | 0.7155 | 0.6711 | 0.7155 |
| llm_matcher | 0.6506 | 0.6054 | 0.0452 | 0.5024 | 0.6054 | 0.6506 | 0.6054 |
| magellan | 0.8056 | 0.5092 | 0.2964 | 0.7149 | 0.5092 | 0.7832 | 0.5092 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| dbpedia_sales | comem | 0.6374 | 0.7143 | -0.0769 |
| dbpedia_sales | ditto_plm | 0.8000 | 0.8122 | -0.0122 |
| dbpedia_sales | llm_matcher | 0.6947 | 0.7143 | -0.0195 |
| dbpedia_sales | magellan | 0.7984 | 0.5934 | 0.2049 |
| metacritic_dbpedia | comem | 0.6154 | 0.5000 | 0.1154 |
| metacritic_dbpedia | ditto_plm | 0.8455 | 0.6188 | 0.2267 |
| metacritic_dbpedia | llm_matcher | 0.6065 | 0.4965 | 0.1100 |
| metacritic_dbpedia | magellan | 0.8128 | 0.4250 | 0.3878 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5582 | 0.6485 | -0.0903 |
| casefusion_only | 0.5556 | 0.6986 | -0.1430 |
| fusionquery_only | 0.5860 | 0.7208 | -0.1348 |
| llm_only | 0.5185 | 0.6140 | -0.0955 |
| ltm_only | 0.5542 | 0.6796 | -0.1254 |
| prefer_higher_trust_only | 0.5701 | 0.7186 | -0.1485 |
| pydi_per_attribute_optimal | 0.5529 | 0.6986 | -0.1456 |
| truthfinder_only | 0.5754 | 0.7208 | -0.1454 |
| voting_only | 0.5516 | 0.6440 | -0.0925 |

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
