# Validation report - music / hard

_Generated at 2026-06-06T03:05:29.544646+00:00_

- domain: `music`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@50d9aba427f3`, em_blocking=`em_blocking_committee_music.yaml@07cec0b9969f`, em_matching=`em_matching_committee_music.yaml@2c90f2517590`, fusion=`fusion_committee_music.yaml@f269c37ab587`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6549 | 0.7430 | -0.0881 |
| norm | macro_f1 | 0.7811 | 0.9786 | -0.1974 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9322 | 0.9505 | -0.0183 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8560 | 0.8242 | 0.0318 |
| fusion | overall_accuracy | 0.5661 | 0.7789 | -0.2128 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.5000 | 0.5517 | -0.0517 |
| duplicate_majority | 0.7442 | 0.8649 | -0.1207 |
| embedding_sbert | 0.7391 | 0.7568 | -0.0176 |
| instance_tf_cosine | 0.6047 | 0.7317 | -0.1271 |
| label_jw | 0.3636 | 0.3200 | 0.0436 |
| llm_openai | 0.8980 | 1.0000 | -0.1020 |
| magneto_slm_llm | 0.7347 | 0.9756 | -0.2409 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.7899 | 0.9810 | -0.1911 |
| passthrough | 0.6944 | 0.9663 | -0.2719 |
| rule_per_attribute_optimal | 0.8591 | 0.9884 | -0.1293 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9765 | 0.9985 | -0.0220 | 0.9962 | 0.9964 |
| embedding_blocker | 0.9726 | 0.9970 | -0.0244 | 0.9962 | 0.9964 |
| sc_block | 0.9965 | 0.9985 | -0.0020 | 0.9924 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8349 | 0.8498 | -0.0150 | 0.9955 | 0.9959 |
| standard_blocker | 0.8422 | 0.8589 | -0.0166 | 0.9868 | 0.9858 |
| token_blocker | 0.9706 | 1.0000 | -0.0294 | 0.9118 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7464 | 0.6914 | 0.0550 | 0.6754 | 0.6914 | 0.7464 | 0.6914 |
| ditto_plm | 0.9698 | 0.9521 | 0.0177 | 0.8765 | 0.9521 | 0.8993 | 0.9521 |
| llm_matcher | 0.7557 | 0.7002 | 0.0555 | 0.6858 | 0.7002 | 0.7557 | 0.7002 |
| magellan | 0.9520 | 0.9531 | -0.0010 | 0.9293 | 0.9531 | 0.9509 | 0.9531 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8000 | 0.8307 | -0.0307 |
| musicbrainz_discogs | ditto_plm | 0.9627 | 0.9194 | 0.0433 |
| musicbrainz_discogs | llm_matcher | 0.8039 | 0.8483 | -0.0444 |
| musicbrainz_discogs | magellan | 0.9365 | 0.9243 | 0.0122 |
| musicbrainz_lastfm | comem | 0.6928 | 0.5522 | 0.1407 |
| musicbrainz_lastfm | ditto_plm | 0.9770 | 0.9848 | -0.0079 |
| musicbrainz_lastfm | llm_matcher | 0.7075 | 0.5522 | 0.1553 |
| musicbrainz_lastfm | magellan | 0.9675 | 0.9818 | -0.0143 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5441 | 0.7740 | -0.2299 |
| casefusion_only | 0.5369 | 0.7234 | -0.1864 |
| fusionquery_only | 0.5369 | 0.7156 | -0.1786 |
| llm_only | 0.5676 | 0.7156 | -0.1480 |
| ltm_only | 0.5243 | 0.7468 | -0.2224 |
| prefer_higher_trust_only | 0.5423 | 0.7714 | -0.2291 |
| pydi_per_attribute_optimal | 0.5441 | 0.7623 | -0.2182 |
| truthfinder_only | 0.5658 | 0.7636 | -0.1979 |
| voting_only | 0.5477 | 0.7675 | -0.2198 |

## Stage: fusion - per attribute

| attribute | best_accuracy | baseline | delta | spread | spread_baseline | spread_delta |
|---|---|---|---|---|---|---|
| artist | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| duration | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| genre | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| label | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| name | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| release-country | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| release-date | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tracks | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
