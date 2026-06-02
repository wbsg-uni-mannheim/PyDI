# Validation report - music / hard

_Generated at 2026-06-01T23:47:56.927101+00:00_

- domain: `music`
- level: `hard`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@50d9aba427f3`, em_blocking=`em_blocking_committee_music.yaml@bcca7ab81298`, em_matching=`em_matching_committee_music.yaml@32e807ea7330`, fusion=`fusion_committee_music.yaml@aaa7a142c94f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.6781 | 0.8761 | -0.1980 |
| norm | macro_f1 | 0.5320 | 0.6577 | -0.1257 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8560 | 0.8325 | 0.0235 |
| fusion | overall_accuracy | 0.5661 | 0.8929 | -0.3268 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.5000 | 1.0000 | -0.5000 |
| duplicate_majority | 0.7442 | 0.7442 | 0.0000 |
| embedding_sbert | 0.7391 | 0.7500 | -0.0109 |
| instance_tf_cosine | 0.6047 | 0.6383 | -0.0336 |
| label_jw | 0.3636 | 1.0000 | -0.6364 |
| llm_openai | 0.9200 | 1.0000 | -0.0800 |
| magneto_slm_llm | 0.8750 | 1.0000 | -0.1250 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5120 | 0.6226 | -0.1106 |
| passthrough | 0.5694 | 0.6983 | -0.1289 |
| rule_per_attribute_optimal | 0.5145 | 0.6523 | -0.1377 |

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
| ditto_plm | 0.9697 | 0.9903 | -0.0206 | 0.9686 | 0.9903 | 0.9653 | 0.9903 |
| llm_matcher | 0.7557 | 0.7002 | 0.0555 | 0.6858 | 0.7002 | 0.7557 | 0.7002 |
| magellan | 0.9520 | 0.9480 | 0.0041 | 0.9272 | 0.9480 | 0.9498 | 0.9480 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8000 | 0.8307 | -0.0307 |
| musicbrainz_discogs | ditto_plm | 0.9664 | 0.9911 | -0.0247 |
| musicbrainz_discogs | llm_matcher | 0.8039 | 0.8483 | -0.0444 |
| musicbrainz_discogs | magellan | 0.9365 | 0.9156 | 0.0210 |
| musicbrainz_lastfm | comem | 0.6928 | 0.5522 | 0.1407 |
| musicbrainz_lastfm | ditto_plm | 0.9731 | 0.9896 | -0.0165 |
| musicbrainz_lastfm | llm_matcher | 0.7075 | 0.5522 | 0.1553 |
| musicbrainz_lastfm | magellan | 0.9675 | 0.9803 | -0.0128 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.5441 | 0.8922 | -0.3481 |
| casefusion_only | 0.5369 | 0.8506 | -0.3137 |
| fusionquery_only | 0.5369 | 0.8429 | -0.3059 |
| llm_only | 0.5676 | 0.8091 | -0.2415 |
| ltm_only | 0.5243 | 0.8662 | -0.3419 |
| prefer_higher_trust_only | 0.5423 | 0.8922 | -0.3499 |
| pydi_per_attribute_optimal | 0.5441 | 0.8896 | -0.3455 |
| truthfinder_only | 0.5658 | 0.8909 | -0.3251 |
| voting_only | 0.5477 | 0.8883 | -0.3406 |

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
