# Validation report - music / medium

_Generated at 2026-06-06T02:52:12.669547+00:00_

- domain: `music`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@50d9aba427f3`, em_blocking=`em_blocking_committee_music.yaml@07cec0b9969f`, em_matching=`em_matching_committee_music.yaml@2c90f2517590`, fusion=`fusion_committee_music.yaml@f269c37ab587`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7683 | 0.7430 | 0.0253 |
| norm | macro_f1 | 0.7874 | 0.9786 | -0.1912 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.9437 | 0.9505 | -0.0068 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8345 | 0.8242 | 0.0103 |
| fusion | overall_accuracy | 0.7654 | 0.7789 | -0.0135 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7143 | 0.5517 | 0.1626 |
| duplicate_majority | 0.7143 | 0.8649 | -0.1506 |
| embedding_sbert | 0.7451 | 0.7568 | -0.0117 |
| instance_tf_cosine | 0.6087 | 0.7317 | -0.1230 |
| label_jw | 0.6977 | 0.3200 | 0.3777 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8980 | 0.9756 | -0.0777 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.7951 | 0.9810 | -0.1860 |
| passthrough | 0.6966 | 0.9663 | -0.2697 |
| rule_per_attribute_optimal | 0.8704 | 0.9884 | -0.1180 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9955 | 0.9985 | -0.0030 | 0.9964 | 0.9964 |
| embedding_blocker | 0.9955 | 0.9970 | -0.0015 | 0.9964 | 0.9964 |
| sc_block | 1.0000 | 0.9985 | 0.0015 | 0.9927 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8288 | 0.8498 | -0.0210 | 0.9960 | 0.9959 |
| standard_blocker | 0.8423 | 0.8589 | -0.0165 | 0.9859 | 0.9858 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9067 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7296 | 0.6914 | 0.0381 | 0.7296 | 0.6914 | 0.7296 | 0.6914 |
| ditto_plm | 0.9417 | 0.9521 | -0.0104 | 0.9111 | 0.9521 | 0.9111 | 0.9521 |
| llm_matcher | 0.7189 | 0.7002 | 0.0186 | 0.7189 | 0.7002 | 0.7189 | 0.7002 |
| magellan | 0.9480 | 0.9531 | -0.0050 | 0.9480 | 0.9531 | 0.9480 | 0.9531 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8625 | 0.8307 | 0.0318 |
| musicbrainz_discogs | ditto_plm | 0.9031 | 0.9194 | -0.0162 |
| musicbrainz_discogs | llm_matcher | 0.8567 | 0.8483 | 0.0084 |
| musicbrainz_discogs | magellan | 0.9173 | 0.9243 | -0.0070 |
| musicbrainz_lastfm | comem | 0.5966 | 0.5522 | 0.0445 |
| musicbrainz_lastfm | ditto_plm | 0.9803 | 0.9848 | -0.0045 |
| musicbrainz_lastfm | llm_matcher | 0.5811 | 0.5522 | 0.0289 |
| musicbrainz_lastfm | magellan | 0.9787 | 0.9818 | -0.0031 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6792 | 0.7740 | -0.0948 |
| casefusion_only | 0.7234 | 0.7234 | 0.0000 |
| fusionquery_only | 0.6896 | 0.7156 | -0.0260 |
| llm_only | 0.6597 | 0.7156 | -0.0558 |
| ltm_only | 0.6494 | 0.7468 | -0.0974 |
| prefer_higher_trust_only | 0.6623 | 0.7714 | -0.1091 |
| pydi_per_attribute_optimal | 0.7455 | 0.7623 | -0.0169 |
| truthfinder_only | 0.7610 | 0.7636 | -0.0026 |
| voting_only | 0.6610 | 0.7675 | -0.1065 |

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
