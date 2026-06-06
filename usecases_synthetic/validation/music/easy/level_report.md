# Validation report - music / easy

_Generated at 2026-06-05T20:14:00.623877+00:00_

- domain: `music`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@50d9aba427f3`, em_blocking=`em_blocking_committee_music.yaml@07cec0b9969f`, em_matching=`em_matching_committee_music.yaml@2c90f2517590`, fusion=`fusion_committee_music.yaml@f269c37ab587`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.9004 | 0.7430 | 0.1574 |
| norm | macro_f1 | 0.8557 | 0.9786 | -0.1228 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8727 | 0.8242 | 0.0485 |
| fusion | overall_accuracy | 0.8660 | 0.7789 | 0.0870 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 0.5517 | 0.4483 |
| duplicate_majority | 0.7442 | 0.8649 | -0.1207 |
| embedding_sbert | 0.8276 | 0.7568 | 0.0708 |
| instance_tf_cosine | 0.7308 | 0.7317 | -0.0009 |
| label_jw | 1.0000 | 0.3200 | 0.6800 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 1.0000 | 0.9756 | 0.0244 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.8426 | 0.9810 | -0.1384 |
| passthrough | 0.8294 | 0.9663 | -0.1369 |
| rule_per_attribute_optimal | 0.8952 | 0.9884 | -0.0931 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9985 | 0.9985 | 0.0000 | 0.9967 | 0.9964 |
| embedding_blocker | 0.9985 | 0.9970 | 0.0015 | 0.9967 | 0.9964 |
| sc_block | 1.0000 | 0.9985 | 0.0015 | 0.9934 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8994 | 0.8498 | 0.0495 | 0.9962 | 0.9959 |
| standard_blocker | 0.9114 | 0.8589 | 0.0526 | 0.9849 | 0.9858 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9079 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8274 | 0.6914 | 0.1360 | 0.8274 | 0.6914 | 0.8274 | 0.6914 |
| ditto_plm | 0.8917 | 0.9521 | -0.0604 | 0.9297 | 0.9521 | 0.9297 | 0.9521 |
| llm_matcher | 0.8240 | 0.7002 | 0.1238 | 0.8240 | 0.7002 | 0.8240 | 0.7002 |
| magellan | 0.9479 | 0.9531 | -0.0052 | 0.9513 | 0.9531 | 0.9513 | 0.9531 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.9058 | 0.8307 | 0.0751 |
| musicbrainz_discogs | ditto_plm | 0.8094 | 0.9194 | -0.1100 |
| musicbrainz_discogs | llm_matcher | 0.8998 | 0.8483 | 0.0516 |
| musicbrainz_discogs | magellan | 0.9186 | 0.9243 | -0.0057 |
| musicbrainz_lastfm | comem | 0.7491 | 0.5522 | 0.1969 |
| musicbrainz_lastfm | ditto_plm | 0.9740 | 0.9848 | -0.0108 |
| musicbrainz_lastfm | llm_matcher | 0.7481 | 0.5522 | 0.1959 |
| musicbrainz_lastfm | magellan | 0.9772 | 0.9818 | -0.0046 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.8416 | 0.7740 | 0.0675 |
| casefusion_only | 0.8299 | 0.7234 | 0.1065 |
| fusionquery_only | 0.7675 | 0.7156 | 0.0519 |
| llm_only | 0.7844 | 0.7156 | 0.0688 |
| ltm_only | 0.8442 | 0.7468 | 0.0974 |
| prefer_higher_trust_only | 0.8623 | 0.7714 | 0.0909 |
| pydi_per_attribute_optimal | 0.8584 | 0.7623 | 0.0961 |
| truthfinder_only | 0.8649 | 0.7636 | 0.1013 |
| voting_only | 0.8532 | 0.7675 | 0.0857 |

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
