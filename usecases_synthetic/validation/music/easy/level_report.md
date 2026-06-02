# Validation report - music / easy

_Generated at 2026-06-01T22:05:29.102779+00:00_

- domain: `music`
- level: `easy`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@50d9aba427f3`, em_blocking=`em_blocking_committee_music.yaml@bcca7ab81298`, em_matching=`em_matching_committee_music.yaml@32e807ea7330`, fusion=`fusion_committee_music.yaml@aaa7a142c94f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.9004 | 0.8761 | 0.0243 |
| norm | macro_f1 | 0.6182 | 0.6577 | -0.0395 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8899 | 0.8325 | 0.0575 |
| fusion | overall_accuracy | 0.8660 | 0.8929 | -0.0269 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 1.0000 | 1.0000 | 0.0000 |
| duplicate_majority | 0.7442 | 0.7442 | 0.0000 |
| embedding_sbert | 0.8276 | 0.7500 | 0.0776 |
| instance_tf_cosine | 0.7308 | 0.6383 | 0.0925 |
| label_jw | 1.0000 | 1.0000 | 0.0000 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 1.0000 | 1.0000 | 0.0000 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5942 | 0.6226 | -0.0284 |
| passthrough | 0.6475 | 0.6983 | -0.0507 |
| rule_per_attribute_optimal | 0.6130 | 0.6523 | -0.0393 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9970 | 0.9985 | -0.0015 | 0.9967 | 0.9964 |
| embedding_blocker | 0.9985 | 0.9970 | 0.0015 | 0.9967 | 0.9964 |
| sc_block | 0.9985 | 0.9985 | 0.0000 | 0.9934 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8994 | 0.8498 | 0.0495 | 0.9962 | 0.9959 |
| standard_blocker | 0.9114 | 0.8589 | 0.0526 | 0.9849 | 0.9858 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9079 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.8274 | 0.6914 | 0.1360 | 0.8274 | 0.6914 | 0.8274 | 0.6914 |
| ditto_plm | 0.9604 | 0.9903 | -0.0299 | 0.9835 | 0.9903 | 0.9827 | 0.9903 |
| llm_matcher | 0.8240 | 0.7002 | 0.1238 | 0.8240 | 0.7002 | 0.8240 | 0.7002 |
| magellan | 0.9480 | 0.9480 | 0.0000 | 0.9513 | 0.9480 | 0.9513 | 0.9480 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.9058 | 0.8307 | 0.0751 |
| musicbrainz_discogs | ditto_plm | 0.9359 | 0.9911 | -0.0552 |
| musicbrainz_discogs | llm_matcher | 0.8998 | 0.8483 | 0.0516 |
| musicbrainz_discogs | magellan | 0.9188 | 0.9156 | 0.0032 |
| musicbrainz_lastfm | comem | 0.7491 | 0.5522 | 0.1969 |
| musicbrainz_lastfm | ditto_plm | 0.9848 | 0.9896 | -0.0047 |
| musicbrainz_lastfm | llm_matcher | 0.7481 | 0.5522 | 0.1959 |
| musicbrainz_lastfm | magellan | 0.9772 | 0.9803 | -0.0032 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.8429 | 0.8922 | -0.0494 |
| casefusion_only | 0.8299 | 0.8506 | -0.0208 |
| fusionquery_only | 0.7675 | 0.8429 | -0.0753 |
| llm_only | 0.7844 | 0.8091 | -0.0247 |
| ltm_only | 0.8442 | 0.8662 | -0.0221 |
| prefer_higher_trust_only | 0.8623 | 0.8922 | -0.0299 |
| pydi_per_attribute_optimal | 0.8584 | 0.8896 | -0.0312 |
| truthfinder_only | 0.8649 | 0.8909 | -0.0260 |
| voting_only | 0.8532 | 0.8883 | -0.0351 |

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
