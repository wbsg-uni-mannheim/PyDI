# Validation report - music / medium

_Generated at 2026-06-01T23:05:00.422007+00:00_

- domain: `music`
- level: `medium`
- with_llm: `True`
- committee_versions: sm=`sm_committee.yaml@cb4a6847ac9e`, norm=`normalization_committee_music.yaml@50d9aba427f3`, em_blocking=`em_blocking_committee_music.yaml@bcca7ab81298`, em_matching=`em_matching_committee_music.yaml@32e807ea7330`, fusion=`fusion_committee_music.yaml@aaa7a142c94f`

## Stage summary

| stage | metric | measured | baseline | delta |
|---|---|---|---|---|
| sm | macro_f1 | 0.7616 | 0.8761 | -0.1145 |
| norm | macro_f1 | 0.5802 | 0.6577 | -0.0775 |
| em_blocking | macro_pair_recall_variant_model_on_regen_test | 0.0000 | 0.0000 | 0.0000 |
| em_matching | macro_f1_variant_model_on_regen_test | 0.8418 | 0.8325 | 0.0093 |
| fusion | overall_accuracy | 0.7654 | 0.8929 | -0.1275 |

## Stage: sm - per member

| member | f1 | f1_baseline | f1_delta |
|---|---|---|---|
| coma_hybrid | 0.7143 | 1.0000 | -0.2857 |
| duplicate_majority | 0.7143 | 0.7442 | -0.0299 |
| embedding_sbert | 0.7451 | 0.7500 | -0.0049 |
| instance_tf_cosine | 0.6087 | 0.6383 | -0.0296 |
| label_jw | 0.6977 | 1.0000 | -0.3023 |
| llm_openai | 1.0000 | 1.0000 | 0.0000 |
| magneto_slm_llm | 0.8511 | 1.0000 | -0.1489 |

## Stage: norm - per member

| member | macro_f1 | macro_f1_baseline | macro_f1_delta |
|---|---|---|---|
| llm_only | 0.5578 | 0.6226 | -0.0648 |
| passthrough | 0.6093 | 0.6983 | -0.0889 |
| rule_per_attribute_optimal | 0.5735 | 0.6523 | -0.0787 |

## Stage: em_blocking - per member

| member | pair_recall | pair_recall_baseline | pair_recall_delta | reduction_ratio | reduction_ratio_baseline |
|---|---|---|---|---|---|
| bm25_blocker | 0.9955 | 0.9985 | -0.0030 | 0.9964 | 0.9964 |
| embedding_blocker | 0.9940 | 0.9970 | -0.0030 | 0.9964 | 0.9964 |
| sc_block | 1.0000 | 0.9985 | 0.0015 | 0.9927 | 0.9927 |
| sorted_neighbourhood_blocker | 0.8288 | 0.8498 | -0.0210 | 0.9960 | 0.9959 |
| standard_blocker | 0.8423 | 0.8589 | -0.0165 | 0.9859 | 0.9858 |
| token_blocker | 1.0000 | 1.0000 | 0.0000 | 0.9067 | 0.9063 |

## Stage: em_matching - per member

| member | f1 | f1_baseline | f1_delta | f1_baseline_test | f1_baseline_test_baseline | f1_regen_test | f1_regen_test_baseline |
|---|---|---|---|---|---|---|---|
| comem | 0.7296 | 0.6914 | 0.0381 | 0.7296 | 0.6914 | 0.7296 | 0.6914 |
| ditto_plm | 0.9716 | 0.9903 | -0.0187 | 0.9835 | 0.9903 | 0.9835 | 0.9903 |
| llm_matcher | 0.7189 | 0.7002 | 0.0186 | 0.7189 | 0.7002 | 0.7189 | 0.7002 |
| magellan | 0.9472 | 0.9480 | -0.0008 | 0.9472 | 0.9480 | 0.9472 | 0.9480 |

## Stage: em_matching - per pair

| pair | member | f1 | f1_baseline | f1_delta |
|---|---|---|---|---|
| musicbrainz_discogs | comem | 0.8625 | 0.8307 | 0.0318 |
| musicbrainz_discogs | ditto_plm | 0.9660 | 0.9911 | -0.0250 |
| musicbrainz_discogs | llm_matcher | 0.8567 | 0.8483 | 0.0084 |
| musicbrainz_discogs | magellan | 0.9156 | 0.9156 | 0.0000 |
| musicbrainz_lastfm | comem | 0.5966 | 0.5522 | 0.0445 |
| musicbrainz_lastfm | ditto_plm | 0.9772 | 0.9896 | -0.0124 |
| musicbrainz_lastfm | llm_matcher | 0.5811 | 0.5522 | 0.0289 |
| musicbrainz_lastfm | magellan | 0.9787 | 0.9803 | -0.0016 |

## Stage: fusion - per member

| member | overall_accuracy | overall_accuracy_baseline | overall_accuracy_delta |
|---|---|---|---|
| accusim_only | 0.6766 | 0.8922 | -0.2156 |
| casefusion_only | 0.7234 | 0.8506 | -0.1273 |
| fusionquery_only | 0.6896 | 0.8429 | -0.1532 |
| llm_only | 0.6597 | 0.8091 | -0.1494 |
| ltm_only | 0.6494 | 0.8662 | -0.2169 |
| prefer_higher_trust_only | 0.6623 | 0.8922 | -0.2299 |
| pydi_per_attribute_optimal | 0.7455 | 0.8896 | -0.1442 |
| truthfinder_only | 0.7610 | 0.8909 | -0.1299 |
| voting_only | 0.6610 | 0.8883 | -0.2273 |

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
