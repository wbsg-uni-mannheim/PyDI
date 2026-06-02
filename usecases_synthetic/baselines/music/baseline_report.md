# Baseline report - music

_Generated at 2026-06-01T21:18:50.564781+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| macro_f1 | 0.8761 |
| macro_precision | 0.9439 |
| macro_recall | 0.8307 |
| max_f1 | 1.0000 |
| min_f1 | 0.6383 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| discogs.artist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.duration | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.genre | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.label | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.release-country | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.release-date | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.tracks | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.artist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.duration | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.genre | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.label | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.release-country | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.release-date | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.tracks | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.artist | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.duration | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.genre | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.label | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.name | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.release-country | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.release-date | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| musicbrainz.tracks | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.6983 |
| best_member_name_f1 | 1.0000 |
| macro_f1 | 0.6577 |
| macro_precision | 0.8130 |
| macro_recall | 0.5905 |
| max_f1 | 0.6983 |
| min_f1 | 0.6226 |

### NORM - per attribute
| attribute | any_correct | best_member_f1 | llm_only | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| artist | 1.0000 | 0.9114 | 0.9114 | 0.8932 | 0.8932 |
| duration | 1.0000 | 0.6743 | 0.6682 | 0.6743 | 0.6743 |
| genre | 1.0000 | 0.5375 | 0.5159 | 0.5375 | 0.2963 |
| label | 1.0000 | 0.5407 | 0.5132 | 0.5407 | 0.5407 |
| name | 1.0000 | 0.9487 | 0.9465 | 0.9487 | 0.9487 |
| release-country | 1.0000 | 0.5286 | 0.1143 | 0.5286 | 0.5286 |
| release-date | 1.0000 | 0.7648 | 0.6888 | 0.7648 | 0.6841 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | embedding_blocker |
| best_member_pair_recall | 0.9970 |
| best_member_reduction_ratio | 0.9964 |
| macro_pair_recall | 0.9505 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.0000 |
| macro_pair_recall_baseline_model_on_regen_test | 0.0000 |
| macro_pair_recall_variant_model_on_baseline_test | 0.0000 |
| macro_pair_recall_variant_model_on_regen_test | 0.0000 |
| macro_reduction_ratio | 0.9789 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.8498 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9903 |
| best_member_name | ditto_plm |
| macro_f1 | 0.8325 |
| macro_f1_baseline_model_on_baseline_test | 0.8325 |
| macro_f1_baseline_model_on_regen_test | 0.8325 |
| macro_f1_baseline_test | 0.8325 |
| macro_f1_regen_test | 0.8325 |
| macro_f1_variant_model_on_baseline_test | 0.8325 |
| macro_f1_variant_model_on_regen_test | 0.8325 |
| macro_precision | 0.9921 |
| macro_recall | 0.7541 |
| max_f1 | 0.9903 |
| min_f1 | 0.6914 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.8929 |
| macro_accuracy | 0.8706 |
| max_accuracy | 0.8929 |
| min_accuracy | 0.8129 |
| overall_accuracy | 0.8929 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| artist | 0.9400 | 0.9700 | 0.9200 | 0.7400 | 0.9700 | 0.8600 | 0.8978 | 0.9200 | 0.9100 | 0.9100 | 0.9100 |
| duration | 0.6700 | 0.6700 | 0.6200 | 0.6200 | 0.5700 | 0.6200 | 0.6211 | 0.6200 | 0.6200 | 0.6200 | 0.6300 |
| genre | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 |
| label | 0.8925 | 0.9140 | 0.8925 | 0.8925 | 0.9140 | 0.8925 | 0.9020 | 0.9140 | 0.9140 | 0.8925 | 0.9140 |
| name | 0.9200 | 0.9300 | 0.9200 | 0.8000 | 0.9200 | 0.9200 | 0.9100 | 0.9200 | 0.9300 | 0.9300 | 0.9300 |
| release-country | 0.9100 | 0.9600 | 0.6400 | 0.8700 | 0.3200 | 0.8800 | 0.8156 | 0.9600 | 0.9100 | 0.9400 | 0.9100 |
| release-date | 0.9500 | 0.9500 | 0.9400 | 0.9500 | 0.9500 | 0.9100 | 0.9433 | 0.9400 | 0.9500 | 0.9500 | 0.9500 |
| tracks | 0.9500 | 0.9800 | 0.9800 | 0.9800 | 0.9500 | 0.9500 | 0.9656 | 0.9600 | 0.9800 | 0.9800 | 0.9600 |
