# Baseline report - music

_Generated at 2026-06-06T02:17:25.039308+00:00_

## Stage: sm

### SM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 1.0000 |
| best_member_name | llm_openai |
| macro_f1 | 0.7430 |
| macro_precision | 0.9464 |
| macro_recall | 0.6667 |
| max_f1 | 1.0000 |
| min_f1 | 0.3200 |

### SM - per attribute
| attribute | any_correct | coma_hybrid | duplicate_majority | embedding_sbert | instance_tf_cosine | label_jw | llm_openai | magneto_slm_llm |
|---|---|---|---|---|---|---|---|---|
| discogs.category | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.duration | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| discogs.imprint | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.origin_loc | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.performer | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.pub_dt | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.title_str | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| discogs.tracks_track-name | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.album_length | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.album_title | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.band | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lastfm.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| lastfm.tracks_track-name | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_2 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| musicbrainz.Attribute_3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_4 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_6 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.Attribute_9 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| musicbrainz.id | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Stage: norm

### NORM - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9884 |
| best_member_name_f1 | 1.0000 |
| macro_f1 | 0.9786 |
| macro_precision | 0.9913 |
| macro_recall | 0.9675 |
| max_f1 | 0.9884 |
| min_f1 | 0.9663 |

### NORM - per attribute
| attribute | any_correct | best_member_f1 | llm_only | passthrough | rule_per_attribute_optimal |
|---|---|---|---|---|---|
| artist | 1.0000 | 1.0000 | 0.9910 | 1.0000 | 1.0000 |
| duration | 1.0000 | 0.9231 | 0.9126 | 0.9231 | 0.9231 |
| id | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| label | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| name | 1.0000 | 1.0000 | 0.9955 | 1.0000 | 1.0000 |
| release-country | 1.0000 | 0.9954 | 0.9954 | 0.9954 | 0.9954 |
| release-date | 1.0000 | 1.0000 | 0.9727 | 0.8455 | 1.0000 |

## Stage: em_blocking

### EM_BLOCKING - aggregated
| metric | value |
|---|---|
| best_member_name | embedding_blocker |
| best_member_pair_recall | 0.9970 |
| best_member_reduction_ratio | 0.9964 |
| macro_pair_recall | 0.9505 |
| macro_pair_recall_baseline_model_on_baseline_test | 0.9505 |
| macro_pair_recall_baseline_model_on_regen_test | 0.9505 |
| macro_pair_recall_variant_model_on_baseline_test | 0.9505 |
| macro_pair_recall_variant_model_on_regen_test | 0.9505 |
| macro_reduction_ratio | 0.9789 |
| max_pair_recall | 1.0000 |
| min_pair_recall | 0.8498 |
| recall_floor | 0.9700 |

## Stage: em_matching

### EM_MATCHING - aggregated
| metric | value |
|---|---|
| best_member_f1 | 0.9531 |
| best_member_name | magellan |
| macro_f1 | 0.8242 |
| macro_f1_baseline_model_on_baseline_test | 0.8242 |
| macro_f1_baseline_model_on_regen_test | 0.8242 |
| macro_f1_baseline_test | 0.8242 |
| macro_f1_regen_test | 0.8242 |
| macro_f1_variant_model_on_baseline_test | 0.8242 |
| macro_f1_variant_model_on_regen_test | 0.8242 |
| macro_precision | 0.9952 |
| macro_recall | 0.7354 |
| max_f1 | 0.9531 |
| min_f1 | 0.6914 |

## Stage: fusion

### FUSION - aggregated
| metric | value |
|---|---|
| best_member_macro_accuracy | 0.7789 |
| macro_accuracy | 0.7549 |
| max_accuracy | 0.7789 |
| min_accuracy | 0.7227 |
| overall_accuracy | 0.7789 |

### FUSION - per attribute
| attribute | accusim_only | best_member_accuracy | casefusion_only | fusionquery_only | llm_only | ltm_only | mean_member_accuracy | prefer_higher_trust_only | pydi_per_attribute_optimal | truthfinder_only | voting_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| artist | 0.9400 | 0.9700 | 0.9200 | 0.7400 | 0.9700 | 0.8600 | 0.8978 | 0.9200 | 0.9100 | 0.9100 | 0.9100 |
| duration | 0.6800 | 0.6800 | 0.6200 | 0.6200 | 0.5700 | 0.6200 | 0.6222 | 0.6200 | 0.6200 | 0.6200 | 0.6300 |
| genre | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 | 0.9091 |
| label | 0.8925 | 0.9140 | 0.8925 | 0.8925 | 0.9140 | 0.8925 | 0.9020 | 0.9140 | 0.9140 | 0.8925 | 0.9140 |
| name | 0.9200 | 0.9300 | 0.9200 | 0.8000 | 0.9200 | 0.9200 | 0.9100 | 0.9200 | 0.9300 | 0.9300 | 0.9300 |
| release-country | 0.9100 | 0.9600 | 0.6400 | 0.8700 | 0.3200 | 0.8800 | 0.8156 | 0.9600 | 0.9100 | 0.9400 | 0.9100 |
| release-date | 0.9500 | 0.9500 | 0.9400 | 0.9500 | 0.9500 | 0.9100 | 0.9433 | 0.9400 | 0.9500 | 0.9500 | 0.9500 |
| tracks | 0.0300 | 0.2300 | 0.0000 | 0.0000 | 0.2300 | 0.0300 | 0.0389 | 0.0300 | 0.0000 | 0.0000 | 0.0300 |
