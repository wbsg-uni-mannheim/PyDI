# PyDI - Use Cases

This folder contains example notebooks implementing end-to-end data integration pipelines with PyDI. Each notebook demonstrates how to integrate data from multiple sources in a specific domain, covering the full pipeline from data loading and profiling through entity matching to data fusion.

## Use Cases Overview

| Use Case | Datasets | Total Records | Pipeline Steps |
|----------|----------|---------------|----------------|
| [Movies](movies/movies_workflow.ipynb) | Academy Awards (4,580), Actors (151), Golden Globes (2,279) | ~7,000 | Profiling → Blocking → Entity Matching → Data Fusion |
| [Companies](companies/companies_workflow.ipynb) | DBpedia (10,092), Forbes (2,000), FullContact (1,931) | ~14,000 | Schema Matching → Normalization → Profiling → Entity Matching → Data Fusion |
| [Games](games/games_workflow.ipynb) | DBpedia (65,000), Metacritic (20,494), Sales (7,878) | ~93,000 | Schema Matching → Normalization → Profiling → Entity Matching → Data Fusion |
| [Music](music/music_workflow.ipynb) | MusicBrainz (4,763), Discogs (22,627), Last.fm (9,865) | ~37,000 | Schema Matching → Normalization → Profiling → Entity Matching → Data Fusion |

## Test Sets

Each use case includes test sets for evaluating the individual integration phases:

- **Entity Matching**: Labeled record pairs for evaluating blocking and matching quality
- **Data Fusion**: Test sets for evaluating fusion accuracy

These test sets are located in the `<domain>/input/entitymatching/` and `<domain>/input/fusion/` subdirectories.

## Running the Notebooks

Each notebook is self-contained and can be run from start to finish. The notebooks expect:

1. Input data in the `<domain>/input/` subdirectory
2. Output will be written to the `<domain>/output/` subdirectory

Make sure you have PyDI installed and, for workflows using LLM-based schema matching, an OpenAI API key configured in your environment.