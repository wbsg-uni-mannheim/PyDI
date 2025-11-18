import pytest
from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.entitymatching import (
    StandardBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker
)


def test_standard_blocking(get_input_data):
    actors_df = get_input_data("movies", "actors")
    golden_globes = get_input_data("movies", "golden_globes")

    actors_df['title_prefix'] = actors_df['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    golden_globes['title_prefix'] = golden_globes['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    standard_blocker_a2g = StandardBlocker(
        actors_df, golden_globes,
        on=['title_prefix'],
        batch_size=1000,
        id_column='id'
    )
    
    standard_candidates_a2g = standard_blocker_a2g.materialize()
    assert len(standard_candidates_a2g) == 277, f"Expected 277 candidates, got {len(standard_candidates_a2g)}"

def test_sorted_neighborhood_blocking(get_input_data):
    actors = get_input_data("movies", "actors")
    golden_globes = get_input_data("movies", "golden_globes")

    actors['title_prefix'] = actors['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    golden_globes['title_prefix'] = golden_globes['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    sn_blocker_a2g = SortedNeighbourhoodBlocker(
        actors, golden_globes,
        key='title',
        window=20,
        batch_size=1000,
        id_column='id'
    )
    sn_candidates_a2g = sn_blocker_a2g.materialize()
    assert len(sn_candidates_a2g) == 4899, f"Expected 4899 candidates, got {len(sn_candidates_a2g)}"

def test_token_blocking(get_input_data):
    actors = get_input_data("movies", "actors")
    golden_globes = get_input_data("movies", "golden_globes")

    actors['title_prefix'] = actors['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    golden_globes['title_prefix'] = golden_globes['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    token_blocker_a2g = TokenBlocker(
        actors, golden_globes,
        column='title',
        batch_size=1000,
        id_column='id',
        ngram_size=2,
        ngram_type='character'
    )
    token_candidates_a2g = token_blocker_a2g.materialize()
    assert len(token_candidates_a2g) == 166834, f"Expected 166834 candidates, got {len(token_candidates_a2g)}"

def test_evaluate_blocking(get_input_data, get_correspondences):
    actors = get_input_data("movies", "actors")
    golden_globes = get_input_data("movies", "golden_globes")

    actors['title_prefix'] = actors['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))
    golden_globes['title_prefix'] = golden_globes['title'].astype(str).apply(lambda x: ''.join([word[:2].upper() for word in x.split()[:3]]))

    standard_blocker_a2g = StandardBlocker(
        actors, golden_globes,
        on=['title_prefix'],
        batch_size=1000,
        id_column='id'
    )
    standard_candidates_a2g = standard_blocker_a2g.materialize()

    from pathlib import Path
    def get_repo_root():
        current = Path.cwd()
        while current != current.parent:
            if (current / 'pyproject.toml').exists():
                return current
            current = current.parent
        return Path.cwd()

    test_gt = get_correspondences("movies", "actors", "golden_globes")
        
    results = EntityMatchingEvaluator.evaluate_blocking(
        standard_candidates_a2g,
        blocker=standard_blocker_a2g,
        test_pairs=test_gt,
    )

    assert results["pair_completeness"] == pytest.approx(0.346, rel=1e-3), f"Expected pair completeness 0.346, got {results['pair_completeness']}"
