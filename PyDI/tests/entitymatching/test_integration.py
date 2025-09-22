"""Integration tests for entity matching using real datasets."""

import pandas as pd
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path

from PyDI.entitymatching import (
    RuleBasedMatcher,
    MLBasedMatcher,
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EntityMatchingEvaluator,
    ensure_record_ids
)


def load_movie_xml_data(xml_path: Path) -> pd.DataFrame:
    """Load movie data from XML file."""
    if not xml_path.exists():
        pytest.skip(f"Dataset not found: {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        data = []
        for movie in root.findall('movie'):
            movie_data = {
                "id": movie.find('id').text if movie.find('id') is not None else "",
                "title": movie.find('title').text if movie.find('title') is not None else "",
                "date": movie.find('date').text if movie.find('date') is not None else ""
            }
            
            # Extract director
            director_elem = movie.find('director')
            if director_elem is not None:
                director_name = director_elem.find('name')
                movie_data["director"] = director_name.text if director_name is not None else ""
            else:
                movie_data["director"] = ""
                
            # Extract first actor
            actors_elem = movie.find('actors')
            if actors_elem is not None:
                first_actor = actors_elem.find('actor')
                if first_actor is not None:
                    actor_name = first_actor.find('name')
                    movie_data["actor"] = actor_name.text if actor_name is not None else ""
                else:
                    movie_data["actor"] = ""
            else:
                movie_data["actor"] = ""
                
            data.append(movie_data)
        
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        pytest.skip(f"Failed to load XML data: {e}")


def load_correspondence_files(train_path: Path, test_path: Path) -> dict:
    """Load correspondence training and test files."""
    data = {}
    
    # Helper function to convert original XML IDs to PyDI format
    def convert_to_pydi_ids(df):
        """Convert original XML IDs to PyDI _id format."""
        # Create mapping from original IDs to PyDI IDs
        # Load the original XML data to get the mapping
        base_path = Path("input/movies/entitymatching/data")
        academy_path = base_path / "academy_awards.xml"
        actors_path = base_path / "actors.xml"
        
        if not academy_path.exists() or not actors_path.exists():
            return df  # Return as-is if XML files not found
        
        try:
            # Load XML data to create ID mappings
            academy_df = load_movie_xml_data(academy_path)
            actors_df = load_movie_xml_data(actors_path)
            
            # Create mappings from original ID to PyDI index-based ID
            academy_id_map = {row["id"]: f"academy_awards_{i:06d}" 
                            for i, row in academy_df.iterrows()}
            actors_id_map = {row["id"]: f"actors_{i:06d}" 
                           for i, row in actors_df.iterrows()}
            
            # Apply mappings to correspondence data
            df = df.copy()
            df["id1"] = df["id1"].map(academy_id_map).fillna(df["id1"])
            df["id2"] = df["id2"].map(actors_id_map).fillna(df["id2"])
            
            return df
        except Exception:
            return df  # Return as-is if conversion fails
    
    if train_path.exists():
        try:
            train_df = pd.read_csv(train_path, names=["id1", "id2", "label"])
            # Convert boolean True/False to 1/0
            train_df["label"] = train_df["label"].map({True: 1, False: 0})
            # Convert IDs to PyDI format
            train_df = convert_to_pydi_ids(train_df)
            data["train"] = train_df
        except Exception as e:
            pytest.skip(f"Failed to load training data: {e}")
    
    if test_path.exists():
        try:
            test_df = pd.read_csv(test_path, names=["id1", "id2", "label"])
            test_df["label"] = test_df["label"].map({True: 1, False: 0})
            # Convert IDs to PyDI format
            test_df = convert_to_pydi_ids(test_df)
            data["test"] = test_df
        except Exception as e:
            pytest.skip(f"Failed to load test data: {e}")
    
    return data


class TestMovieEntityMatchingIntegration:
    """Integration tests using the movie datasets (academy_awards + actors)."""
    
    @pytest.fixture(scope="class")
    def movie_datasets(self):
        """Load movie datasets for integration testing."""
        base_path = Path("input/movies/entitymatching/data")
        
        academy_path = base_path / "academy_awards.xml"
        actors_path = base_path / "actors.xml"
        
        academy_df = load_movie_xml_data(academy_path)
        actors_df = load_movie_xml_data(actors_path)
        
        # Set dataset names and ensure record IDs
        academy_df.attrs["dataset_name"] = "academy_awards"
        actors_df.attrs["dataset_name"] = "actors"
        
        academy_df = ensure_record_ids(academy_df)
        actors_df = ensure_record_ids(actors_df)
        
        return {
            "academy_awards": academy_df,
            "actors": actors_df
        }
    
    @pytest.fixture(scope="class")
    def movie_correspondences(self):
        """Load movie correspondence files."""
        base_path = Path("input/movies/entitymatching/splits")
        
        train_path = base_path / "gs_academy_awards_2_actors_training.csv"
        test_path = base_path / "gs_academy_awards_2_actors_test.csv"
        
        return load_correspondence_files(train_path, test_path)
    
    def test_rule_based_matching_workflow(self, movie_datasets, movie_correspondences):
        """Test complete rule-based matching workflow."""
        if "train" not in movie_correspondences:
            pytest.skip("Training correspondence data not available")
        
        # Get datasets
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Create comparators
        comparators = [
            StringComparator("title", similarity_function="jaro_winkler"),
            DateComparator("date", max_days_difference=365)  # 1 year tolerance
        ]
        
        # Create matcher
        matcher = RuleBasedMatcher()
        
        # Use training data as candidates (in real scenario, this would come from blocking)
        train_pairs = movie_correspondences["train"][["id1", "id2"]].head(50)  # Limit for speed
        
        # Perform matching
        matches = matcher.match(
            academy_df,
            actors_df,
            [train_pairs],
            comparators=comparators,
            weights=[0.8, 0.2],
            threshold=0.5
        )
        
        # Verify results
        assert isinstance(matches, pd.DataFrame)
        assert list(matches.columns) == ["id1", "id2", "score", "notes"]
        
        # Should have some matches
        if len(matches) > 0:
            # Check score validity
            assert all(0.5 <= score <= 1.0 for score in matches["score"])
            
            # Check ID format consistency
            assert all(id_val.startswith("academy_awards_") for id_val in matches["id1"])
            assert all(id_val.startswith("actors_") for id_val in matches["id2"])
    
    def test_ml_based_matching_workflow(self, movie_datasets, movie_correspondences):
        """Test complete ML-based matching workflow."""
        if "train" not in movie_correspondences or "test" not in movie_correspondences:
            pytest.skip("Training/test correspondence data not available")
        
        # Get datasets
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Limit data size for test performance
        train_data = movie_correspondences["train"].head(100)
        test_data = movie_correspondences["test"].head(20)
        
        # Create feature extractor
        comparators = [
            StringComparator("title"),
            DateComparator("date")
        ]
        feature_extractor = FeatureExtractor(comparators)
        
        # Extract training features
        train_pairs = train_data[["id1", "id2"]]
        train_labels = train_data["label"]
        
        train_features = feature_extractor.create_features(
            academy_df, actors_df, train_pairs, train_labels
        )
        
        # Prepare training data
        feature_columns = [col for col in train_features.columns if col not in ["id1", "id2", "label"]]
        X_train = train_features[feature_columns]
        y_train = train_features["label"]
        
        # Train a simple classifier
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            classifier = RandomForestClassifier(n_estimators=10, random_state=42)
            classifier.fit(X_train, y_train)
            
            # Create ML matcher
            ml_matcher = MLBasedMatcher(feature_extractor)
            
            # Test predictions
            test_pairs = test_data[["id1", "id2"]].head(10)
            
            matches = ml_matcher.match(
                academy_df,
                actors_df,
                [test_pairs],
                trained_classifier=classifier,
                threshold=0.5
            )
            
            # Verify results
            assert isinstance(matches, pd.DataFrame)
            assert list(matches.columns) == ["id1", "id2", "score", "notes"]
            
            # Check metadata
            assert "classifier_type" in matches.attrs
            assert matches.attrs["classifier_type"] == "RandomForestClassifier"
            
        except ImportError:
            pytest.skip("scikit-learn not available for ML testing")
    
    def test_evaluation_workflow(self, movie_datasets, movie_correspondences):
        """Test evaluation workflow with real data."""
        if "test" not in movie_correspondences:
            pytest.skip("Test correspondence data not available")
        
        # Get datasets
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Create some mock correspondences for evaluation
        test_data = movie_correspondences["test"].head(50)
        
        # Create simple rule-based matches
        comparators = [StringComparator("title")]
        matcher = RuleBasedMatcher()
        
        test_pairs = test_data[["id1", "id2"]]
        matches = matcher.match(
            academy_df,
            actors_df,
            [test_pairs],
            comparators=comparators,
            weights=[1.0],
            threshold=0.1  # Low threshold to get some results
        )
        
        # Evaluate results
        test_ground_truth = test_data[["id1", "id2", "label"]]
        
        if len(matches) > 0:
            evaluation_results = EntityMatchingEvaluator.evaluate(
                matches,
                test_ground_truth
            )
            
            # Check evaluation results
            assert isinstance(evaluation_results, dict)
            assert "precision" in evaluation_results
            assert "recall" in evaluation_results
            assert "f1" in evaluation_results
            
            # Values should be reasonable
            assert 0.0 <= evaluation_results["precision"] <= 1.0
            assert 0.0 <= evaluation_results["recall"] <= 1.0
            assert 0.0 <= evaluation_results["f1"] <= 1.0
    
    def test_cluster_analysis_workflow(self, movie_datasets, movie_correspondences):
        """Test cluster consistency analysis workflow."""
        if "train" not in movie_correspondences:
            pytest.skip("Training correspondence data not available")
        
        # Get datasets
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Create correspondences using rule-based matching
        comparators = [StringComparator("title")]
        matcher = RuleBasedMatcher()
        
        # Use subset of training data
        train_pairs = movie_correspondences["train"][["id1", "id2"]].head(30)
        
        matches = matcher.match(
            academy_df,
            actors_df,
            [train_pairs],
            comparators=comparators,
            weights=[1.0],
            threshold=0.2
        )
        
        if len(matches) > 0:
            # Analyze cluster consistency
            cluster_report = EntityMatchingEvaluator.create_cluster_consistency_report(
                matches
            )
            
            # Verify cluster analysis results
            assert isinstance(cluster_report, pd.DataFrame)
            
            expected_columns = [
                "cluster_id", "cluster_size", "total_edges", "expected_edges",
                "consistency_ratio", "is_consistent"
            ]
            for col in expected_columns:
                assert col in cluster_report.columns
            
            # Check cluster size distribution
            size_distribution = EntityMatchingEvaluator.create_cluster_size_distribution(
                matches
            )
            
            assert isinstance(size_distribution, pd.DataFrame)
            assert "cluster_size" in size_distribution.columns
            assert "frequency" in size_distribution.columns
            assert "percentage" in size_distribution.columns
    
    def test_threshold_analysis_workflow(self, movie_datasets, movie_correspondences):
        """Test threshold sweep analysis workflow."""
        if "test" not in movie_correspondences:
            pytest.skip("Test correspondence data not available")
        
        # Get datasets
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Create correspondences with various scores
        comparators = [StringComparator("title")]
        matcher = RuleBasedMatcher()
        
        test_pairs = movie_correspondences["test"][["id1", "id2"]].head(20)
        
        matches = matcher.match(
            academy_df,
            actors_df,
            [test_pairs],
            comparators=comparators,
            weights=[1.0],
            threshold=0.0  # Get all matches
        )
        
        if len(matches) > 0:
            # Perform threshold sweep
            test_ground_truth = movie_correspondences["test"][["id1", "id2", "label"]].head(20)
            
            thresholds = [0.0, 0.3, 0.6, 0.9]
            sweep_results = EntityMatchingEvaluator.threshold_sweep(
                matches,
                test_ground_truth,
                thresholds=thresholds
            )
            
            # Verify results
            assert isinstance(sweep_results, pd.DataFrame)
            
            expected_columns = [
                "threshold", "precision", "recall", "f1",
                "correspondences_count"
            ]
            for col in expected_columns:
                assert col in sweep_results.columns
            
            # Should generally see decreasing correspondence count with higher threshold
            if len(sweep_results) > 1:
                counts = sweep_results.sort_values("threshold")["correspondences_count"].tolist()
                # Allow some flexibility as scores might be similar
                assert counts == sorted(counts, reverse=True) or len(set(counts)) == 1
    
    def test_end_to_end_workflow(self, movie_datasets, movie_correspondences, temp_output_dir):
        """Test complete end-to-end entity matching workflow."""
        if "train" not in movie_correspondences or "test" not in movie_correspondences:
            pytest.skip("Complete correspondence data not available")
        
        # Get datasets
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Limit data for performance
        train_data = movie_correspondences["train"].head(50)
        test_data = movie_correspondences["test"].head(20)
        
        # Step 1: Rule-based matching
        comparators = [
            StringComparator("title"),
            DateComparator("date", max_days_difference=730)  # 2 years
        ]
        
        rule_matcher = RuleBasedMatcher()
        train_pairs = train_data[["id1", "id2"]]
        
        rule_matches = rule_matcher.match(
            academy_df,
            actors_df,
            [train_pairs],
            comparators=comparators,
            weights=[0.7, 0.3],
            threshold=0.4
        )
        
        # Step 2: Evaluation
        if len(rule_matches) > 0:
            train_ground_truth = train_data[["id1", "id2", "label"]]
            
            eval_results = EntityMatchingEvaluator.evaluate(
                rule_matches,
                train_ground_truth,
                out_dir=str(temp_output_dir / "evaluation")
            )
            
            # Check that evaluation files were created
            assert "output_files" in eval_results
            for file_path in eval_results["output_files"]:
                assert Path(file_path).exists()
        
        # Step 3: Cluster analysis with output
        if len(rule_matches) > 0:
            cluster_report = EntityMatchingEvaluator.create_cluster_consistency_report(
                rule_matches,
                out_dir=str(temp_output_dir / "clusters")
            )
            
            # Check cluster analysis files
            cluster_dir = temp_output_dir / "clusters"
            assert (cluster_dir / "cluster_consistency_report.csv").exists()
            assert (cluster_dir / "cluster_analysis_summary.json").exists()
        
        # Step 4: Record groups export
        if len(rule_matches) > 0:
            groups_file = temp_output_dir / "record_groups.json"
            EntityMatchingEvaluator.write_record_groups_by_consistency(
                str(groups_file),
                rule_matches
            )
            
            assert groups_file.exists()
    
    def test_data_quality_checks(self, movie_datasets):
        """Test data quality and consistency checks."""
        academy_df = movie_datasets["academy_awards"]
        actors_df = movie_datasets["actors"]
        
        # Check that datasets have required structure
        assert "_id" in academy_df.columns
        assert "_id" in actors_df.columns
        assert "title" in academy_df.columns
        assert "title" in actors_df.columns
        
        # Check dataset names in attrs
        assert academy_df.attrs["dataset_name"] == "academy_awards"
        assert actors_df.attrs["dataset_name"] == "actors"
        
        # Check ID format consistency
        academy_ids = academy_df["_id"].tolist()
        actors_ids = actors_df["_id"].tolist()
        
        assert all(id_val.startswith("academy_awards_") for id_val in academy_ids)
        assert all(id_val.startswith("actors_") for id_val in actors_ids)
        
        # Check for duplicate IDs
        assert len(academy_ids) == len(set(academy_ids))
        assert len(actors_ids) == len(set(actors_ids))
        
        # Basic data completeness check
        if len(academy_df) > 0:
            title_completeness = (academy_df["title"].notna() & (academy_df["title"] != "")).mean()
            assert title_completeness > 0.5  # At least 50% should have titles
        
        if len(actors_df) > 0:
            title_completeness = (actors_df["title"].notna() & (actors_df["title"] != "")).mean()
            assert title_completeness > 0.5


class TestEntityMatchingPerformance:
    """Performance and scalability tests for entity matching."""
    
    def test_large_candidate_set_handling(self, sample_movies_left, sample_movies_right):
        """Test handling of large candidate sets."""
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Create large candidate set (Cartesian product)
        left_ids = sample_movies_left["_id"].tolist()
        right_ids = sample_movies_right["_id"].tolist()
        
        large_candidates = []
        for left_id in left_ids:
            for right_id in right_ids:
                large_candidates.append({"id1": left_id, "id2": right_id})
        
        candidate_df = pd.DataFrame(large_candidates)
        
        # Test rule-based matcher with large candidate set
        comparators = [StringComparator("title")]
        matcher = RuleBasedMatcher()
        
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            [candidate_df],
            comparators=comparators,
            weights=[1.0],
            threshold=0.8  # High threshold to reduce results
        )
        
        # Should handle large candidate set without errors
        assert isinstance(matches, pd.DataFrame)
    
    def test_batch_processing(self, sample_movies_left, sample_movies_right, candidate_pair_batches):
        """Test batch processing of candidate pairs."""
        # Prepare datasets
        sample_movies_left = ensure_record_ids(sample_movies_left)
        sample_movies_right = ensure_record_ids(sample_movies_right)
        
        # Create more batches
        additional_batches = []
        for i in range(5):
            batch = pd.DataFrame({
                "id1": [f"academy_awards_00000{i%4}"],
                "id2": [f"actors_00000{i%4}"]
            })
            additional_batches.append(batch)
        
        all_batches = candidate_pair_batches + additional_batches
        
        # Test with multiple batches
        comparators = [StringComparator("title")]
        matcher = RuleBasedMatcher()
        
        matches = matcher.match(
            sample_movies_left,
            sample_movies_right,
            all_batches,
            comparators=comparators,
            weights=[1.0],
            threshold=0.1
        )
        
        # Should process all batches
        assert isinstance(matches, pd.DataFrame)