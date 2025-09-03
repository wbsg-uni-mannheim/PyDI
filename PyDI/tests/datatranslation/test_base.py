"""Tests for base data translation classes."""

import pandas as pd
import pytest
from abc import ABC

from PyDI.datatranslation.base import BaseTranslator
from PyDI.schemamatching.base import SchemaMapping


class ConcreteTranslator(BaseTranslator):
    """Concrete implementation for testing the abstract base class."""
    
    def translate(self, df: pd.DataFrame, corr: SchemaMapping) -> pd.DataFrame:
        """Simple test implementation that returns the input DataFrame unchanged."""
        return df.copy()


class TestBaseTranslator:
    """Test the BaseTranslator abstract base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseTranslator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTranslator()
    
    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        translator = ConcreteTranslator()
        assert isinstance(translator, BaseTranslator)
    
    def test_translate_method_signature(self, movies_df_for_translation, sample_schema_mapping):
        """Test that the translate method has the correct signature."""
        translator = ConcreteTranslator()
        result = translator.translate(movies_df_for_translation, sample_schema_mapping)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should be a copy (different object)
        assert result is not movies_df_for_translation
        
        # Should have the same data
        pd.testing.assert_frame_equal(result, movies_df_for_translation)
    
    def test_abstract_method_must_be_implemented(self):
        """Test that subclasses must implement the translate method."""
        
        class IncompleteTranslator(BaseTranslator):
            pass
            
        with pytest.raises(TypeError):
            IncompleteTranslator()
    
    def test_multiple_concrete_implementations(self, movies_df_for_translation, sample_schema_mapping):
        """Test that multiple concrete implementations can coexist."""
        
        class AnotherTranslator(BaseTranslator):
            def translate(self, df: pd.DataFrame, corr: SchemaMapping) -> pd.DataFrame:
                return df.copy()
        
        translator1 = ConcreteTranslator()
        translator2 = AnotherTranslator()
        
        result1 = translator1.translate(movies_df_for_translation, sample_schema_mapping)
        result2 = translator2.translate(movies_df_for_translation, sample_schema_mapping)
        
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert isinstance(translator1, BaseTranslator)
        assert isinstance(translator2, BaseTranslator)