#!/usr/bin/env python3
"""
Test script to verify enhanced tokenized_match with threshold and SimilarityRegistry.
"""

from PyDI.fusion.evaluation import tokenized_match
from functools import partial

def test_enhanced_tokenized_match():
    """Test tokenized_match with various threshold values."""
    
    print("Testing enhanced tokenized_match with thresholds...")
    
    # Test cases: (fused_value, gold_value, threshold, expected_result, description)
    test_cases = [
        # List to List comparisons with thresholds
        (['A', 'B'], ['A', 'B'], 1.0, True, "Identical lists, exact threshold"),
        (['A', 'B'], ['A', 'B'], 0.8, True, "Identical lists, lower threshold"),
        (['A', 'B'], ['A', 'B', 'C'], 1.0, False, "Subset vs superset, exact threshold"),
        (['A', 'B'], ['A', 'B', 'C'], 0.7, False, "Subset vs superset, 0.7 threshold (Jaccard=2/3=0.67)"),
        (['A', 'B'], ['A', 'B', 'C'], 0.6, True, "Subset vs superset, 0.6 threshold (Jaccard=2/3=0.67)"),
        (['A', 'B'], ['A', 'C'], 1.0, False, "Half different, exact threshold"),
        (['A', 'B'], ['A', 'C'], 0.5, False, "Half different, 0.5 threshold (Jaccard=1/3=0.33)"),
        (['A', 'B'], ['A', 'C'], 0.3, True, "Half different, 0.3 threshold (Jaccard=1/3=0.33)"),
        
        # String tokenization with thresholds  
        ('The Matrix', 'Matrix, The', 1.0, True, "Reordered tokens, exact threshold"),
        ('The Matrix', 'Matrix Action', 1.0, False, "Partial tokens, exact threshold"),
        ('The Matrix', 'Matrix Action', 0.5, False, "Partial tokens, 0.5 threshold (Jaccard=1/3=0.33)"),
        ('The Matrix', 'Matrix Action', 0.3, True, "Partial tokens, 0.3 threshold (Jaccard=1/3=0.33)"),
        ('Action Matrix', 'Matrix Action', 1.0, True, "Same tokens different order, exact"),
        ('The Dark Knight', 'Dark Knight Batman', 0.8, False, "2/4 common tokens (Jaccard=2/4=0.5)"),
        ('The Dark Knight', 'Dark Knight Batman', 0.5, True, "2/4 common tokens (Jaccard=2/4=0.5)"),
        
        # Mixed list/string with thresholds
        (['Tom Hanks'], 'Tom Hanks', 1.0, True, "Single item list vs string, exact"),
        (['Tom Hanks'], 'Tom Hanks', 0.8, True, "Single item list vs string, lower threshold"),
        (['Tom Hanks', 'Meg Ryan'], 'Tom Hanks', 1.0, False, "Multi-item list vs single string, exact"),
        (['Tom Hanks', 'Meg Ryan'], 'Tom Hanks', 0.5, True, "Multi-item list vs single string, 0.5 (Jaccard=1/2=0.5)"),
        (['Tom Hanks', 'Meg Ryan'], 'Tom Hanks', 0.3, True, "Multi-item list vs single string, 0.3 (Jaccard=1/3=0.33)"),
        
        # Edge cases
        ([], [], 1.0, True, "Both empty lists"),
        ([''], [''], 1.0, True, "Lists with empty strings"),
        (None, None, 1.0, True, "Both None values"),
        (['A'], [], 1.0, False, "Non-empty vs empty list"),
        (['A'], [], 0.0, False, "Non-empty vs empty list, zero threshold"),
    ]
    
    print(f"\\nRunning {len(test_cases)} test cases...")
    
    passed = 0
    failed = 0
    
    for i, (fused, gold, thresh, expected, description) in enumerate(test_cases, 1):
        try:
            result = tokenized_match(fused, gold, threshold=thresh)
            status = "✅ PASS" if result == expected else "❌ FAIL"
            if result != expected:
                failed += 1
                print(f"{i:2d}. {status} | {description}")
                print(f"    Threshold: {thresh}, Expected: {expected}, Got: {result}")
                print(f"    Fused: {repr(fused)}, Gold: {repr(gold)}")
                print()
            else:
                passed += 1
                print(f"{i:2d}. {status} | {description} (threshold={thresh})")
        except Exception as e:
            failed += 1
            print(f"{i:2d}. ❌ ERROR | {description}")
            print(f"    Exception: {e}")
            print(f"    Threshold: {thresh}, Fused: {repr(fused)}, Gold: {repr(gold)}")
            print()
    
    print(f"\\nTest Results: {passed} passed, {failed} failed")
    return failed == 0

def test_backward_compatibility():
    """Test that default threshold=1.0 maintains backward compatibility."""
    print("\\n" + "="*60)
    print("Testing backward compatibility (default threshold=1.0)...")
    
    # These should behave exactly like the old tokenized_match
    test_cases = [
        (['A', 'B'], ['A', 'B'], True, "Identical lists"),
        (['A', 'B'], ['B', 'A'], True, "Same lists different order"),
        (['A', 'B'], ['A', 'B', 'C'], False, "Subset vs superset"),
        ('The Matrix', 'Matrix, The', True, "Reordered title tokens"),
        ('The Matrix', 'Matrix Action', False, "Different tokens"),
        (['Tom Hanks'], 'Tom Hanks', True, "Single item list vs string"),
    ]
    
    all_passed = True
    for i, (fused, gold, expected, description) in enumerate(test_cases, 1):
        # Test with default threshold (should be 1.0)
        result_default = tokenized_match(fused, gold)  # No threshold specified
        result_explicit = tokenized_match(fused, gold, threshold=1.0)  # Explicit 1.0
        
        if result_default == expected and result_explicit == expected and result_default == result_explicit:
            print(f"{i}. ✅ PASS | {description}")
        else:
            print(f"{i}. ❌ FAIL | {description}")
            print(f"   Expected: {expected}")
            print(f"   Default threshold: {result_default}")
            print(f"   Explicit 1.0: {result_explicit}")
            all_passed = False
    
    return all_passed

def test_with_partial_functions():
    """Test using tokenized_match with functools.partial for different thresholds."""
    print("\\n" + "="*60)
    print("Testing with functools.partial (strategy usage pattern)...")
    
    # Create different evaluators with different thresholds
    exact_match = tokenized_match  # Default threshold=1.0
    loose_match = partial(tokenized_match, threshold=0.6)
    very_loose_match = partial(tokenized_match, threshold=0.3)
    
    # Test data
    fused_actors = ['Tom Hanks', 'Meg Ryan']
    gold_actors = ['Tom Hanks', 'Meg Ryan', 'Greg Kinnear']
    
    print(f"Fused: {fused_actors}")
    print(f"Gold:  {gold_actors}")
    print(f"(Jaccard similarity = 2/3 = 0.67)")
    print()
    
    # Test each evaluator
    exact_result = exact_match(fused_actors, gold_actors)
    loose_result = loose_match(fused_actors, gold_actors) 
    very_loose_result = very_loose_match(fused_actors, gold_actors)
    
    print(f"Exact match (threshold=1.0):     {exact_result}")
    print(f"Loose match (threshold=0.6):     {loose_result}")
    print(f"Very loose match (threshold=0.3): {very_loose_result}")
    
    # Expected: False, True, True
    expected_results = [False, True, True]
    actual_results = [exact_result, loose_result, very_loose_result]
    
    success = actual_results == expected_results
    print(f"\\nResults match expected: {'✅ PASS' if success else '❌ FAIL'}")
    if not success:
        print(f"Expected: {expected_results}")
        print(f"Actual:   {actual_results}")
    
    return success

if __name__ == "__main__":
    print("Testing Enhanced tokenized_match with Threshold and SimilarityRegistry")
    print("="*70)
    
    success1 = test_enhanced_tokenized_match()
    success2 = test_backward_compatibility() 
    success3 = test_with_partial_functions()
    
    if success1 and success2 and success3:
        print("\\n🎉 ALL TESTS PASSED!")
        print("✅ Enhanced tokenized_match with thresholds is working correctly")
        print("✅ Backward compatibility maintained")  
        print("✅ Ready for use with functools.partial in fusion strategies")
    else:
        print("\\n❌ SOME TESTS FAILED!")
        print("Please review the implementation.")