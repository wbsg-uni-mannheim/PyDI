#!/usr/bin/env python3
"""
Test script to verify tokenized_match works with different input types.
"""

from PyDI.fusion.evaluation import tokenized_match

def test_tokenized_match():
    """Test tokenized_match with various input combinations."""
    
    print("Testing tokenized_match function...")
    
    # Test cases: (fused_value, gold_value, expected_result, description)
    test_cases = [
        # List to List comparisons
        (['Tom Hanks', 'Meg Ryan'], ['Tom Hanks', 'Meg Ryan'], True, "Identical lists in same order"),
        (['Tom Hanks', 'Meg Ryan'], ['Meg Ryan', 'Tom Hanks'], True, "Same lists in different order"),
        (['Tom Hanks'], ['Tom Hanks'], True, "Single item lists"),
        (['Tom Hanks', 'Meg Ryan'], ['Tom Hanks'], False, "Different length lists"),
        (['Tom Hanks', 'Meg Ryan'], ['Tom Cruise', 'Meg Ryan'], False, "Partially different lists"),
        
        # Mixed List/String comparisons  
        (['Tom Hanks'], 'Tom Hanks', True, "Single item list vs matching string"),
        ('Tom Hanks', ['Tom Hanks'], True, "String vs matching single item list"),
        (['Tom Hanks', 'Meg Ryan'], 'Tom Hanks', False, "Multi-item list vs single string"),
        
        # String to String comparisons (original functionality)
        ('The Matrix', 'Matrix, The', True, "Reordered title tokens"),
        ('Tom Hanks', 'Tom Hanks', True, "Identical strings"),
        ('Tom Hanks', 'Tom Cruise', False, "Different names"),
        ('The Dark Knight', 'Dark Knight, The', True, "Reordered movie title"),
        
        # Empty/None cases
        ([], [], True, "Both empty lists"),
        (None, None, True, "Both None"),
        (['Tom Hanks'], [], False, "Non-empty vs empty list"),
        (['Tom Hanks'], None, False, "List vs None"),
        
        # Edge cases
        ([''], [''], True, "Lists with empty strings"),
        (['Tom Hanks', ''], ['Tom Hanks'], False, "List with empty string vs without"),
    ]
    
    print(f"\\nRunning {len(test_cases)} test cases...")
    
    passed = 0
    failed = 0
    
    for i, (fused, gold, expected, description) in enumerate(test_cases, 1):
        try:
            result = tokenized_match(fused, gold)
            status = "✅ PASS" if result == expected else "❌ FAIL"
            if result != expected:
                failed += 1
                print(f"{i:2d}. {status} | {description}")
                print(f"    Expected: {expected}, Got: {result}")
                print(f"    Fused: {repr(fused)}, Gold: {repr(gold)}")
            else:
                passed += 1
                print(f"{i:2d}. {status} | {description}")
        except Exception as e:
            failed += 1
            print(f"{i:2d}. ❌ ERROR | {description}")
            print(f"    Exception: {e}")
            print(f"    Fused: {repr(fused)}, Gold: {repr(gold)}")
    
    print(f"\\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tokenized_match tests PASSED!")
        return True
    else:
        print("❌ Some tokenized_match tests FAILED!")
        return False

def test_real_world_scenarios():
    """Test with real-world actor list scenarios."""
    
    print("\\n" + "="*60)
    print("Testing real-world actor list scenarios...")
    
    # Scenarios that would occur with actor lists from fusion
    scenarios = [
        # Scenario 1: Perfect match  
        (['F. Murray Abraham', 'Tom Hulce'], ['F. Murray Abraham', 'Tom Hulce'], True,
         "Perfect match - same actors same order"),
        
        # Scenario 2: Different order
        (['F. Murray Abraham', 'Tom Hulce'], ['Tom Hulce', 'F. Murray Abraham'], True,
         "Same actors different order"),
        
        # Scenario 3: Subset match (fusion selected fewer actors)
        (['F. Murray Abraham'], ['F. Murray Abraham', 'Tom Hulce'], False,
         "Fusion selected subset of actors"),
        
        # Scenario 4: Different actors
        (['Leonardo DiCaprio'], ['F. Murray Abraham', 'Tom Hulce'], False,
         "Completely different actors"),
        
        # Scenario 5: Mixed case should work
        (['tom hanks', 'MEG RYAN'], ['Tom Hanks', 'Meg Ryan'], False,  # Lists use exact comparison
         "Mixed case actors (lists use exact comparison)"),
    ]
    
    print(f"Running {len(scenarios)} real-world scenarios...")
    
    for i, (fused, gold, expected, description) in enumerate(scenarios, 1):
        result = tokenized_match(fused, gold)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{i}. {status} | {description}")
        if result != expected:
            print(f"   Expected: {expected}, Got: {result}")
        print(f"   Fused: {fused}")
        print(f"   Gold:  {gold}")
        print()


if __name__ == "__main__":
    success = test_tokenized_match()
    test_real_world_scenarios()
    
    if success:
        print("\\n🎉 tokenized_match is now ready for list-based evaluation!")
    else:
        print("\\n❌ tokenized_match still needs fixes!")