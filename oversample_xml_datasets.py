#!/usr/bin/env python3
"""
Script to load XML movie datasets, randomly oversample them to 100K entities,
and save them with _large.xml suffix.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from typing import List, Tuple

def parse_xml_dataset(file_path: str) -> Tuple[ET.Element, List[ET.Element]]:
    """
    Parse XML dataset and return root element and list of movie elements.
    
    Args:
        file_path: Path to the XML file
        
    Returns:
        Tuple of (root_element, list_of_movie_elements)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    movies = root.findall('movie')
    print(f"Loaded {len(movies)} movies from {file_path}")
    return root, movies

def get_next_id(movies: List[ET.Element], dataset_prefix: str) -> int:
    """
    Get the next available ID number for new records.
    
    Args:
        movies: List of movie elements
        dataset_prefix: Prefix used in IDs (e.g., 'academy_awards_')
        
    Returns:
        Next available ID number
    """
    max_id = 0
    for movie in movies:
        id_elem = movie.find('id')
        if id_elem is not None and id_elem.text:
            # Extract number from ID like 'academy_awards_1' -> 1
            id_parts = id_elem.text.split('_')
            if len(id_parts) >= 2 and id_parts[-1].isdigit():
                max_id = max(max_id, int(id_parts[-1]))
    return max_id + 1

def oversample_movies(movies: List[ET.Element], target_size: int, dataset_prefix: str) -> List[ET.Element]:
    """
    Randomly oversample movies to reach target size.
    
    Args:
        movies: Original list of movie elements
        target_size: Target number of movies (100K)
        dataset_prefix: Prefix for dataset IDs
        
    Returns:
        List of movies with target size
    """
    if len(movies) >= target_size:
        print(f"Dataset already has {len(movies)} movies, randomly sampling {target_size}")
        return random.sample(movies, target_size)
    
    print(f"Oversampling from {len(movies)} to {target_size} movies")
    
    # Calculate how many copies we need
    copies_needed = target_size - len(movies)
    next_id = get_next_id(movies, dataset_prefix)
    
    # Start with original movies
    result = movies.copy()
    
    # Add random samples with new IDs
    for i in range(copies_needed):
        # Randomly select a movie to duplicate
        source_movie = random.choice(movies)
        
        # Create a deep copy
        new_movie = ET.Element('movie')
        new_movie.text = source_movie.text
        new_movie.tail = source_movie.tail
        new_movie.attrib = source_movie.attrib.copy()
        
        # Copy all child elements
        for child in source_movie:
            new_child = ET.Element(child.tag)
            new_child.text = child.text
            new_child.tail = child.tail
            new_child.attrib = child.attrib.copy()
            
            # Handle nested elements (like actors)
            for grandchild in child:
                new_grandchild = ET.Element(grandchild.tag)
                new_grandchild.text = grandchild.text
                new_grandchild.tail = grandchild.tail
                new_grandchild.attrib = grandchild.attrib.copy()
                
                # Handle great-grandchildren (like actor names)
                for great_grandchild in grandchild:
                    new_great_grandchild = ET.Element(great_grandchild.tag)
                    new_great_grandchild.text = great_grandchild.text
                    new_great_grandchild.tail = great_grandchild.tail
                    new_great_grandchild.attrib = great_grandchild.attrib.copy()
                    new_grandchild.append(new_great_grandchild)
                
                new_child.append(new_grandchild)
            
            new_movie.append(new_child)
        
        # Update the ID to be unique
        id_elem = new_movie.find('id')
        if id_elem is not None:
            id_elem.text = f"{dataset_prefix}_{next_id + i}"
        
        result.append(new_movie)
        
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} additional movies...")
    
    return result

def save_xml_dataset(root: ET.Element, movies: List[ET.Element], output_path: str):
    """
    Save the oversampled dataset to XML file.
    
    Args:
        root: Original root element
        movies: List of movie elements (original + oversampled)
        output_path: Path to save the output file
    """
    # Create new root element with same attributes
    new_root = ET.Element(root.tag)
    new_root.attrib = root.attrib.copy()
    
    # Add all movies to the new root
    for movie in movies:
        new_root.append(movie)
    
    # Create tree and save
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"Saved {len(movies)} movies to {output_path}")

def main():
    """Main function to process all XML datasets."""
    input_dir = Path("PyDI/tutorial/input/movies/data")
    target_size = 100000
    
    # Process each XML file
    xml_files = list(input_dir.glob("*.xml"))
    
    for xml_file in xml_files:
        if xml_file.name.endswith("_large.xml"):
            print(f"Skipping already processed file: {xml_file.name}")
            continue
            
        print(f"\nProcessing {xml_file.name}...")
        
        # Extract dataset prefix from filename
        dataset_prefix = xml_file.stem
        
        # Load dataset
        root, movies = parse_xml_dataset(str(xml_file))
        
        # Oversample
        random.seed(42)  # For reproducibility
        oversampled_movies = oversample_movies(movies, target_size, dataset_prefix)
        
        # Save
        output_path = xml_file.parent / f"{xml_file.stem}_large.xml"
        save_xml_dataset(root, oversampled_movies, str(output_path))
        
    print("\nAll datasets processed successfully!")

if __name__ == "__main__":
    main()