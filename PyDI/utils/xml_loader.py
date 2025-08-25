"""
XML data loading utilities for PyDI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import xml.etree.ElementTree as ET

import pandas as pd


def load_xml_to_dataframe(
    xml_path: Union[str, Path],
    dataset_name: Optional[str] = None,
    record_tag: str = "release",
    id_column: str = "id",
    flatten_nested: bool = True,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """Load XML file into a pandas DataFrame.
    
    Parameters
    ----------
    xml_path : str or Path
        Path to the XML file.
    dataset_name : str, optional
        Name to assign to the dataset (stored in df.attrs).
    record_tag : str, optional
        XML tag that represents individual records. Default is "release".
    id_column : str, optional
        Name of the ID column to create. Default is "id".
    flatten_nested : bool, optional
        Whether to flatten nested structures like tracks. Default is True.
    max_records : int, optional
        Maximum number of records to load.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with the XML data loaded.
    """
    xml_path = Path(xml_path)
    
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    logging.info(f"Loading XML file: {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")
    
    records = []
    
    # Find all record elements
    if root.tag == record_tag:
        # Root is the record container
        record_elements = [root]
    else:
        # Look for record elements in the tree
        record_elements = root.findall(f".//{record_tag}")
        if not record_elements:
            # Try direct children
            record_elements = [child for child in root if child.tag == record_tag]
    
    if not record_elements:
        logging.warning(f"No elements with tag '{record_tag}' found in XML")
        return pd.DataFrame()
    
    logging.info(f"Found {len(record_elements)} records")
    
    for i, record_elem in enumerate(record_elements):
        if max_records and i >= max_records:
            break
            
        record_data = _extract_record_data(record_elem, flatten_nested)
        records.append(record_data)
    
    if not records:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Set dataset metadata
    if dataset_name is None:
        dataset_name = xml_path.stem
    
    df.attrs["dataset_name"] = dataset_name
    df.attrs["source"] = {
        "path": str(xml_path),
        "format": "xml",
        "record_tag": record_tag
    }
    
    # Add global record IDs
    if "_id" not in df.columns:
        df["_id"] = [f"{dataset_name}_{i:06d}" for i in range(len(df))]
    
    logging.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df


def _extract_record_data(element: ET.Element, flatten_nested: bool = True) -> Dict[str, str]:
    """Extract data from an XML element into a flat dictionary."""
    data = {}
    
    # Process direct child elements
    for child in element:
        if child.tag == "tracks" and flatten_nested:
            # Special handling for tracks - flatten into the main record
            track_data = _extract_tracks_data(child)
            data.update(track_data)
        elif len(child) == 0:
            # Leaf node - get text content
            data[child.tag] = child.text or ""
        elif flatten_nested:
            # Nested structure - try to flatten
            nested_data = _extract_record_data(child, flatten_nested=True)
            for key, value in nested_data.items():
                data[f"{child.tag}_{key}"] = value
        else:
            # Keep as nested structure (convert to string)
            data[child.tag] = ET.tostring(child, encoding="unicode")
    
    return data


def _extract_tracks_data(tracks_element: ET.Element) -> Dict[str, str]:
    """Extract track information and aggregate into main record fields."""
    track_data = {}
    
    tracks = tracks_element.findall("track")
    if not tracks:
        return track_data
    
    # Aggregate track information
    track_names = []
    track_durations = []
    track_positions = []
    
    for track in tracks:
        name_elem = track.find("name")
        if name_elem is not None and name_elem.text:
            track_names.append(name_elem.text)
        
        duration_elem = track.find("duration") 
        if duration_elem is not None and duration_elem.text:
            try:
                duration = float(duration_elem.text)
                track_durations.append(duration)
            except (ValueError, TypeError):
                pass
        
        position_elem = track.find("position")
        if position_elem is not None and position_elem.text:
            track_positions.append(position_elem.text)
    
    # Store aggregated track data
    if track_names:
        track_data["track_names"] = " | ".join(track_names)
        track_data["track_count"] = str(len(track_names))
    
    if track_durations:
        track_data["total_track_duration"] = str(sum(track_durations))
        track_data["avg_track_duration"] = str(sum(track_durations) / len(track_durations))
    
    if track_positions:
        track_data["track_positions"] = " | ".join(track_positions)
    
    return track_data


def load_music_datasets(input_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load the music schema matching datasets.
    
    Parameters
    ---------- 
    input_dir : str or Path
        Path to the input directory containing the music datasets.
    
    Returns
    -------
    dict
        Dictionary mapping dataset names to DataFrames.
    """
    input_dir = Path(input_dir)
    schemamatching_dir = input_dir / "music" / "schemamatching" / "data"
    
    if not schemamatching_dir.exists():
        raise FileNotFoundError(f"Music datasets directory not found: {schemamatching_dir}")
    
    datasets = {}
    
    # Load the three music datasets
    dataset_files = {
        "discogs": "discogs.xml",
        "lastfm": "lastfm.xml", 
        "musicbrainz": "musicbrainz.xml"
    }
    
    for dataset_name, filename in dataset_files.items():
        file_path = schemamatching_dir / filename
        if file_path.exists():
            try:
                if dataset_name == "musicbrainz":
                    # MusicBrainz has a different structure
                    df = load_xml_to_dataframe(
                        file_path,
                        dataset_name=dataset_name,
                        record_tag="release",
                        flatten_nested=True,
                        max_records=100  # Limit for testing
                    )
                elif dataset_name == "discogs":
                    # Discogs uses "album" as record tag
                    df = load_xml_to_dataframe(
                        file_path,
                        dataset_name=dataset_name,
                        record_tag="album",
                        flatten_nested=True
                    )
                else:
                    # LastFM uses standard structure
                    df = load_xml_to_dataframe(
                        file_path,
                        dataset_name=dataset_name,
                        record_tag="album" if "album" in filename else "release",
                        flatten_nested=True
                    )
                
                datasets[dataset_name] = df
                logging.info(f"Loaded {dataset_name}: {len(df)} records")
                
            except Exception as e:
                logging.error(f"Failed to load {dataset_name}: {e}")
        else:
            logging.warning(f"Dataset file not found: {file_path}")
    
    return datasets