"""Streamlit application for interactive entity matching with PyDI."""

from __future__ import annotations

from io import StringIO
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

try:  # pragma: no cover - optional keyboard shortcut helper
    import streamlit_hotkeys as hotkeys
except ImportError:  # pragma: no cover - dependency not installed
    hotkeys = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard for Streamlit execution
    from .interactive import MatchingSessionResult, run_interactive_matching
except ImportError:  # Allows running via `streamlit run PyDI/...`
    from PyDI.entitymatching.labeling_tool.interactive import MatchingSessionResult, run_interactive_matching

try:  # pragma: no cover - import guard for Streamlit execution
    from ...io import load_xml
except ImportError:
    from PyDI.io import load_xml


def _coerce_nested_to_strings(df: pd.DataFrame, separator: str = " | ") -> pd.DataFrame:
    """Convert nested list/tuple/set/dict values to readable strings."""

    def _normalize(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, pd.Series):
            return _normalize(value.dropna().tolist())

        if isinstance(value, np.ndarray):
            return _normalize(value.tolist())

        if isinstance(value, (list, tuple, set)):
            parts = [_normalize(item) for item in value if item is not None]
            cleaned = [part for part in parts if part not in (None, "")]
            if not cleaned:
                return None
            return separator.join(str(part) for part in cleaned)

        if isinstance(value, dict):
            parts = []
            for item in value.values():
                normalized = _normalize(item)
                if normalized not in (None, ""):
                    parts.append(normalized)
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return separator.join(str(part) for part in parts)

        text = str(value).strip()
        if isinstance(value, str):
            return text or None
        return value

    if df.empty:
        return df

    return df.applymap(_normalize)


def _prepare_dataset(
    df: pd.DataFrame,
    dataset_label: str,
    id_column: str = "pydi_id",
) -> pd.DataFrame:
    """Optionally add a generated identifier column to a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    dataset_label : str
        Prefix for generated identifiers.
    id_column : str, default "pydi_id"
        Column name to use for identifiers.
    """

    df = df.copy()
    if id_column in df.columns:
        return df

    generated_ids = [f"{dataset_label}_{i:06d}" for i in range(len(df))]
    df.insert(0, id_column, generated_ids)
    return df


def _dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _get_label_state() -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """Retrieve the shared annotation state dictionary."""
    return st.session_state.setdefault("entity_matching_labels", {})


def _set_label(pair_key: Tuple[Any, Any], label: Optional[str]) -> None:
    """Store or remove a label for a pair."""
    labels = _get_label_state()
    entry = labels.get(pair_key, {})
    if label:
        entry["label"] = label
        if "corner" not in entry:
            entry["corner"] = False
        labels[pair_key] = entry
    else:
        entry.pop("label", None)
        if not entry.get("corner"):
            labels.pop(pair_key, None)
        else:
            labels[pair_key] = entry


def _set_corner(pair_key: Tuple[Any, Any], flag: bool) -> None:
    """Record corner-case flag for a pair."""
    labels = _get_label_state()
    entry = labels.get(pair_key, {})
    entry["corner"] = bool(flag)
    if not entry.get("corner") and not entry.get("label"):
        labels.pop(pair_key, None)
    else:
        labels[pair_key] = entry


def _labels_state_to_dataframe(labels_state: Dict[Tuple[Any, Any], Dict[str, Any]]) -> pd.DataFrame:
    """Convert the session label state to a DataFrame for export."""
    if not labels_state:
        return pd.DataFrame(columns=["id1", "id2", "label", "corner"])
    labeled_rows = [
        {
            "id1": key[0],
            "id2": key[1],
            "label": value.get("label"),
            "corner": value.get("corner", False),
        }
        for key, value in labels_state.items()
        if value.get("label")
    ]
    if not labeled_rows:
        return pd.DataFrame(columns=["id1", "id2", "label", "corner"])
    return pd.DataFrame(labeled_rows)


_HOTKEY_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("swipe_non_match", "ArrowLeft", "ArrowLeft"),
    ("swipe_match", "ArrowRight", "ArrowRight"),
    ("swipe_corner", "Space", "Space"),
)

if hotkeys is not None:  # pragma: no cover - optional dependency
    _hotkey_specs: List[Any] = []
    for _name, _key, _ in _HOTKEY_DEFINITIONS:
        try:
            _hotkey_specs.append(
                hotkeys.hk(
                    _name,
                    _key,
                    prevent_default=True,
                    ignore_repeat=False,
                )
            )
        except Exception:
            continue
    if _hotkey_specs:
        try:
            hotkeys.activate(_hotkey_specs, key="pydi_swipe_hotkeys")
        except Exception:
            pass


def _read_hotkey_event() -> Optional[str]:
    """Return the keyboard event captured via streamlit-hotkeys if available."""
    if hotkeys is None:
        return None
    for name, _, event_value in _HOTKEY_DEFINITIONS:
        try:
            if hotkeys.pressed(name, key="pydi_swipe_hotkeys"):
                return event_value
        except Exception:
            continue
    return None


def _advance_swipe_index(offset: int, total: int) -> None:
    """Move the swipe cursor forward/backwards with wrap-around."""
    if total <= 0:
        st.session_state["swipe_index"] = 0
        return
    current = st.session_state.get("swipe_index", 0)
    st.session_state["swipe_index"] = (current + offset) % total


def _format_cell(value: Any) -> str:
    """Format a cell value for side-by-side rendering."""
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    text = str(value).strip()
    return text if text else "—"


def _capture_keyboard_event() -> Optional[str]:
    """Capture key presses for labeling shortcuts via a lightweight HTML component."""
    hotkey_event = _read_hotkey_event()
    if hotkey_event:
        return hotkey_event

    script = """
    <script>
    (function() {
        const globalFlag = "_pydiKeyboardListenerAttached";
        if (window[globalFlag]) {
            return;
        }
        window[globalFlag] = true;
        const sendValue = (value) => {
            if (window.Streamlit && window.Streamlit.setComponentValue) {
                window.Streamlit.setComponentValue(value);
            }
        };
        if (window.Streamlit && window.Streamlit.setComponentReady) {
            window.Streamlit.setComponentReady();
        }
        if (window.Streamlit && window.Streamlit.setFrameHeight) {
            window.Streamlit.setFrameHeight(0);
        }
        sendValue(null);
        const handler = function(event) {
            const key = event.key || event.code || "";
            try {
                const doc = window.document;
                const active = doc ? doc.activeElement : null;
                const activeTag = active && active.tagName ? active.tagName.toUpperCase() : "";
                if (["INPUT", "TEXTAREA", "SELECT"].includes(activeTag) || (active && active.isContentEditable)) {
                    return;
                }
            } catch (err) {
                // Ignore focus checks if cross-origin access is blocked.
            }
            if (["ArrowLeft", "ArrowRight", " ", "Spacebar", "Space"].includes(key)) {
                event.preventDefault();
                const value = key === " " || key === "Spacebar" ? "Space" : key;
                sendValue(value);
                setTimeout(function() { sendValue(null); }, 100);
            }
        };
        try {
            window.addEventListener("keydown", handler, true);
        } catch (err) {
            // ignore
        }
    })();
    </script>
    """
    event = st_html(script, height=0)
    return event if isinstance(event, str) and event else None


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def _load_demo_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Provide small movie datasets for quick experimentation."""
    left_records = [
        ("L1", "The Matrix", 1999, "The Wachowskis", "USA", "Sci-Fi"),
        ("L2", "Inception", 2010, "Christopher Nolan", "USA", "Sci-Fi"),
        ("L3", "Interstellar", 2014, "Christopher Nolan", "USA", "Sci-Fi"),
        ("L4", "The Dark Knight", 2008, "Christopher Nolan", "USA", "Action"),
        ("L5", "Memento", 2000, "Christopher Nolan", "USA", "Thriller"),
        ("L6", "Arrival", 2016, "Denis Villeneuve", "Canada", "Sci-Fi"),
        ("L7", "Blade Runner 2049", 2017, "Denis Villeneuve", "USA", "Sci-Fi"),
        ("L8", "Dune", 2021, "Denis Villeneuve", "USA", "Sci-Fi"),
        ("L9", "Tenet", 2020, "Christopher Nolan", "USA", "Sci-Fi"),
        ("L10", "The Prestige", 2006, "Christopher Nolan", "UK", "Drama"),
    ]
    right_records = [
        ("R1", "Matrix", 1999, "Wachowski Sisters", "United States", "Science Fiction"),
        ("R2", "Inception", 2010, "Christopher Nolan", "United States", "Science Fiction"),
        ("R3", "Interstellar", 2014, "Christopher Nolan", "United States", "Science Fiction"),
        ("R4", "The Dark Knight", 2008, "Christopher Nolan", "United States", "Action"),
        ("R5", "Momento", 2000, "Christopher Nolan", "United States", "Thriller"),
        ("R6", "Arrival", 2016, "Denis Villeneuve", "Canada", "Science Fiction"),
        ("R7", "Blade Runner 2049", 2017, "Denis Villeneuve", "United States", "Science Fiction"),
        ("R8", "Dune Part One", 2021, "Denis Villeneuve", "United States", "Science Fiction"),
        ("R9", "Tenet", 2020, "Christopher Nolan", "United States", "Science Fiction"),
        ("R10", "Prestige", 2006, "Christopher Nolan", "United Kingdom", "Drama"),
        ("R11", "John Wick", 2014, "Chad Stahelski", "United States", "Action"),
        ("R12", "The Matrix Reloaded", 2003, "The Wachowskis", "United States", "Science Fiction"),
    ]
    columns = ["entity_id", "title", "year", "director", "country", "genre"]
    return (
        pd.DataFrame(left_records, columns=columns),
        pd.DataFrame(right_records, columns=columns),
        "entity_id",
    )


# ---------------------------------------------------------------------------
# Sidebar builders
# ---------------------------------------------------------------------------


def _read_uploaded_dataset(upload: Any, dataset_label: str) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    filename = upload.name.lower()
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(upload)
        if filename.endswith(".tsv"):
            return pd.read_csv(upload, sep="\t")
        if filename.endswith(".xml"):
            tmp_path: Optional[Path] = None
            try:
                if hasattr(upload, "seek"):
                    try:
                        upload.seek(0)
                    except Exception:
                        pass

                if isinstance(upload, (str, os.PathLike)):
                    xml_source: Any = upload
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        if hasattr(upload, "read"):
                            shutil.copyfileobj(upload, tmp_file)
                        else:
                            data = getattr(upload, "getvalue", lambda: b"")()
                            tmp_file.write(data)
                    xml_source = tmp_path

                df_xml = load_xml(
                    xml_source,
                    nested_handling="aggregate",
                    add_index=False,
                    include_provenance_columns=False,
                )
                df_xml = df_xml.rename(columns=lambda col: col.replace("_text", "") if isinstance(col, str) else col)
                df_xml = _coerce_nested_to_strings(df_xml)
                return df_xml
            finally:
                if hasattr(upload, "seek"):
                    try:
                        upload.seek(0)
                    except Exception:
                        pass

                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass
                    except PermissionError:
                        pass
        if filename.endswith(".json"):
            return pd.read_json(upload)
        if filename.endswith(".parquet"):
            return pd.read_parquet(upload)
        # Try CSV as fallback
        return pd.read_csv(upload)
    except Exception as exc:
        raise ValueError(f"Failed to load {upload.name}: {exc}") from exc


def _build_blocking_config(
    id_column: str,
    shared_columns: Sequence[str],
    string_columns: Sequence[str],
) -> Tuple[List[Dict[str, Any]], bool]:
    st.sidebar.subheader("Blocking")

    disable_blocking = st.sidebar.checkbox(
        "Run without blocking (full cartesian product)", value=False, help="Generate candidates without any blocking."
    )

    configs: List[Dict[str, Any]] = []

    if disable_blocking:
        return configs, True

    if st.sidebar.checkbox("Standard (exact key)", value=bool(shared_columns), help="Exact match on selected columns"):
        selected = st.sidebar.multiselect(
            "Standard key columns",
            options=shared_columns,
            default=[col for col in shared_columns if col != id_column][:1],
        )
        if selected:
            configs.append(
                {
                    "type": "standard",
                    "name": "StandardBlocker",
                    "params": {
                        "on": selected,
                    },
                }
            )

    if string_columns and st.sidebar.checkbox("Token overlap", value=False, help="Token-based blocking on a chosen text column"):
        token_col = st.sidebar.selectbox("Token column", options=string_columns)
        ngram_size = st.sidebar.slider("Token n-gram size", 1, 6, 1)
        ngram_type = st.sidebar.selectbox("Token n-gram type", options=["character", "word"])
        configs.append(
            {
                "type": "token",
                "name": "TokenBlocker",
                "params": {
                    "column": token_col,
                    "ngram_size": ngram_size if ngram_size > 1 else None,
                    "ngram_type": ngram_type if ngram_size > 1 else None,
                },
            }
        )

    if shared_columns and st.sidebar.checkbox("Sorted neighbourhood", value=False, help="Sliding window over a sort key"):
        sort_key = st.sidebar.selectbox("Sort key", options=shared_columns)
        window = st.sidebar.slider("Window size", min_value=2, max_value=20, value=5, step=1)
        configs.append(
            {
                "type": "sorted_neighbourhood",
                "name": "SortedNeighbourhoodBlocker",
                "params": {
                    "key": sort_key,
                    "window": window,
                },
            }
        )

    if string_columns and st.sidebar.checkbox("Embedding ANN", value=False, help="Sentence-transformer embeddings with ANN search"):
        embed_cols = st.sidebar.multiselect(
            "Embedding text columns",
            options=string_columns,
            default=string_columns[:1],
        )
        if embed_cols:
            top_k = st.sidebar.slider("Top-k neighbours", min_value=5, max_value=200, value=50, step=5)
            threshold = st.sidebar.slider("Embedding similarity threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
            configs.append(
                {
                    "type": "embedding",
                    "name": "EmbeddingBlocker",
                    "params": {
                        "text_cols": embed_cols,
                        "top_k": top_k,
                        "threshold": threshold,
                        "index_backend": st.sidebar.selectbox(
                            "ANN backend",
                            options=["sklearn", "faiss", "hnsw"],
                            index=0,
                        ),
                    },
                }
            )

    return configs, False


def _build_comparator_config(
    string_columns: Sequence[str],
    numeric_columns: Sequence[str],
    date_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    st.sidebar.subheader("Comparators")
    configs: List[Dict[str, Any]] = []

    with st.sidebar.expander("String similarities", expanded=bool(string_columns)):
        selected = st.multiselect(
            "Columns",
            options=string_columns,
            default=string_columns[:2],
        )
        similarity_fn = st.selectbox("Similarity function", options=["jaro_winkler", "cosine", "jaccard"], index=0)
        for col in selected:
            weight = st.slider(f"Weight for {col}", 0.1, 5.0, 1.0, 0.1, key=f"weight_string_{col}")
            configs.append(
                {
                    "type": "string",
                    "column": col,
                    "weight": weight,
                    "params": {
                        "similarity_function": similarity_fn,
                    },
                }
            )

    if numeric_columns:
        with st.sidebar.expander("Numeric similarities", expanded=False):
            selected = st.multiselect(
                "Numeric columns",
                options=numeric_columns,
                default=[],
            )
            for col in selected:
                weight = st.slider(f"Weight for {col}", 0.1, 5.0, 0.5, 0.1, key=f"weight_numeric_{col}")
                configs.append(
                    {
                        "type": "numeric",
                        "column": col,
                        "weight": weight,
                        "params": {"method": "relative_difference"},
                    }
                )

    if date_columns:
        with st.sidebar.expander("Date similarities", expanded=False):
            selected = st.multiselect("Date columns", options=date_columns, default=[])
            tolerance = st.number_input("Max difference (days)", min_value=0, max_value=365, value=7)
            for col in selected:
                weight = st.slider(f"Weight for {col}", 0.1, 5.0, 0.5, 0.1, key=f"weight_date_{col}")
                configs.append(
                    {
                        "type": "date",
                        "column": col,
                        "weight": weight,
                        "params": {"max_days_difference": tolerance},
                    }
                )

    return configs


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def _render_metrics(result: MatchingSessionResult) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Candidate pairs", f"{result.metrics['total_candidates']:,}")
    reduction_pct = result.metrics["reduction_ratio"] * 100
    col2.metric("Reduction ratio", f"{reduction_pct:.2f}%")

def _render_batch_table(result: MatchingSessionResult) -> pd.DataFrame:
    if result.sampled_batch.empty:
        st.info("No pairs available under the current configuration.")
        return result.sampled_batch

    display_cols = [
        "id1",
        "left_preview",
        "id2",
        "right_preview",
        "matchness",
        "score",
        "embedding_similarity",
        "key_overlap_score",
        "rarity_score",
        "bin",
        "bin_reason",
        "source_blockers_display",
        "corner",
    ]
    display_df = result.sampled_batch.copy()
    for col in display_cols:
        if col not in display_df.columns:
            display_df[col] = None
    display_df = display_df[display_cols]

    labels_state = _get_label_state()
    display_df["label"] = [
        labels_state.get((row["id1"], row["id2"]), {}).get("label", "") for _, row in display_df.iterrows()
    ]
    display_df["corner"] = [
        labels_state.get((row["id1"], row["id2"]), {}).get("corner", False) for _, row in display_df.iterrows()
    ]

    st.markdown("#### Curated Batch")
    edited_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "matchness": st.column_config.NumberColumn("Matchness", format="%.3f"),
            "score": st.column_config.NumberColumn("Rule score", format="%.3f"),
            "embedding_similarity": st.column_config.NumberColumn("Embedding sim", format="%.3f"),
            "key_overlap_score": st.column_config.NumberColumn("Key overlap", format="%.2f"),
            "rarity_score": st.column_config.NumberColumn("Rarity score", format="%.3f"),
            "bin": st.column_config.Column("Bin"),
            "bin_reason": st.column_config.Column("Corner bucket"),
            "source_blockers_display": st.column_config.Column("Source blockers"),
            "label": st.column_config.SelectboxColumn(
                "Label",
                options=["match", "non_match"],
                required=True,
                width="small",
            ),
            "corner": st.column_config.CheckboxColumn("Corner case", default=False),
        },
        key="label_editor",
        width="stretch",
    )

    missing_labels = False
    for _, row in edited_df.iterrows():
        label = row.get("label")
        pair_key = (row["id1"], row["id2"])
        corner_flag = bool(row.get("corner", False))
        if not label:
            missing_labels = True
        _set_label(pair_key, label if label else None)
        _set_corner(pair_key, corner_flag)

    if missing_labels:
        st.warning("Every row must have a Match or Non-match label.")

    return display_df


def _render_swipe_view(
    result: MatchingSessionResult,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    id_column: str,
    display_columns: Sequence[str],
) -> None:
    """Render a keyboard-friendly swipe interface for sequential labeling."""

    if result.sampled_batch.empty:
        st.info("No pairs available for swipe labeling.")
        return

    sample = result.sampled_batch.reset_index(drop=True)
    total = len(sample)

    if "swipe_index" not in st.session_state or st.session_state["swipe_index"] >= total:
        st.session_state["swipe_index"] = 0
        st.session_state.pop("swipe_completed", None)

    if st.session_state.get("swipe_completed"):
        st.success("Swipe pass complete! Download your labels below.")
        labels_state = _get_label_state()
        labeled_df = _labels_state_to_dataframe(labels_state)
        if not labeled_df.empty:
            csv_buffer = StringIO()
            labeled_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download labeled pairs",
                data=csv_buffer.getvalue(),
                file_name="pydi_swipe_labels.csv",
                mime="text/csv",
            )
        else:
            st.info("No labels recorded yet.")
        st.caption("Use the navigation buttons to revisit pairs or rerun the sampler for a new batch.")
        return

    index = st.session_state.get("swipe_index", 0) % total
    pair = sample.iloc[index]
    pair_key = (pair["id1"], pair["id2"])

    def _lookup(df: pd.DataFrame, identifier: Any) -> pd.Series:
        if df.empty or id_column not in df.columns:
            return pd.Series(dtype=object)
        matches = df[df[id_column] == identifier]
        if matches.empty:
            return pd.Series(dtype=object)
        return matches.iloc[0]

    left_row = _lookup(df_left, pair["id1"])
    right_row = _lookup(df_right, pair["id2"])

    columns_to_show = [col for col in display_columns if col in left_row.index or col in right_row.index]
    extra_columns: List[str] = []
    for col in pd.Index(left_row.index).union(pd.Index(right_row.index)):
        if col == id_column or col in columns_to_show:
            continue
        extra_columns.append(col)
    ordered_columns: List[str] = []
    seen: set[str] = set()
    for col in list(columns_to_show) + extra_columns:
        if col == id_column or col in seen:
            continue
        ordered_columns.append(col)
        seen.add(col)
    if not ordered_columns:
        ordered_columns = [col for col in df_left.columns if col != id_column]
    available_columns = ordered_columns

    st.markdown("#### Swipe Labeling")
    st.caption("Use ← for Non-match, → for Match, Space for Corner case, or the buttons below to record decisions.")
    if hotkeys is None:
        st.caption("Tip: install `streamlit-hotkeys` (`pip install streamlit-hotkeys`) for more reliable keyboard shortcuts.")
    labels_state = _get_label_state()
    match_count = sum(1 for entry in labels_state.values() if entry.get("label") == "match")
    non_match_count = sum(1 for entry in labels_state.values() if entry.get("label") == "non_match")
    corner_count = sum(1 for entry in labels_state.values() if entry.get("corner"))
    st.caption("Annotation progress")
    progress_cols = st.columns(3)
    progress_cols[0].metric("Matches", match_count)
    progress_cols[1].metric("Non-matches", non_match_count)
    progress_cols[2].metric("Corner cases", corner_count)
    labeled_snapshot = _labels_state_to_dataframe(labels_state)
    if not labeled_snapshot.empty:
        snapshot_buffer = StringIO()
        labeled_snapshot.to_csv(snapshot_buffer, index=False)
        st.download_button(
            "Download current labels",
            data=snapshot_buffer.getvalue(),
            file_name="pydi_labels_partial.csv",
            mime="text/csv",
            key="download_partial_labels",
        )

    selector_key = "swipe_visible_columns_selector"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = list(available_columns)
    else:
        filtered = [col for col in st.session_state[selector_key] if col in available_columns]
        if not filtered:
            filtered = list(available_columns)
        st.session_state[selector_key] = filtered

    visible_columns = st.multiselect(
        "Columns to display",
        options=available_columns,
        default=st.session_state[selector_key],
        key=selector_key,
    )
    if not visible_columns:
        visible_columns = list(available_columns)

    matchness_val = pair.get("matchness")
    matchness_display = "n/a"
    if matchness_val is not None and not (isinstance(matchness_val, float) and math.isnan(matchness_val)):
        try:
            matchness_display = f"{float(matchness_val):.3f}"
        except (TypeError, ValueError):
            matchness_display = str(matchness_val)
    st.markdown(
        f"**Pair {index + 1} of {total}** — id1: `{pair['id1']}` • id2: `{pair['id2']}` • matchness: `{matchness_display}`"
    )

    current_annotation = labels_state.get(pair_key, {})
    current_label = current_annotation.get("label")
    current_corner = current_annotation.get("corner", False)
    corner_toggle_key = f"corner_toggle_{pair['id1']}_{pair['id2']}"
    overrides_map: Dict[str, bool] = st.session_state.setdefault("_corner_toggle_overrides", {})
    if corner_toggle_key in overrides_map:
        current_corner = overrides_map.pop(corner_toggle_key)
    st.session_state[corner_toggle_key] = current_corner
    _set_corner(pair_key, current_corner)

    table_data = {
        "field": visible_columns,
        "left": [_format_cell(left_row.get(col)) for col in visible_columns],
        "right": [_format_cell(right_row.get(col)) for col in visible_columns],
    }
    table_df = pd.DataFrame(table_data).set_index("field").T
    table_df.index = ["Left record", "Right record"]
    table_df = table_df.reset_index().rename(columns={"index": "Record"})
    st.markdown("##### Attribute comparison")
    st.dataframe(
        table_df,
        hide_index=True,
        column_config={col: st.column_config.Column(col) for col in table_df.columns},
        width="stretch",
    )

    keyboard_event = _capture_keyboard_event()
    if keyboard_event == "ArrowLeft":
        _set_label(pair_key, "non_match")
        _advance_swipe_index(1, total)
        if st.session_state["swipe_index"] == 0:
            st.session_state["swipe_completed"] = True
        st.rerun()
    elif keyboard_event == "ArrowRight":
        _set_label(pair_key, "match")
        _advance_swipe_index(1, total)
        if st.session_state["swipe_index"] == 0:
            st.session_state["swipe_completed"] = True
        st.rerun()
    elif keyboard_event in {"Space", " "}:
        current_corner = bool(st.session_state.get(corner_toggle_key, current_corner))
        new_corner = not current_corner
        _set_corner(pair_key, new_corner)
        overrides_map = st.session_state.setdefault("_corner_toggle_overrides", {})
        overrides_map[corner_toggle_key] = new_corner
        st.rerun()

    button_cols = st.columns([1, 1, 1], gap="small")
    if button_cols[0].button("⬅️ Non-match (←)", key=f"swipe_non_match_{index}", use_container_width=True):
        _set_label(pair_key, "non_match")
        _advance_swipe_index(1, total)
        if st.session_state["swipe_index"] == 0:
            st.session_state["swipe_completed"] = True
        st.rerun()
    corner_active = bool(st.session_state.get(corner_toggle_key, current_corner))
    corner_button_label = "␣ Corner case (Space)"
    if button_cols[1].button(
        corner_button_label,
        key=f"swipe_corner_{index}",
        type="primary" if corner_active else "secondary",
        use_container_width=True,
    ):
        new_corner = not corner_active
        _set_corner(pair_key, new_corner)
        overrides_map = st.session_state.setdefault("_corner_toggle_overrides", {})
        overrides_map[corner_toggle_key] = new_corner
        st.rerun()
    if button_cols[2].button("➡️ Match (→)", key=f"swipe_match_{index}", use_container_width=True):
        _set_label(pair_key, "match")
        _advance_swipe_index(1, total)
        if st.session_state["swipe_index"] == 0:
            st.session_state["swipe_completed"] = True
        st.rerun()

    if not current_label:
        st.warning("Assign a Match or Non-match label before moving on.")

    nav_cols = st.columns([1, 1, 1])
    if nav_cols[0].button("⬅️ Back", key=f"swipe_back_{index}"):
        _advance_swipe_index(-1, total)
        st.rerun()
    if nav_cols[1].button("🔁 Skip", key=f"swipe_skip_{index}"):
        _advance_swipe_index(1, total)
        st.rerun()
    if nav_cols[2].button("➡️ Next", key=f"swipe_next_{index}"):
        _advance_swipe_index(1, total)
        if st.session_state["swipe_index"] == 0:
            st.session_state["swipe_completed"] = True
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="PyDI Entity Matching Curator", layout="wide")
    st.title("PyDI Entity Matching Curator")
    st.write(
        "Union multiple blockers, score candidates with PyDI matchers, and curate balanced batches "
        "for labeling. Configure the pipeline on the left and run the sampler to "
        "surface easy positives, easy negatives, and tricky corner cases."
    )

    st.sidebar.header("Data")
    upload_left = st.sidebar.file_uploader("Left dataset", type=["csv", "tsv", "json", "parquet", "xml"])
    upload_right = st.sidebar.file_uploader("Right dataset", type=["csv", "tsv", "json", "parquet", "xml"])

    try:
        df_left = _read_uploaded_dataset(upload_left, "left")
        df_right = _read_uploaded_dataset(upload_right, "right")
    except ValueError as exc:
        st.error(str(exc))
        return

    if df_left.empty or df_right.empty:
        st.info("Upload both datasets to continue.")
        return

    st.sidebar.subheader("Identifiers")
    left_has_pydi = "pydi_id" in df_left.columns
    right_has_pydi = "pydi_id" in df_right.columns

    if st.sidebar.checkbox(
        "Generate PyDI IDs for left dataset", value=False, key="pydi_id_left", disabled=left_has_pydi
    ):
        df_left = _prepare_dataset(df_left, "left")
        left_has_pydi = True
    if st.sidebar.checkbox(
        "Generate PyDI IDs for right dataset", value=False, key="pydi_id_right", disabled=right_has_pydi
    ):
        df_right = _prepare_dataset(df_right, "right")
        right_has_pydi = True

    default_id_candidates = ["pydi_id", "id", "entity_id"]
    shared_columns = [col for col in df_left.columns if col in df_right.columns]
    if not shared_columns:
        st.error("No shared columns found between datasets. Generate PyDI IDs or align column names to continue.")
        return

    default_id = next((col for col in default_id_candidates if col in shared_columns), shared_columns[0])

    id_column = st.sidebar.selectbox(
        "Record identifier column",
        options=shared_columns,
        index=shared_columns.index(default_id) if default_id in shared_columns else 0,
    )

    candidate_columns = [col for col in shared_columns if col != id_column]
    string_columns: List[str] = []
    numeric_columns: List[str] = []
    date_columns: List[str] = []

    for col in candidate_columns:
        left_series = df_left[col]
        right_series = df_right[col]

        if pd.api.types.is_datetime64_any_dtype(left_series) or pd.api.types.is_datetime64_any_dtype(right_series):
            date_columns.append(col)
            continue

        if pd.api.types.is_numeric_dtype(left_series) and pd.api.types.is_numeric_dtype(right_series):
            numeric_columns.append(col)
            continue

        if (
            pd.api.types.is_string_dtype(left_series)
            or pd.api.types.is_string_dtype(right_series)
            or pd.api.types.is_object_dtype(left_series)
            or pd.api.types.is_object_dtype(right_series)
        ):
            sample = pd.concat([left_series, right_series], ignore_index=True).dropna().astype(str)
            if not sample.empty:
                string_columns.append(col)
            continue

    display_columns = candidate_columns if candidate_columns else [col for col in df_left.columns if col != id_column]

    st.session_state["swipe_left_df"] = df_left
    st.session_state["swipe_right_df"] = df_right
    st.session_state["swipe_id_column"] = id_column
    st.session_state["swipe_display_columns"] = display_columns

    blocking_config, blocking_disabled = _build_blocking_config(id_column, shared_columns, string_columns)
    comparator_config = _build_comparator_config(string_columns, numeric_columns, date_columns)

    batch_size = st.sidebar.number_input("Batch size", min_value=10, max_value=200, value=50, step=5)
    corner_share = 0.30
    easy_share = (1.0 - corner_share) / 2
    tau_low = 0.25
    tau_high = 0.75

    st.subheader("Dataset snapshots")
    col_left, col_right = st.columns(2)
    col_left.dataframe(df_left.head(10), width="stretch")
    col_left.caption(f"Left dataset — {len(df_left)} rows")
    col_right.dataframe(df_right.head(10), width="stretch")
    col_right.caption(f"Right dataset — {len(df_right)} rows")

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download left dataset (with pydi_id)",
        data=_dataframe_to_csv_bytes(df_left),
        file_name="pydi_left_dataset.csv",
        mime="text/csv",
    )
    download_cols[1].download_button(
        "Download right dataset (with pydi_id)",
        data=_dataframe_to_csv_bytes(df_right),
        file_name="pydi_right_dataset.csv",
        mime="text/csv",
    )

    if not blocking_disabled and not blocking_config:
        st.warning("Select at least one blocking strategy to proceed.")
        return
    if not comparator_config:
        st.warning("Select at least one comparator to score candidate pairs.")
        return

    if st.sidebar.button("Run sampler", type="primary"):
        with st.spinner("Running PyDI pipeline..."):
            try:
                result = run_interactive_matching(
                    df_left=df_left,
                    df_right=df_right,
                    id_column=id_column,
                    blocking_options=blocking_config,
                    comparator_options=comparator_config,
                    tau_low=tau_low,
                    tau_high=tau_high,
                    matcher_threshold=0.0,
                    epsilon_random=0.0,
                    batch_size=int(batch_size),
                    mix=(easy_share, easy_share, corner_share),
                    random_seed=42,
                    preview_columns=string_columns[:3],
                )
                st.session_state["pydi_matching_result"] = result
                st.session_state["swipe_index"] = 0
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Pipeline failed: {exc}")
                return

    result: Optional[MatchingSessionResult] = st.session_state.get("pydi_matching_result")
    if result is None:
        st.info("Configure the pipeline and click **Run sampler** to generate a batch.")
        return

    _render_metrics(result)
    _render_batch_table(result)
    _render_swipe_view(
        result,
        st.session_state.get("swipe_left_df", pd.DataFrame()),
        st.session_state.get("swipe_right_df", pd.DataFrame()),
        id_column=st.session_state.get("swipe_id_column", id_column),
        display_columns=st.session_state.get("swipe_display_columns", []),
    )


if __name__ == "__main__":
    main()
