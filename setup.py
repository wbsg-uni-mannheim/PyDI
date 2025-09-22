"""Setuptools entry point for building and distributing PyDI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib
from setuptools import find_packages, setup


ROOT = Path(__file__).parent
PROJECT_METADATA: dict[str, Any] = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]


def _collect_author_fields(authors: list[dict[str, str]]) -> tuple[str | None, str | None]:
    names = [author["name"] for author in authors if author.get("name")]
    emails = [author["email"] for author in authors if author.get("email")]
    return (
        ", ".join(names) if names else None,
        ", ".join(emails) if emails else None,
    )


def _collect_license_fields(license_block: str | dict[str, str] | None) -> tuple[str | None, list[str] | None]:
    if not license_block:
        return None, None
    if isinstance(license_block, str):
        return license_block, None
    if "text" in license_block:
        return license_block["text"], None
    if "file" in license_block:
        return None, [license_block["file"]]
    return None, None


def _collect_urls(urls: dict[str, str] | None) -> tuple[str | None, dict[str, str] | None]:
    if not urls:
        return None, None
    homepage = urls.get("Homepage")
    remaining = {key: value for key, value in urls.items() if key != "Homepage"}
    return homepage, remaining or None


def _load_long_description(readme_path: str | dict[str, str]) -> str:
    path_str = readme_path if isinstance(readme_path, str) else readme_path.get("file", "README.md")
    return (ROOT / path_str).read_text("utf-8")


authors = PROJECT_METADATA.get("authors", [])
author, author_email = _collect_author_fields(authors)

license_block = PROJECT_METADATA.get("license")
license_name, license_files = _collect_license_fields(license_block)

homepage, project_urls = _collect_urls(PROJECT_METADATA.get("urls"))

setup(
    name=PROJECT_METADATA["name"],
    version=PROJECT_METADATA["version"],
    description=PROJECT_METADATA.get("description", ""),
    long_description=_load_long_description(PROJECT_METADATA.get("readme", "README.md")),
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    python_requires=PROJECT_METADATA.get("requires-python"),
    classifiers=PROJECT_METADATA.get("classifiers", []),
    install_requires=PROJECT_METADATA.get("dependencies", []),
    extras_require=PROJECT_METADATA.get("optional-dependencies", {}),
    packages=find_packages(where=".", include=["PyDI*"]),
    include_package_data=True,
    license=license_name,
    license_files=license_files,
    url=homepage,
    project_urls=project_urls,
)
