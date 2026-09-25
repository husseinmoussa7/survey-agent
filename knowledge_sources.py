import os
import logging
from typing import Optional

# We try to import CrewAI Knowledge APIs in a safe, version-agnostic way.
try:
    from crewai import Knowledge  # type: ignore
    # Newer CrewAI exposes directory sources under this namespace
    from crewai.knowledge.source import LocalDirectory  # type: ignore
    _CREWAI_KNOWLEDGE_AVAILABLE = True
except Exception:  # pragma: no cover - best-effort compatibility
    Knowledge = None  # type: ignore
    LocalDirectory = None  # type: ignore
    _CREWAI_KNOWLEDGE_AVAILABLE = False


logger = logging.getLogger(__name__)


def _pdf_source(path: str) -> Optional[object]:
    """Create a LocalDirectory knowledge source for PDFs under a path.

    Returns None if CrewAI knowledge sources are unavailable or the path doesn't exist.
    """
    if not _CREWAI_KNOWLEDGE_AVAILABLE:
        logger.warning("CrewAI Knowledge APIs not available; skipping knowledge for '%s'", path)
        return None
    if not os.path.isdir(path):
        logger.warning("Knowledge path does not exist or is not a directory: %s", path)
        return None
    try:
        # Recursive glob to include nested directories; PDFs only
        return LocalDirectory(path=path, glob="**/*.pdf")
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to create LocalDirectory knowledge source for %s: %s", path, e)
        return None


def build_unified_pdf_knowledge() -> Optional[object]:
    """Build a single Knowledge object aggregating PDFs from the specified folders.

    - knowledge/Sample Papers: PDFs only (e.g., stimuli)
    - knowledge/JCR Papers: PDFs only
    - knowledge/New Papers: PDFs only

    Returns a Knowledge object if supported, else None.
    """
    if not _CREWAI_KNOWLEDGE_AVAILABLE:
        return None

    sample_papers = _pdf_source(os.path.join("knowledge", "Sample Papers"))
    jcr_papers = _pdf_source(os.path.join("knowledge", "JCR Papers"))
    new_papers = _pdf_source(os.path.join("knowledge", "New Papers"))

    sources = [s for s in [sample_papers, jcr_papers, new_papers] if s is not None]
    if not sources:
        logger.warning("No knowledge sources discovered; proceeding without knowledge.")
        return None

    try:
        return Knowledge(sources=sources)
    except TypeError:
        # Some older CrewAI versions expect a list rather than a kwarg
        try:
            return Knowledge(sources)  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to construct Knowledge object: %s", e)
            return None
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to construct Knowledge object: %s", e)
        return None


def attach_knowledge_to_crew(crew, knowledge_obj) -> None:
    """Attach knowledge to a Crew instance in a version-tolerant way."""
    if knowledge_obj is None:
        return
    # Prefer attribute if available
    try:
        if hasattr(crew, "knowledge"):
            setattr(crew, "knowledge", knowledge_obj)
            return
    except Exception:
        pass
    # Try method-based API
    for method_name in ("add_knowledge", "load_knowledge", "with_knowledge"):
        try:
            if hasattr(crew, method_name):
                getattr(crew, method_name)(knowledge_obj)
                return
        except Exception:
            continue
    # If all else fails, just log and continue without blocking
    logger.warning("Could not attach knowledge to Crew instance using available hooks.")


# Build once and expose shared knowledge handles for reuse across crews
try:
    _UNIFIED_KNOWLEDGE = build_unified_pdf_knowledge()
except Exception:
    _UNIFIED_KNOWLEDGE = None

# For clarity in imports from survey.py
initial_crew_knowledge = _UNIFIED_KNOWLEDGE
enhancement_crew_knowledge = _UNIFIED_KNOWLEDGE
paper_crew_knowledge = _UNIFIED_KNOWLEDGE

