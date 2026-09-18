from backend.database.db import (
    init_db,
    save_document,
    get_all_documents,
    get_document_by_id,
    delete_document_by_id,
    save_verification,
    get_verification_history,
    get_verification_by_id,
    find_most_similar_claim,
    get_analytics
)

__all__ = [
    "init_db",
    "save_document",
    "get_all_documents",
    "get_document_by_id",
    "delete_document_by_id",
    "save_verification",
    "get_verification_history",
    "get_verification_by_id",
    "find_most_similar_claim",
    "get_analytics"
]
