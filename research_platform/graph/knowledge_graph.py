import sqlite3
import json
import os
from typing import List, Dict, Any, Tuple

class ResearchKnowledgeGraph:
    """
    Edison v4.0 Persistent Knowledge Graph.
    Uses SQLite to manage entities and relationships in a semantic web structure.
    """

    def __init__(self, db_path: str = "research_platform/data/knowledge_graph.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Entities: Research topics, papers, code, hypotheses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Relationships: "is-a", "references", "optimized-by", "contradicts"
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    source_id TEXT,
                    target_id TEXT,
                    relation TEXT,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (source_id, target_id, relation),
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )
            """)
            conn.commit()

    def add_entity(self, entity_id: str, name: str, entity_type: str, metadata: Dict[str, Any] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO entities (id, name, type, metadata) VALUES (?, ?, ?, ?)",
                (entity_id, name, entity_type, json.dumps(metadata or {}))
            )
            conn.commit()

    def add_relation(self, source_id: str, target_id: str, relation: str, weight: float = 1.0):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO relationships (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
                (source_id, target_id, relation, weight)
            )
            conn.commit()

    def query_subgraph(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """Returns a subgraph centered around the entity_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # This is a simplified 1-depth query
            cursor.execute("""
                SELECT e.id, e.name, e.type, r.relation, r.target_id
                FROM entities e
                JOIN relationships r ON e.id = r.source_id
                WHERE e.id = ?
            """, (entity_id,))

            results = cursor.fetchall()
            return {
                "entity_id": entity_id,
                "connections": [
                    {"relation": r[3], "target_id": r[4]} for r in results
                ]
            }

    def get_all_relations(self) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_id, target_id, relation FROM relationships")
            return cursor.fetchall()

    def get_all_entities(self, entity_type: str = None) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if entity_type:
                cursor.execute("SELECT id, name, type FROM entities WHERE type = ?", (entity_type,))
            else:
                cursor.execute("SELECT id, name, type FROM entities")
            return cursor.fetchall()
