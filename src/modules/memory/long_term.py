"""
Long-term memory: Vector database for semantic memory storage.
Uses ChromaDB for embedding and retrieving important memories.
"""
import logging
from typing import List, Dict, Any, Optional
import uuid
import json
import os

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Manages long-term semantic memory using ChromaDB."""
    
    def __init__(self, persist_directory: str, collection_name: str = "memories"):
        """Initialize with ChromaDB persistence directory."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Long-term memory initialized: {self.persist_directory}")
            logger.info(f"Collection '{self.collection_name}' has {self.collection.count()} memories")
            
        except ImportError:
            logger.error("chromadb not installed. Install: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_memory(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_id: Optional[str] = None
    ) -> str:
        """Add memory to vector database."""
        try:
            if memory_id is None:
                memory_id = f"mem_{uuid.uuid4()}"
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            logger.info(f"Added memory {memory_id} to long-term storage")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    def add_memories_batch(
        self,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        memory_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add multiple memories in batch."""
        try:
            if memory_ids is None:
                memory_ids = [f"mem_{uuid.uuid4()}" for _ in contents]
            
            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=memory_ids
            )
            
            logger.info(f"Added {len(contents)} memories to long-term storage")
            return memory_ids
            
        except Exception as e:
            logger.error(f"Failed to add memories batch: {e}")
            raise
    
    def search_memories(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for relevant memories using semantic similarity."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            memories = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    memories.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            logger.debug(f"Found {len(memories)} relevant memories for query")
            return {'memories': memories, 'count': len(memories)}
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return {'memories': [], 'count': 0}
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific memory by ID."""
        try:
            result = self.collection.get(ids=[memory_id])
            
            if result and result['documents']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    def get_recent_memories(
        self,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get recent memories ordered by timestamp."""
        try:
            result = self.collection.get(
                where=where,
                limit=n_results
            )
            
            memories = []
            if result and result['documents']:
                for i in range(len(result['documents'])):
                    memories.append({
                        'id': result['ids'][i],
                        'content': result['documents'][i],
                        'metadata': result['metadatas'][i]
                    })
            
            memories.sort(
                key=lambda x: x['metadata'].get('timestamp', 0),
                reverse=True
            )
            
            return memories[:n_results]
            
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str):
        """Delete specific memory."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory {memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise
    
    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get memory count: {e}")
            return 0
    
    def clear_all_memories(self):
        """Clear all memories (use with caution)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("All long-term memories cleared!")
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            raise