import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.memory.cache import MemoryCache
from modules.memory.short_term import ShortTermMemory
from setting import SHORT_TERM_MEMORY_SIZE


class TestMemoryCache:
    def test_add_message(self):
        cache = MemoryCache()
        cache.add_message("test_key", "message1")
        cache.add_message("test_key", "message2")
        
        messages = cache.get_messages("test_key")
        assert len(messages) == 2
        assert messages[0] == "message1"
        assert messages[1] == "message2"
    
    def test_get_messages_all(self):
        cache = MemoryCache()
        cache.add_message("test_key", "msg1")
        cache.add_message("test_key", "msg2")
        cache.add_message("test_key", "msg3")
        
        messages = cache.get_messages("test_key", 0, -1)
        assert len(messages) == 3
    
    def test_get_messages_range(self):
        cache = MemoryCache()
        for i in range(5):
            cache.add_message("test_key", f"msg{i}")
        
        messages = cache.get_messages("test_key", 1, 3)
        assert len(messages) == 3
        assert messages[0] == "msg1"
        assert messages[2] == "msg3"
    
    def test_get_messages_negative_index(self):
        cache = MemoryCache()
        for i in range(5):
            cache.add_message("test_key", f"msg{i}")
        
        messages = cache.get_messages("test_key", -3, -1)
        assert len(messages) == 3
        assert messages[0] == "msg2"
        assert messages[2] == "msg4"
    
    def test_trim(self):
        cache = MemoryCache()
        for i in range(5):
            cache.add_message("test_key", f"msg{i}")
        
        cache.trim("test_key", 1, 3)
        messages = cache.get_messages("test_key")
        assert len(messages) == 3
        assert messages[0] == "msg1"
        assert messages[2] == "msg3"
    
    def test_trim_negative_index(self):
        cache = MemoryCache()
        for i in range(5):
            cache.add_message("test_key", f"msg{i}")
        
        cache.trim("test_key", -3, -1)
        messages = cache.get_messages("test_key")
        assert len(messages) == 3
        assert messages[0] == "msg2"
    
    def test_delete(self):
        cache = MemoryCache()
        cache.add_message("test_key", "message")
        assert cache.exists("test_key") is True
        
        cache.delete("test_key")
        assert cache.exists("test_key") is False
        assert len(cache.get_messages("test_key")) == 0
    
    def test_exists(self):
        cache = MemoryCache()
        assert cache.exists("nonexistent") is False
        
        cache.add_message("test_key", "message")
        assert cache.exists("test_key") is True
    
    def test_multiple_keys(self):
        cache = MemoryCache()
        cache.add_message("key1", "msg1")
        cache.add_message("key2", "msg2")
        
        assert len(cache.get_messages("key1")) == 1
        assert len(cache.get_messages("key2")) == 1
        assert cache.get_messages("key1")[0] == "msg1"
        assert cache.get_messages("key2")[0] == "msg2"
    
    def test_clear_all(self):
        cache = MemoryCache()
        cache.add_message("key1", "msg1")
        cache.add_message("key2", "msg2")
        
        cache.clear_all()
        assert cache.exists("key1") is False
        assert cache.exists("key2") is False


class TestShortTermMemoryWithCache:
    def test_init_with_cache(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        assert memory.storage is not None
        assert isinstance(memory.storage, MemoryCache)
    
    def test_add_user_message(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        memory.start_new_session()
        
        entry = memory.add_user_message(
            message="Hello",
            emotion="happy",
            lang="en"
        )
        
        assert "user" in entry
        assert entry["user"]["message"] == "Hello"
        assert entry["user"]["emotion"] == "happy"
        
        messages = memory.get_recent_messages()
        assert len(messages) == 1
    
    def test_add_chino_response(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        memory.start_new_session()
        
        entry = memory.add_chino_response(
            text_spoken="こんにちは",
            text_display="Xin chào",
            lang="jp",
            emotion="normal"
        )
        
        assert "chino-kafuu" in entry
        assert entry["chino-kafuu"]["message"]["text_spoken"] == "こんにちは"
        assert entry["chino-kafuu"]["message"]["text_display"] == "Xin chào"
        
        messages = memory.get_recent_messages()
        assert len(messages) == 1
    
    def test_get_recent_messages(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        memory.start_new_session()
        
        memory.add_user_message("Message 1")
        memory.add_user_message("Message 2")
        memory.add_user_message("Message 3")
        
        messages = memory.get_recent_messages()
        assert len(messages) == 3
        
        recent = memory.get_recent_messages(count=2)
        assert len(recent) == 2
    
    def test_get_conversation_context(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        memory.start_new_session()
        
        memory.add_user_message("Hello", emotion="happy")
        memory.add_chino_response(
            text_spoken="こんにちは",
            text_display="Xin chào",
            emotion="normal"
        )
        
        context = memory.get_conversation_context()
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert context[1]["role"] == "Chino"
        assert context[1]["content"] == "Xin chào"
    
    def test_clear(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        memory.start_new_session()
        
        memory.add_user_message("Message 1")
        memory.add_user_message("Message 2")
        
        assert len(memory.get_recent_messages()) == 2
        
        memory.clear()
        assert len(memory.get_recent_messages()) == 0
        assert memory.current_session_id is None
    
    def test_start_new_session(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        
        session_id1 = memory.start_new_session()
        assert session_id1 is not None
        assert memory.current_session_id == session_id1
        
        memory.add_user_message("Message 1")
        
        session_id2 = memory.start_new_session()
        assert session_id2 != session_id1
        assert len(memory.get_recent_messages()) == 0
    
    def test_max_size_limit(self):
        max_size = 3
        memory = ShortTermMemory(storage_type="cache", max_size=max_size)
        memory.start_new_session()
        
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        
        messages = memory.get_recent_messages()
        assert len(messages) <= max_size
    
    def test_buffer_property(self):
        memory = ShortTermMemory(storage_type="cache", max_size=10)
        memory.start_new_session()
        
        memory.add_user_message("Message 1")
        memory.add_user_message("Message 2")
        
        buffer = memory.buffer
        assert isinstance(buffer, list)
        assert len(buffer) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
