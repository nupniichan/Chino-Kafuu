"""
View token usage statistics from logs.
Run this script to see token usage summary.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.memory.token_logger import TokenLogger


def main():
    """Display token usage statistics."""
    logger = TokenLogger()
    
    print("\n" + "="*70)
    print("📊 TOKEN USAGE STATISTICS")
    print("="*70)
    print(f"Log file: {logger.log_file}")
    print("-"*70)
    
    # Get summary stats
    stats = logger.get_stats_summary()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"Total Events: {stats.get('events', 0)}")
    print(f"Total Tokens (cumulative): {stats.get('total_tokens', 0):,}")
    print(f"Total Messages: {stats.get('total_messages', 0)}")
    print(f"Max Tokens (peak): {stats.get('max_tokens', 0):,}")
    print(f"Avg Tokens per Event: {stats.get('avg_tokens_per_event', 0):.2f}")
    print("="*70 + "\n")
    
    # Show recent entries
    if os.path.exists(logger.log_file):
        print("Recent entries (last 5):")
        print("-"*70)
        
        import csv
        with open(logger.log_file, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            recent = reader[-5:] if len(reader) > 5 else reader
            
            for row in recent:
                dt = row.get('datetime', '')
                tokens = row.get('short_term_tokens', '0')
                messages = row.get('short_term_messages', '0')
                event = row.get('event_type', '')
                print(f"[{dt}] {messages} msgs, {tokens} tokens - {event}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
