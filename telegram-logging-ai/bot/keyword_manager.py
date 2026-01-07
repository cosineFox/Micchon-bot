import asyncio
import random
import re
from datetime import datetime, timedelta
from typing import List, Set, Dict
import logging

logger = logging.getLogger(__name__)


class KeywordManager:
    """Manages intelligent keyword selection for waifu random replies"""

    def __init__(self, initial_keywords: List[str] = None):
        """
        Initialize keyword manager

        Args:
            initial_keywords: Initial list of keywords to seed the system
        """
        self.keywords: Set[str] = set(initial_keywords or self._get_default_keywords())
        self.keyword_weights: Dict[str, float] = {}  # Weight for each keyword (0.0-1.0)
        self.keyword_usage: Dict[str, datetime] = {}  # Last time keyword was used
        self.keyword_frequency: Dict[str, int] = {}  # How often keyword has been used
        self._lock = asyncio.Lock()

    def _get_default_keywords(self) -> List[str]:
        """Get default keywords for initial seeding"""
        return [
            # Common greetings and expressions
            "hello", "hi", "hey", "good morning", "good evening", "good night",
            "how are you", "how's it going", "what's up", "what's new",
            
            # Emotional expressions
            "love", "miss", "think of", "care about", "wonder",
            
            # Daily activities
            "working", "coding", "programming", "building", "making", "creating",
            "project", "idea", "plan", "goal", "task", "work",
            
            # Food and drink
            "eating", "food", "lunch", "dinner", "breakfast", "snack", "coffee", "tea",
            
            # Relaxation
            "tired", "rest", "sleep", "break", "relax", "chill", "weekend", "vacation",
            
            # Hobbies
            "reading", "book", "movie", "film", "show", "game", "gaming", "music",
            
            # Tech terms
            "ai", "bot", "code", "debug", "bug", "fix", "update", "deploy",
            
            # Personal terms
            "remember", "memory", "past", "future", "dream", "hope", "wish",
            
            # Questions
            "why", "how", "what", "when", "where", "who", "which", "can", "could",
            
            # Expressions of gratitude
            "thank", "thanks", "appreciate", "grateful", "appreciation",
            
            # Concerns
            "worry", "concern", "stressed", "stress", "anxious", "nervous",
            
            # Achievements
            "done", "finished", "completed", "achieved", "success", "accomplished",
            
            # Time-related
            "today", "tomorrow", "yesterday", "now", "later", "soon", "already",
            
            # Feelings
            "happy", "sad", "excited", "bored", "exciting", "fun", "funny",
            
            # Interests
            "learning", "study", "knowledge", "understand", "discover", "explore"
        ]

    async def add_keyword(self, keyword: str) -> bool:
        """Add a new keyword to the system"""
        async with self._lock:
            keyword = keyword.lower().strip()
            if keyword and keyword not in self.keywords:
                self.keywords.add(keyword)
                self.keyword_weights[keyword] = 0.5  # Default weight
                self.keyword_frequency[keyword] = 0
                logger.debug(f"Added new keyword: {keyword}")
                return True
            return False

    async def remove_keyword(self, keyword: str) -> bool:
        """Remove a keyword from the system"""
        async with self._lock:
            keyword = keyword.lower().strip()
            if keyword in self.keywords:
                self.keywords.remove(keyword)
                self.keyword_weights.pop(keyword, None)
                self.keyword_usage.pop(keyword, None)
                self.keyword_frequency.pop(keyword, None)
                logger.debug(f"Removed keyword: {keyword}")
                return True
            return False

    async def update_keyword_weight(self, keyword: str, weight: float):
        """Update the weight of a keyword (0.0-1.0)"""
        async with self._lock:
            keyword = keyword.lower().strip()
            if keyword in self.keywords:
                self.keyword_weights[keyword] = max(0.0, min(1.0, weight))
                logger.debug(f"Updated weight for keyword '{keyword}': {weight}")

    async def increment_keyword_usage(self, keyword: str):
        """Increment the usage count for a keyword"""
        async with self._lock:
            keyword = keyword.lower().strip()
            if keyword in self.keywords:
                self.keyword_frequency[keyword] = self.keyword_frequency.get(keyword, 0) + 1
                self.keyword_usage[keyword] = datetime.now()

    async def get_relevant_keywords(self, text: str) -> List[str]:
        """Get keywords that match the given text"""
        async with self._lock:
            text_lower = text.lower()
            matches = []
            
            for keyword in self.keywords:
                # Check if keyword is in the text (with word boundaries for better matching)
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    matches.append(keyword)
            
            return matches

    async def should_respond_randomly(self, text: str) -> bool:
        """
        Determine if the waifu should respond randomly based on keywords in the text

        Returns:
            True if should respond randomly, False otherwise
        """
        # Get keywords without lock (get_relevant_keywords has its own lock)
        relevant_keywords = await self.get_relevant_keywords(text)

        if not relevant_keywords:
            return False

        async with self._lock:
            # Calculate probability based on keyword weights
            total_weight = sum(self.keyword_weights.get(kw, 0.5) for kw in relevant_keywords)
            avg_weight = total_weight / len(relevant_keywords) if relevant_keywords else 0.0

            # Use the average weight as probability
            should_respond = random.random() < avg_weight

            if should_respond:
                # Update usage for all matching keywords (without separate lock)
                for kw in relevant_keywords:
                    kw_lower = kw.lower().strip()
                    if kw_lower in self.keywords:
                        self.keyword_frequency[kw_lower] = self.keyword_frequency.get(kw_lower, 0) + 1
                        self.keyword_usage[kw_lower] = datetime.now()

            logger.debug(f"Keyword check: '{text[:30]}...' -> {relevant_keywords}, "
                        f"weight: {avg_weight:.2f}, respond: {should_respond}")

            return should_respond

    async def update_keywords_from_memory(self, recent_memories: List[object]) -> List[str]:
        """
        Update keywords based on recent memories to keep them relevant

        Args:
            recent_memories: List of recent memory objects

        Returns:
            List of newly added keywords
        """
        async with self._lock:
            new_keywords = []

            # Extract potential keywords from recent memories
            for memory in recent_memories:
                content = getattr(memory, 'content', '')
                if content:
                    # Extract potential keywords from content
                    potential_keywords = await self._extract_keywords_from_text(content)

                    for kw in potential_keywords:
                        if await self.add_keyword(kw):
                            new_keywords.append(kw)

            logger.info(f"Added {len(new_keywords)} new keywords from recent memories")
            return new_keywords

    async def learn_from_historical_chats(self, master_repo) -> List[str]:
        """
        Learn new keywords autonomously from historical chat records in the database

        Args:
            master_repo: Master repository to access historical chat data

        Returns:
            List of newly learned keywords
        """
        async with self._lock:
            new_keywords = []

            try:
                # Get a larger sample of historical memories to learn from
                historical_memories = await master_repo.get_recent(hours=168, limit=500)  # Last week, up to 500 entries

                # Analyze patterns in the historical data
                keyword_frequency = {}
                conversation_patterns = []

                for memory in historical_memories:
                    content = getattr(memory, 'content', '')
                    if content:
                        # Extract potential keywords and track their frequency
                        potential_keywords = await self._extract_keywords_from_text(content)

                        for kw in potential_keywords:
                            keyword_frequency[kw] = keyword_frequency.get(kw, 0) + 1

                        # Identify conversation patterns (e.g., common phrases that lead to responses)
                        conversation_patterns.append(content)

                # Identify high-frequency terms that appear often in conversations
                high_freq_keywords = [kw for kw, freq in keyword_frequency.items() if freq >= 3]

                # Add high-frequency keywords to our system
                for kw in high_freq_keywords:
                    if await self.add_keyword(kw):
                        new_keywords.append(kw)

                # Identify trigger phrases that often precede important conversations
                trigger_phrases = await self._identify_trigger_phrases(conversation_patterns)
                for phrase in trigger_phrases:
                    if await self.add_keyword(phrase):
                        new_keywords.append(phrase)

                logger.info(f"Learned {len(new_keywords)} new keywords from historical chats")
                return new_keywords

            except Exception as e:
                logger.error(f"Error learning from historical chats: {e}")
                return []

    async def _identify_trigger_phrases(self, conversation_patterns: List[str]) -> List[str]:
        """
        Identify common phrases that often trigger meaningful conversations

        Args:
            conversation_patterns: List of conversation snippets

        Returns:
            List of potential trigger phrases
        """
        trigger_indicators = []

        # Look for common phrases that often precede important conversations
        common_triggers = [
            # Expressions of feeling
            "i feel", "i think", "i believe", "i want", "i need", "i hope", "i wish",
            # Expressions of activity
            "working on", "building", "making", "creating", "developing", "working with",
            # Expressions of status
            "tired", "stressed", "excited", "happy", "sad", "frustrated", "proud",
            # Expressions of time
            "today", "tomorrow", "yesterday", "later", "now", "soon", "already",
            # Expressions of achievement
            "finished", "completed", "done", "achieved", "succeeded", "failed",
            # Expressions of interest
            "interesting", "cool", "amazing", "wonderful", "great", "awesome",
            # Expressions of concern
            "worry", "concern", "problem", "issue", "trouble", "difficulty",
            # Expressions of gratitude
            "thank you", "thanks", "appreciate", "grateful", "appreciation"
        ]

        # Find phrases in conversation patterns that match trigger indicators
        found_triggers = []
        for pattern in conversation_patterns:
            pattern_lower = pattern.lower()
            for trigger in common_triggers:
                if trigger in pattern_lower and trigger not in found_triggers:
                    found_triggers.append(trigger)

        return found_triggers

    async def should_initiate_conversation(self, last_user_message_time: datetime,
                                        last_bot_message_time: datetime,
                                        user_id: int) -> bool:
        """
        Determine if the bot should initiate a conversation after a period of inactivity

        Args:
            last_user_message_time: When the user last sent a message
            last_bot_message_time: When the bot last sent a message
            user_id: The user ID to check

        Returns:
            True if the bot should initiate conversation, False otherwise
        """
        now = datetime.now()

        # Calculate time since last user message
        if last_user_message_time:
            time_since_user_msg = now - last_user_message_time
        else:
            time_since_user_msg = timedelta(days=1)  # If no record, assume long time

        # Calculate time since last bot message
        if last_bot_message_time:
            time_since_bot_msg = now - last_bot_message_time
        else:
            time_since_bot_msg = timedelta(days=1)  # If no record, assume long time

        # If user hasn't messaged in 2+ hours and bot hasn't messaged in 1+ hour, consider initiating
        if (time_since_user_msg > timedelta(hours=2) and
            time_since_bot_msg > timedelta(hours=1)):
            # Use a probability factor to make it feel natural (not too predictable)
            import random
            return random.random() < 0.3  # 30% chance to initiate after 2 hours of user inactivity

        return False

    async def generate_initiative_response(self, user_id: int, master_repo) -> str:
        """
        Generate a simple initiative response based on user's recent activity

        Args:
            user_id: The user ID
            master_repo: Master repository to access user's recent memories

        Returns:
            A simple initiative response
        """
        # Get recent memories for this user to personalize the response
        recent_memories = await master_repo.get_recent(hours=24, limit=10)

        # Filter for memories from this specific user
        user_memories = [mem for mem in recent_memories
                        if mem.metadata.get('user_id') == user_id]

        if user_memories:
            # Look for recent topics to reference
            recent_topics = []
            for memory in user_memories[:3]:  # Look at most recent 3 memories
                content = memory.content.lower()
                # Extract potential topics from recent messages
                potential_topics = await self._extract_keywords_from_text(content)
                recent_topics.extend(potential_topics[:2])  # Take up to 2 topics per message

            if recent_topics:
                # Create a response that references recent topics
                import random
                topic = random.choice(list(set(recent_topics)))  # Get a unique topic
                initiative_responses = [
                    f"Hi there! I've been thinking about {topic} since our last chat. How are you doing?",
                    f"Hello! I noticed we were talking about {topic} earlier. I hope you're having a good day!",
                    f"Hey! I was just reflecting on our conversation about {topic}. How's everything going?",
                    f"Hi! I hope you're doing well. I was reminded of {topic} from our previous chat.",
                    f"Hello there! I've missed our chats about {topic} and other things. How have you been?"
                ]
                return random.choice(initiative_responses)

        # Fallback if no recent topics found
        fallback_responses = [
            "Hi there! I hope you're having a good day!",
            "Hello! I was wondering how you're doing today.",
            "Hey! I hope everything is going well for you.",
            "Hi! Just wanted to say hello and see how you're doing.",
            "Hello there! I hope you're taking good care of yourself."
        ]

        import random
        return random.choice(fallback_responses)

    async def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract potential keywords from text using simple heuristics"""
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Filter for meaningful words (not stop words, not too short)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        potential_keywords = []
        for word in words:
            if len(word) > 3 and word not in stop_words and word.isalpha():
                # Check if it's not already in our keywords
                if word not in self.keywords:
                    potential_keywords.append(word)
        
        # Return top 5 most frequent words as potential keywords
        word_freq = {}
        for word in potential_keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top 5
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]

    async def get_keyword_statistics(self) -> Dict:
        """Get statistics about keywords"""
        async with self._lock:
            return {
                'total_keywords': len(self.keywords),
                'keywords': list(self.keywords),
                'weights': dict(self.keyword_weights),
                'frequencies': dict(self.keyword_frequency),
                'last_used': {kw: str(dt) for kw, dt in self.keyword_usage.items()}
            }

    async def adjust_weights_based_on_usage(self):
        """Adjust keyword weights based on usage patterns"""
        async with self._lock:
            now = datetime.now()
            
            for keyword in self.keywords:
                last_used = self.keyword_usage.get(keyword)
                frequency = self.keyword_frequency.get(keyword, 0)
                
                if last_used:
                    # Calculate days since last use
                    days_since_use = (now - last_used).days
                    
                    # If used recently, slightly increase weight
                    if days_since_use == 0:  # Used today
                        current_weight = self.keyword_weights.get(keyword, 0.5)
                        self.keyword_weights[keyword] = min(1.0, current_weight + 0.05)
                    elif days_since_use > 7:  # Not used in a week, decrease weight
                        current_weight = self.keyword_weights.get(keyword, 0.5)
                        self.keyword_weights[keyword] = max(0.1, current_weight - 0.05)
                else:
                    # Never used, keep default weight
                    if keyword not in self.keyword_weights:
                        self.keyword_weights[keyword] = 0.5


# Global singleton instance
_keyword_manager: KeywordManager = None


def get_keyword_manager() -> KeywordManager:
    """Get or create the global keyword manager instance"""
    global _keyword_manager
    
    if _keyword_manager is None:
        _keyword_manager = KeywordManager()
    
    return _keyword_manager