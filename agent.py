# performance_agent/agent.py
"""
PERFORMANCE AGENT
Instagram Content Performance Analysis Agent

Architecture:
- DataLoader: Fetches data from files (mock) or Instagram API (production)
- Analyzers: Vision AI, Caption NLP, Comment Sentiment
- InsightGenerator: Patterns + Recommendations
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Post:
    """Represents a single Instagram post"""
    id: str
    media_type: str
    media_url: str
    local_file: str
    caption: str
    timestamp: str
    permalink: str
    like_count: int
    comments_count: int
    impressions: int
    reach: int
    saved: int
    shares: int
    comments: list
    
    # Analysis results (filled by agent)
    content_type: Optional[str] = None
    vision_analysis: Optional[dict] = None
    caption_analysis: Optional[dict] = None
    comment_analysis: Optional[dict] = None
    engagement_rate: Optional[float] = None
    conversion_score: Optional[float] = None


@dataclass
class Account:
    """Represents an Instagram account"""
    ig_user_id: str
    username: str
    name: str
    followers_count: int
    media_count: int


# =============================================================================
# DATA LOADER (Swappable: File vs API)
# =============================================================================

class DataLoaderBase(ABC):
    """Abstract base class for data loading"""
    
    @abstractmethod
    def load_account(self, account_id: str) -> Account:
        pass
    
    @abstractmethod
    def load_posts(self, account_id: str, limit: int = 50) -> List[Post]:
        pass


class FileDataLoader(DataLoaderBase):
    """Loads data from local JSON files (for demo/testing)"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self._cache = {}
    
    def _load_json(self, account_id: str) -> Dict:
        """Load and cache JSON file"""
        if account_id not in self._cache:
            # Map account_id to filename
            file_map = {
                "laneige_jp": "api_mock_laneige_jp.json",
                "laneigethailand": "api_mock_laneige_th.json", 
                "laneigesg": "api_mock_laneige_sg.json"
            }
            
            filename = file_map.get(account_id)
            if not filename:
                raise ValueError(f"Unknown account: {account_id}")
            
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                self._cache[account_id] = json.load(f)
        
        return self._cache[account_id]
    
    def load_account(self, account_id: str) -> Account:
        """Load account info from JSON file"""
        data = self._load_json(account_id)
        acc = data["account"]
        return Account(
            ig_user_id=acc["ig_user_id"],
            username=acc["username"],
            name=acc["name"],
            followers_count=acc["followers_count"],
            media_count=acc["media_count"]
        )
    
    def load_posts(self, account_id: str, limit: int = 50) -> List[Post]:
        """Load posts from JSON file"""
        data = self._load_json(account_id)
        posts = []
        
        for media in data["media"][:limit]:
            # Extract insights from nested structure
            insights = {item["name"]: item["values"][0]["value"] 
                       for item in media.get("insights", {}).get("data", [])}
            
            # Extract comment texts
            comments = [c["text"] for c in media.get("comments", {}).get("data", [])]
            
            post = Post(
                id=media["id"],
                media_type=media["media_type"],
                media_url=media["media_url"],
                local_file=media.get("local_file", ""),
                caption=media["caption"],
                timestamp=media["timestamp"],
                permalink=media["permalink"],
                like_count=media["like_count"],
                comments_count=media["comments_count"],
                impressions=insights.get("impressions", 0),
                reach=insights.get("reach", 0),
                saved=insights.get("saved", 0),
                shares=insights.get("shares", 0),
                comments=comments
            )
            posts.append(post)
        
        return posts


class InstagramAPILoader(DataLoaderBase):
    """
    Loads data from Instagram Graph API (for production)
    
    TODO: Implement when ready for production
    - Requires: access_token, app_id, app_secret
    - Uses: requests library to call Meta Graph API
    """
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def load_account(self, account_id: str) -> Account:
        # TODO: Implement API call
        # GET /{ig-user-id}?fields=username,name,followers_count,media_count
        raise NotImplementedError("Instagram API loader not yet implemented")
    
    def load_posts(self, account_id: str, limit: int = 50) -> List[Post]:
        # TODO: Implement API call
        # GET /{ig-user-id}/media?fields=id,caption,media_type,timestamp,...
        raise NotImplementedError("Instagram API loader not yet implemented")


# =============================================================================
# ANALYZERS (Swappable: Mock vs Claude API)
# =============================================================================

class AnalyzerBase(ABC):
    """Abstract base class for analyzers"""
    
    @abstractmethod
    def analyze_vision(self, image_path: str) -> Dict:
        pass
    
    @abstractmethod
    def analyze_caption(self, caption: str) -> Dict:
        pass
    
    @abstractmethod
    def analyze_comments(self, comments: List[str]) -> Dict:
        pass


class MockAnalyzer(AnalyzerBase):
    """
    Returns pre-computed analysis results (for demo)
    Uses rules-based logic to simulate AI analysis
    """
    
    # Content type mapping based on filename
    CONTENT_TYPE_MAP = {
        "1__제품_단체샷.jpg": "product_group",
        "2__제품_단독샷.jpg": "product_solo",
        "3__제품_질감샷.jpg": "product_texture",
        "4__제품_및_모델샷.jpg": "product_model",
        "5__제품없는_모델샷.jpg": "model_only",
        "6__제품_모델_단체샷.jpg": "product_models_group"
    }
    
    # Pre-defined vision analysis for each content type
    VISION_ANALYSIS = {
        "product_group": {
            "subject_type": "product_only",
            "product_visibility": "hero_shot",
            "product_count": 3,
            "human_presence": False,
            "face_visible": False,
            "composition_style": "minimal",
            "background": "gradient_light",
            "text_overlay_percentage": 0,
            "dominant_colors": ["pink", "blue", "green", "white"],
            "lighting": "studio_soft",
            "mood": "clean_organized",
            "production_quality": "professional"
        },
        "product_solo": {
            "subject_type": "product_only",
            "product_visibility": "hero_shot",
            "product_count": 2,
            "human_presence": False,
            "face_visible": False,
            "composition_style": "minimal",
            "background": "solid_white",
            "text_overlay_percentage": 0,
            "dominant_colors": ["blue", "white"],
            "lighting": "studio_bright",
            "mood": "clean_luxurious",
            "production_quality": "professional"
        },
        "product_texture": {
            "subject_type": "product_detail",
            "product_visibility": "detail_shot",
            "product_count": 1,
            "human_presence": False,
            "face_visible": False,
            "composition_style": "minimal",
            "background": "solid_white",
            "text_overlay_percentage": 0,
            "dominant_colors": ["blue", "white"],
            "lighting": "studio_soft",
            "mood": "clinical_premium",
            "production_quality": "professional",
            "special_elements": ["texture_focus", "cream_visible"]
        },
        "product_model": {
            "subject_type": "person_with_product",
            "product_visibility": "held_by_model",
            "product_count": 1,
            "human_presence": True,
            "face_visible": True,
            "person_count": 1,
            "composition_style": "moderate",
            "background": "solid_light_gray",
            "text_overlay_percentage": 0,
            "dominant_colors": ["blue", "skin_tone", "white"],
            "lighting": "studio_beauty",
            "mood": "aspirational_friendly",
            "production_quality": "professional"
        },
        "model_only": {
            "subject_type": "person_only",
            "product_visibility": "not_visible",
            "product_count": 0,
            "human_presence": True,
            "face_visible": True,
            "person_count": 1,
            "composition_style": "minimal",
            "background": "solid_light_gray",
            "text_overlay_percentage": 0,
            "dominant_colors": ["skin_tone", "white"],
            "lighting": "studio_beauty",
            "mood": "authentic_skincare",
            "production_quality": "professional"
        },
        "product_models_group": {
            "subject_type": "people_with_products",
            "product_visibility": "held_by_models",
            "product_count": 3,
            "human_presence": True,
            "face_visible": True,
            "person_count": 3,
            "composition_style": "moderate",
            "background": "solid_light_purple",
            "text_overlay_percentage": 0,
            "dominant_colors": ["pink", "blue", "green", "white", "skin_tones"],
            "lighting": "studio_bright",
            "mood": "inclusive_joyful",
            "production_quality": "professional",
            "special_elements": ["diverse_models", "group_shot"]
        }
    }
    
    def analyze_vision(self, image_path: str) -> Dict:
        """Return pre-computed vision analysis based on filename"""
        filename = os.path.basename(image_path)
        content_type = self.CONTENT_TYPE_MAP.get(filename, "unknown")
        
        analysis = self.VISION_ANALYSIS.get(content_type, {}).copy()
        analysis["content_type"] = content_type
        analysis["image_file"] = filename
        
        return analysis
    
    def analyze_caption(self, caption: str) -> Dict:
        """Analyze caption using rule-based logic"""
        import re
        
        # Count hashtags
        hashtags = re.findall(r'#\w+', caption)
        
        # Count emojis (simplified)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", 
            flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(caption)
        
        # Detect CTA patterns
        cta_patterns = [
            r'swipe', r'➡️', r'👇', r'👆', r'comment', r'コメント',
            r'教えて', r'drop a', r'double tap', r'link in bio',
            r'ซื้อ', r'สั่ง'  # Thai purchase words
        ]
        has_cta = any(re.search(p, caption.lower()) for p in cta_patterns)
        
        # Detect mentions
        mentions = re.findall(r'@\w+', caption)
        
        # Detect language (simplified)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', caption):
            language = "ja"
        elif re.search(r'[\u0E00-\u0E7F]', caption):
            language = "th"
        else:
            language = "en"
        
        # Determine tone
        if has_cta:
            tone = "engaging"
        elif any(word in caption.lower() for word in ['✨', 'secret', '秘密', 'เคล็ดลับ']):
            tone = "aspirational"
        elif len(hashtags) > 5:
            tone = "promotional"
        else:
            tone = "informative"
        
        return {
            "caption_length": len(caption),
            "hashtag_count": len(hashtags),
            "hashtags": hashtags,
            "emoji_count": len(emojis),
            "cta_present": has_cta,
            "mention_count": len(mentions),
            "mentions": mentions,
            "language": language,
            "tone": tone
        }
    
    def analyze_comments(self, comments: List[str]) -> Dict:
        """Analyze comments using rule-based sentiment"""
        if not comments:
            return {
                "total_comments": 0,
                "sentiment_score": 0,
                "positive_ratio": 0,
                "neutral_ratio": 0,
                "negative_ratio": 0,
                "top_topics": [],
                "purchase_intent_count": 0,
                "questions_count": 0
            }
        
        # Positive indicators
        positive_words = [
            '好き', '最高', '大好き', '綺麗', '可愛い', '素敵', 'リピート',  # Japanese
            'love', 'amazing', 'best', 'perfect', 'beautiful', 'great', 'hg',  # English
            'สวย', 'ดี', 'ชอบ', 'รัก', 'เด้ง', 'ปัง',  # Thai
            '😍', '💙', '💕', '✨', '🥰', '❤️'  # Emojis
        ]
        
        # Negative indicators
        negative_words = [
            'べたつく', '高い', '合わない',  # Japanese
            'sticky', 'expensive', 'broke out', 'didnt work',  # English
            'แพง', 'เหนียว',  # Thai
            '😢', '😭'
        ]
        
        # Question indicators
        question_patterns = ['?', '？', 'ますか', 'ですか', 'มั้ย', 'คะ', 'ไหม']
        
        # Purchase intent indicators
        purchase_words = [
            '買', '欲しい', 'どこで', 'リピート',  # Japanese
            'buy', 'order', 'where', 'price', 'want',  # English
            'ซื้อ', 'สั่ง', 'ราคา', 'อยากได้'  # Thai
        ]
        
        # Topic indicators
        topic_keywords = {
            'texture': ['テクスチャー', '質感', 'texture', 'เนื้อ', 'gel', 'ジェル'],
            'hydration': ['保湿', '潤い', 'hydrat', 'moisture', 'ชุ่มชื้น'],
            'skin': ['肌', 'skin', 'ผิว', 'glow'],
            'price': ['値段', '高い', 'price', 'ราคา', 'แพง'],
            'purchase_intent': purchase_words,
            'scent': ['香り', '匂い', 'scent', 'smell', 'กลิ่น']
        }
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        questions_count = 0
        purchase_intent_count = 0
        topic_counts = {topic: 0 for topic in topic_keywords}
        
        for comment in comments:
            comment_lower = comment.lower()
            
            # Sentiment
            is_positive = any(word in comment_lower or word in comment for word in positive_words)
            is_negative = any(word in comment_lower or word in comment for word in negative_words)
            
            if is_positive and not is_negative:
                positive_count += 1
            elif is_negative:
                negative_count += 1
            else:
                neutral_count += 1
            
            # Questions
            if any(p in comment for p in question_patterns):
                questions_count += 1
            
            # Purchase intent
            if any(word in comment_lower or word in comment for word in purchase_words):
                purchase_intent_count += 1
            
            # Topics
            for topic, keywords in topic_keywords.items():
                if any(kw in comment_lower or kw in comment for kw in keywords):
                    topic_counts[topic] += 1
        
        total = len(comments)
        
        # Get top topics
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [t[0] for t in top_topics if t[1] > 0][:5]
        
        return {
            "total_comments": total,
            "sentiment_score": round(positive_count / total if total > 0 else 0, 2),
            "positive_ratio": round(positive_count / total if total > 0 else 0, 2),
            "neutral_ratio": round(neutral_count / total if total > 0 else 0, 2),
            "negative_ratio": round(negative_count / total if total > 0 else 0, 2),
            "top_topics": top_topics,
            "purchase_intent_count": purchase_intent_count,
            "questions_count": questions_count
        }


class ClaudeAnalyzer(AnalyzerBase):
    """
    Uses Claude API for real AI analysis (for production)
    
    Requires: 
    - ANTHROPIC_API_KEY
    - pip install anthropic
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def analyze_vision(self, image_path: str) -> Dict:
        """Analyze image using Claude Vision API"""
        import base64
        
        # Read image file
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Detect actual image type from file header (magic bytes)
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            media_type = "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            media_type = "image/webp"
        else:
            # Default to jpeg
            media_type = "image/jpeg"
        
        # Encode to base64
        image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
        
        prompt = """Analyze this Instagram post image for a beauty brand. Return ONLY a JSON object with these fields:

{
  "content_type": "product_solo" | "product_group" | "product_texture" | "product_model" | "model_only" | "product_models_group",
  "subject_type": "product_only" | "product_detail" | "person_with_product" | "person_only" | "people_with_products",
  "product_visibility": "hero_shot" | "detail_shot" | "held_by_model" | "not_visible" | "background",
  "product_count": number,
  "human_presence": boolean,
  "face_visible": boolean,
  "person_count": number,
  "composition_style": "minimal" | "moderate" | "busy",
  "background": "solid_white" | "solid_color" | "gradient" | "studio" | "outdoor" | "lifestyle",
  "text_overlay_percentage": number (0-100),
  "dominant_colors": ["color1", "color2"],
  "lighting": "studio_bright" | "studio_soft" | "natural" | "dramatic",
  "mood": "clean_luxurious" | "aspirational" | "playful" | "clinical" | "inclusive_joyful" | "authentic",
  "production_quality": "professional" | "semi_pro" | "ugc_style"
}

Return ONLY valid JSON, no other text."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        # Parse JSON response
        try:
            response_text = response.content[0].text
            # Strip markdown code blocks if present
            if response_text.startswith("```"):
                # Remove ```json and ``` 
                lines = response_text.strip().split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]  # Remove first line (```json)
                if lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove last line (```)
                response_text = "\n".join(lines)
            
            result = json.loads(response_text)
            result["image_file"] = os.path.basename(image_path)
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return basic result
            return {
                "content_type": "unknown",
                "image_file": os.path.basename(image_path),
                "raw_response": response.content[0].text
            }
    
    def analyze_caption(self, caption: str) -> Dict:
        """Analyze caption using Claude API"""
        
        prompt = f"""Analyze this Instagram caption for a beauty brand. Return ONLY a JSON object:

Caption: {caption}

Return this JSON structure:
{{
  "caption_length": number,
  "hashtag_count": number,
  "hashtags": ["list", "of", "hashtags"],
  "emoji_count": number,
  "cta_present": boolean,
  "cta_type": "swipe" | "comment" | "link" | "purchase" | "none",
  "mention_count": number,
  "mentions": ["@mention1"],
  "language": "en" | "ja" | "th" | "ko" | "zh",
  "tone": "informative" | "aspirational" | "engaging" | "promotional" | "educational"
}}

Return ONLY valid JSON, no other text."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            # Fallback to mock analyzer for caption
            mock = MockAnalyzer()
            return mock.analyze_caption(caption)
    
    def analyze_comments(self, comments: List[str]) -> Dict:
        """Analyze comments using Claude API"""
        
        if not comments:
            return {
                "total_comments": 0,
                "sentiment_score": 0,
                "positive_ratio": 0,
                "neutral_ratio": 0,
                "negative_ratio": 0,
                "top_topics": [],
                "purchase_intent_count": 0,
                "questions_count": 0
            }
        
        comments_text = "\n".join([f"- {c}" for c in comments[:20]])  # Limit to 20
        
        prompt = f"""Analyze these Instagram comments for a beauty brand post. Return ONLY a JSON object:

Comments:
{comments_text}

Return this JSON structure:
{{
  "total_comments": number,
  "sentiment_score": number (0-1, where 1 is most positive),
  "positive_ratio": number (0-1),
  "neutral_ratio": number (0-1),
  "negative_ratio": number (0-1),
  "top_topics": ["topic1", "topic2", "topic3"],
  "purchase_intent_count": number (comments showing intent to buy),
  "questions_count": number (comments asking questions)
}}

Return ONLY valid JSON, no other text."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            result = json.loads(response.content[0].text)
            result["total_comments"] = len(comments)
            return result
        except json.JSONDecodeError:
            mock = MockAnalyzer()
            return mock.analyze_comments(comments)


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """Calculates engagement rate and conversion score"""
    
    @staticmethod
    def calculate_engagement_rate(post: Post, followers: int) -> float:
        """
        Engagement Rate = (Likes + Comments + Saves + Shares) / Reach * 100
        """
        if post.reach == 0:
            return 0.0
        
        total_engagement = post.like_count + post.comments_count + post.saved + post.shares
        return round((total_engagement / post.reach) * 100, 2)
    
    @staticmethod
    def calculate_conversion_score(post: Post, comment_analysis: dict) -> float:
        """
        Conversion Score (1-10) based on:
        - Saves rate (40%)
        - Purchase intent comments (30%)
        - Shares rate (20%)
        - Overall engagement (10%)
        """
        if post.reach == 0:
            return 0.0
        
        # Saves rate (normalized to 0-10)
        saves_rate = (post.saved / post.reach) * 100
        saves_score = min(saves_rate * 5, 10)  # Cap at 10
        
        # Purchase intent ratio
        purchase_intent = comment_analysis.get("purchase_intent_count", 0)
        total_comments = comment_analysis.get("total_comments", 1)
        purchase_score = min((purchase_intent / total_comments) * 10, 10) if total_comments > 0 else 0
        
        # Shares rate
        shares_rate = (post.shares / post.reach) * 100
        shares_score = min(shares_rate * 10, 10)
        
        # Overall engagement
        engagement_score = min(post.like_count / post.reach * 50, 10)
        
        # Weighted score
        score = (
            saves_score * 0.4 +
            purchase_score * 0.3 +
            shares_score * 0.2 +
            engagement_score * 0.1
        )
        
        return round(score, 1)


# =============================================================================
# INSIGHT GENERATOR
# =============================================================================

class InsightGenerator:
    """Generates insights from analyzed posts"""
    
    def __init__(self):
        self.content_type_labels = {
            "product_group": "제품 단체샷",
            "product_solo": "제품 단독샷",
            "product_texture": "제품 질감샷",
            "product_model": "제품 및 모델샷",
            "model_only": "제품없는 모델샷",
            "product_models_group": "제품 모델 단체샷"
        }
    
    def generate_account_insights(self, account: Account, posts: List[Post]) -> Dict:
        """Generate insights for a single account"""
        
        # Group posts by content type
        by_content_type = {}
        for post in posts:
            ct = post.content_type or "unknown"
            if ct not in by_content_type:
                by_content_type[ct] = []
            by_content_type[ct].append(post)
        
        # Calculate average engagement by content type
        engagement_by_type = {}
        for ct, ct_posts in by_content_type.items():
            avg_engagement = sum(p.engagement_rate or 0 for p in ct_posts) / len(ct_posts)
            engagement_by_type[ct] = round(avg_engagement, 2)
        
        # Sort by engagement
        sorted_types = sorted(engagement_by_type.items(), key=lambda x: x[1], reverse=True)
        
        # Find top and bottom performers
        top_type = sorted_types[0] if sorted_types else (None, 0)
        bottom_type = sorted_types[-1] if sorted_types else (None, 0)
        
        # Calculate overall stats
        all_engagement = [p.engagement_rate for p in posts if p.engagement_rate]
        avg_engagement = sum(all_engagement) / len(all_engagement) if all_engagement else 0
        
        # Determine if account prefers human content
        human_types = ["product_model", "model_only", "product_models_group"]
        product_types = ["product_group", "product_solo", "product_texture"]
        
        human_engagement = [engagement_by_type.get(t, 0) for t in human_types if t in engagement_by_type]
        product_engagement = [engagement_by_type.get(t, 0) for t in product_types if t in engagement_by_type]
        
        avg_human = sum(human_engagement) / len(human_engagement) if human_engagement else 0
        avg_product = sum(product_engagement) / len(product_engagement) if product_engagement else 0
        
        content_preference = "human_focused" if avg_human > avg_product else "product_focused"
        preference_strength = abs(avg_human - avg_product)
        
        # Generate recommendation
        top_label = self.content_type_labels.get(top_type[0], top_type[0])
        bottom_label = self.content_type_labels.get(bottom_type[0], bottom_type[0])
        
        if content_preference == "product_focused":
            recommendation = f"제품 중심 콘텐츠 권장. {top_label} 형식이 가장 효과적 (평균 {top_type[1]}% engagement)"
        else:
            recommendation = f"인물 중심 콘텐츠 권장. {top_label} 형식이 가장 효과적 (평균 {top_type[1]}% engagement)"
        
        return {
            "account": account.username,
            "total_posts_analyzed": len(posts),
            "avg_engagement_rate": round(avg_engagement, 2),
            "engagement_by_content_type": engagement_by_type,
            "top_performing_type": {
                "type": top_type[0],
                "label": top_label,
                "avg_engagement": top_type[1]
            },
            "bottom_performing_type": {
                "type": bottom_type[0],
                "label": bottom_label,
                "avg_engagement": bottom_type[1]
            },
            "content_preference": content_preference,
            "preference_strength": round(preference_strength, 2),
            "recommendation": recommendation
        }
    
    def generate_comparison_insights(self, all_insights: dict) -> Dict:
        """Compare insights across multiple accounts"""
        
        comparisons = []
        recommendations = []
        
        # Extract data
        accounts = list(all_insights.keys())
        
        for account, insights in all_insights.items():
            pref = insights["content_preference"]
            top = insights["top_performing_type"]["label"]
            eng = insights["avg_engagement_rate"]
            
            comparisons.append({
                "account": account,
                "preference": pref,
                "top_content": top,
                "avg_engagement": eng
            })
        
        # Generate cross-market insights
        product_focused = [c for c in comparisons if c["preference"] == "product_focused"]
        human_focused = [c for c in comparisons if c["preference"] == "human_focused"]
        
        if product_focused:
            accounts_str = ", ".join([c["account"] for c in product_focused])
            recommendations.append(f"제품 중심 시장 ({accounts_str}): 제품 히어로샷, 텍스처 강조 콘텐츠 권장")
        
        if human_focused:
            accounts_str = ", ".join([c["account"] for c in human_focused])
            recommendations.append(f"인물 중심 시장 ({accounts_str}): 모델 포함, 사용 씬 콘텐츠 권장")
        
        return {
            "accounts_analyzed": accounts,
            "comparisons": comparisons,
            "cross_market_recommendations": recommendations
        }


# =============================================================================
# MAIN AGENT
# =============================================================================

class PerformanceAgent:
    """
    Main agent that orchestrates data loading, analysis, and insight generation
    """
    
    def __init__(
        self,
        data_loader: DataLoaderBase,
        analyzer: AnalyzerBase,
        image_dir: str = "images"
    ):
        self.data_loader = data_loader
        self.analyzer = analyzer
        self.image_dir = image_dir
        self.metrics = MetricsCalculator()
        self.insight_generator = InsightGenerator()
    
    def _find_image_file(self, filename: str) -> str:
        """Find image file, handling Unicode normalization (Mac NFD vs NFC)"""
        import unicodedata
        
        # Direct path
        direct_path = os.path.join(self.image_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        # Try with Unicode normalization
        normalized_filename = unicodedata.normalize("NFC", filename)
        
        # List all files and find match
        for f in os.listdir(self.image_dir):
            if unicodedata.normalize("NFC", f) == normalized_filename:
                return os.path.join(self.image_dir, f)
        
        # Fallback to direct path (will fail with clear error)
        return direct_path
    
    def analyze_post(self, post: Post, followers: int) -> Post:
        """Run full analysis on a single post"""
        
        # Vision analysis
        image_path = self._find_image_file(post.local_file)
        post.vision_analysis = self.analyzer.analyze_vision(image_path)
        post.content_type = post.vision_analysis.get("content_type", "unknown")
        
        # Caption analysis
        post.caption_analysis = self.analyzer.analyze_caption(post.caption)
        
        # Comment analysis
        post.comment_analysis = self.analyzer.analyze_comments(post.comments)
        
        # Metrics
        post.engagement_rate = self.metrics.calculate_engagement_rate(post, followers)
        post.conversion_score = self.metrics.calculate_conversion_score(post, post.comment_analysis)
        
        return post
    
    def analyze_account(self, account_id: str) -> Dict:
        """Run full analysis on an account"""
        
        # Load data
        account = self.data_loader.load_account(account_id)
        posts = self.data_loader.load_posts(account_id)
        
        # Analyze each post
        analyzed_posts = []
        for post in posts:
            analyzed_post = self.analyze_post(post, account.followers_count)
            analyzed_posts.append(analyzed_post)
        
        # Generate insights
        insights = self.insight_generator.generate_account_insights(account, analyzed_posts)
        
        return {
            "account": {
                "username": account.username,
                "name": account.name,
                "followers": account.followers_count
            },
            "posts": [self._post_to_dict(p) for p in analyzed_posts],
            "insights": insights
        }
    
    def analyze_multiple_accounts(self, account_ids: List[str]) -> Dict:
        """Analyze multiple accounts and compare"""
        
        all_results = {}
        all_insights = {}
        
        for account_id in account_ids:
            result = self.analyze_account(account_id)
            all_results[account_id] = result
            all_insights[account_id] = result["insights"]
        
        # Generate comparison
        comparison = self.insight_generator.generate_comparison_insights(all_insights)
        
        return {
            "accounts": all_results,
            "comparison": comparison
        }
    
    def _post_to_dict(self, post: Post) -> Dict:
        """Convert Post dataclass to dictionary"""
        return {
            "id": post.id,
            "media_type": post.media_type,
            "local_file": post.local_file,
            "caption": post.caption,
            "timestamp": post.timestamp,
            "permalink": post.permalink,
            "metrics": {
                "likes": post.like_count,
                "comments": post.comments_count,
                "saves": post.saved,
                "shares": post.shares,
                "reach": post.reach,
                "impressions": post.impressions
            },
            "analysis": {
                "content_type": post.content_type,
                "vision": post.vision_analysis,
                "caption": post.caption_analysis,
                "comments": post.comment_analysis,
                "engagement_rate": post.engagement_rate,
                "conversion_score": post.conversion_score
            }
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_agent(mode: str = "demo", **kwargs) -> PerformanceAgent:
    """
    Factory function to create agent with appropriate loader and analyzer
    
    Args:
        mode: "demo", "production", or "hybrid"
        **kwargs: Additional arguments (api_key, data_dir, etc.)
    
    Usage:
        # Demo mode (default) - mock data, mock analysis
        agent = create_agent("demo", data_dir="data/raw")
        
        # Production mode - real Instagram API, real Claude API
        agent = create_agent("production", 
                            instagram_token="...", 
                            claude_api_key="...")
        
        # Hybrid mode - mock data files, real Claude API
        agent = create_agent("hybrid",
                            data_dir="data/raw",
                            claude_api_key="...")
    """
    
    if mode == "demo":
        data_loader = FileDataLoader(data_dir=kwargs.get("data_dir", "data/raw"))
        analyzer = MockAnalyzer()
    
    elif mode == "production":
        # Check for Claude API key
        claude_api_key = kwargs.get("claude_api_key")
        if not claude_api_key:
            raise ValueError("Production mode requires claude_api_key")
        
        # Use file loader if no instagram token (hybrid behavior)
        instagram_token = kwargs.get("instagram_token")
        if instagram_token:
            data_loader = InstagramAPILoader(access_token=instagram_token)
        else:
            # Fallback to file loader
            data_loader = FileDataLoader(data_dir=kwargs.get("data_dir", "data/raw"))
        
        analyzer = ClaudeAnalyzer(api_key=claude_api_key)
    
    elif mode == "hybrid":
        # File data + real Claude API
        claude_api_key = kwargs.get("claude_api_key")
        if not claude_api_key:
            raise ValueError("Hybrid mode requires claude_api_key")
        
        data_loader = FileDataLoader(data_dir=kwargs.get("data_dir", "data/raw"))
        analyzer = ClaudeAnalyzer(api_key=claude_api_key)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return PerformanceAgent(
        data_loader=data_loader,
        analyzer=analyzer,
        image_dir=kwargs.get("image_dir", "images")
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Default to demo mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    print(f"Running Performance Agent in {mode} mode...")
    
    # Create agent
    agent = create_agent(mode, data_dir="data/raw", image_dir="images")
    
    # Analyze all accounts
    accounts = ["laneige_jp", "laneigethailand", "laneigesg"]
    
    results = agent.analyze_multiple_accounts(accounts)
    
    # Output results
    output_file = "analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for account_id, data in results["accounts"].items():
        insights = data["insights"]
        print(f"\n📊 {account_id}")
        print(f"   Avg Engagement: {insights['avg_engagement_rate']}%")
        print(f"   Top Content: {insights['top_performing_type']['label']}")
        print(f"   Preference: {insights['content_preference']}")
        print(f"   💡 {insights['recommendation']}")
    
    print("\n" + "="*60)
    print("CROSS-MARKET RECOMMENDATIONS")
    print("="*60)
    for rec in results["comparison"]["cross_market_recommendations"]:
        print(f"   • {rec}")
