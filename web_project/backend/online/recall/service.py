"""
召回服务

协调推荐流水线的多通道召回。
从多个来源（嵌入向量、热门趋势等）检索多样化的候选。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import redis

from app.config import settings
from .base import RecallStrategy
from .youtubednn import YouTubeDNNRecallStrategy
from .trending import GlobalTrendingRecallStrategy, UserPreferenceRecallStrategy
from .item_based import ItemEmbeddingRecallStrategy
from .resource_manager import RecallResourceManager

logger = logging.getLogger(__name__)

class RecallService:
    """召回服务 - 多通道候选检索"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecallService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        # 初始化资源管理器（单例）
        self.resource_manager = RecallResourceManager()
        
        # 初始化 Redis
        try:
            self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        except Exception as e:
            logger.error(f"连接到 Redis 失败: {e}")
            self.redis_client = None
            
        # 初始化召回策略
        self.strategies: List[RecallStrategy] = [                                    
            UserPreferenceRecallStrategy(),
            ItemEmbeddingRecallStrategy(),
            GlobalTrendingRecallStrategy(),
            YouTubeDNNRecallStrategy(),
        ]

        self.merge_strategy = "weighted"
        self.weights = [4, 0, 6, 0]
        
        self.initialized = True
        logger.info("召回服务初始化完成, 多通道策略.")
        
    def _fetch_user_features_from_redis(self, user_id: int) -> Dict[str, Any]:
        """从 Redis 获取用户画像和历史"""
        if not self.redis_client:
            return {}
            
        features = {}
        try:
            # 假设键名: user:{id}:profile (哈希), user:{id}:history (movie_ids 列表)
            profile_key = f"user:{user_id}:profile"
            history_key = f"user:{user_id}:history"
            
            # 获取画像
            profile = self.redis_client.hgetall(profile_key)
            if profile:
                features.update(profile)
                
            # 获取历史
            history = self.redis_client.lrange(history_key, 0, -1)
            if history:
                # 如果需要转换为整数（Redis 返回字符串）
                features["hist_movie_ids"] = [int(mid) for mid in history]
                        
            # 如果 frequent_genres 在 Redis 中是字符串（如 "Action,Comedy"），则拆分它
            if "frequent_genres" in features and isinstance(features["frequent_genres"], str):
                features["frequent_genres"] = [g.strip() for g in features["frequent_genres"].split(",") if g.strip()]

        except Exception as e:
            logger.error(f"从 Redis 获取用户特征失败: {e}")
            
        return features

    def _merge_results_round_robin(self, results_list: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        """
        使用轮询（蛇形合并）策略合并多通道结果。
        交替从各通道获取结果以确保多样性。
        """
        merged_candidates = []
        seen_movie_ids = set()
        
        # 过滤有效结果并将其视为来源
        sources = []
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                logger.error(f"Strategy {self.strategies[i].__class__.__name__} failed: {results}")
                sources.append([])
            elif not results:
                sources.append([])
            else:
                sources.append(results)
                
        # 轮询逻辑
        direction = 1  # 1 表示正向，-1 表示反向
        current_idx = 0
        source_pointers = [0] * len(sources)
        
        while len(merged_candidates) < top_k:
            # 检查是否所有来源都已耗尽
            all_exhausted = True
            for i, src in enumerate(sources):
                if source_pointers[i] < len(src):
                    all_exhausted = False
                    break
            if all_exhausted:
                break
                
            # 尝试从当前来源获取物品
            src_list = sources[current_idx]
            ptr = source_pointers[current_idx]
            
            if ptr < len(src_list):
                item = src_list[ptr]
                # 递增该来源的指针
                source_pointers[current_idx] += 1
                
                mid = item["movie_id"]
                if mid not in seen_movie_ids:
                    merged_candidates.append(item)
                    seen_movie_ids.add(mid)
            
            # 移动到下一个来源
            current_idx += direction
            
            # 检查边界，必要时反转方向
            if direction == 1 and current_idx >= len(sources):
                direction = -1
                current_idx = len(sources) - 1
            elif direction == -1 and current_idx < 0:
                direction = 1
                current_idx = 0

        logger.info(f"RecallService._merge_results_round_robin, 召回结果: {len(merged_candidates)}")
        return merged_candidates

    def _merge_results_weighted(
        self, results_list, weights, top_k
    ):
        result = []
        seen = set()
        pointers = [0] * len(results_list)

        sources = [
            r if isinstance(r, list) else []
            for r in results_list
        ]

        while len(result) < top_k:
            added = False

            for i, weight in enumerate(weights):
                count = 0

                while (
                    count < weight
                    and pointers[i] < len(sources[i])
                ):
                    item = sources[i][pointers[i]]
                    pointers[i] += 1

                    mid = item["movie_id"]

                    if mid in seen:
                        continue

                    result.append(item)
                    seen.add(mid)

                    count += 1
                    added = True

                    if len(result) >= top_k:
                        return result

            if not added:
                break

        return result

    async def recommend(self, user_features: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        协调多通道召回
        """
        user_id = user_features.get("user_id")
        
        # 如果可用，使用 Redis 数据补充
        if user_id:
            redis_features = self._fetch_user_features_from_redis(user_id)
            # 合并：Redis 覆盖传入的特征还是相反？
            # 通常传入的特征（实时）> Redis（缓存）
            # 但这里我们想要填补空白
            for k, v in redis_features.items():
                if k not in user_features or not user_features[k]:
                    user_features[k] = v

        # 获取用户历史交互电影
        history_ids = {
            int(mid)
            for mid in user_features.get("hist_movie_ids", [])
        }

        tasks = []
        active_indices = []

        recall_k = top_k + len(history_ids)

        for i, strategy in enumerate(self.strategies):

            if (
                self.merge_strategy == "weighted"
                and self.weights[i] == 0
            ):
                continue

            tasks.append(
                strategy.recall(user_features, recall_k)
            )

            active_indices.append(i)

        active_results = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        results_list = [[] for _ in self.strategies]

        for i, results in zip(active_indices, active_results):
            results_list[i] = results

        if user_id:
            history_ids.update(
                int(mid)
                for mid in redis_features.get("hist_movie_ids", [])
            )

        # 过滤各通道中的历史电影
        for i, results in enumerate(results_list):
            if isinstance(results, list):
                results_list[i] = [
                    item for item in results
                    if int(item["movie_id"]) not in history_ids
                ]
        
        # 使用轮询合并结果
        if self.merge_strategy == "weighted":
            return self._merge_results_weighted(
                results_list,
                self.weights,
                top_k
            )

        return self._merge_results_round_robin(
            results_list, top_k
        )
        
    # 辅助方法：暴露遗留代码期望的资源属性（可选）
    @property
    def movie_genre_map(self):
        return self.resource_manager.movie_genre_map
        
    def set_movie_genre_map(self, movies):
        self.resource_manager.set_movie_genre_map(movies)


# 单例访问器
_recall_service: Optional[RecallService] = None


def get_recall_service() -> RecallService:
    """获取或创建单例 RecallService 实例"""
    global _recall_service
    if _recall_service is None:
        try:
            _recall_service = RecallService()
            logger.info("召回服务初始化完成.")
        except Exception as e:
            logger.error(f"召回服务初始化失败: {e}")
    return _recall_service

