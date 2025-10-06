import os

from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsClusterer:
    def __init__(self, mongo_uri: str, db_name: str = "finamhackathon"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.news_collection = self.db["news"]
        self.clusters_collection = self.db["clusters"]
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.8
        self.batch_interval = 300
        
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Создание необходимых индексов."""
        try:
            self.clusters_collection.create_index([("centroid", "vector")])
        except Exception as e:
            logger.warning(f"Vector index creation failed: {e}. Using fallback method.")
        
        self.news_collection.create_index([("clustered", 1)])
        self.news_collection.create_index([("cluster_id", 1)])
        self.news_collection.create_index([("created_at", -1)])
        self.clusters_collection.create_index([("cluster_id", 1)], unique=True)
        self.clusters_collection.create_index([("updated_at", -1)])
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусную схожесть между двумя векторами."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def update_cluster_centroid(self, cluster_id: int, new_embedding: np.ndarray) -> None:
        """Обновляет центроид кластера с экспоненциальным скользящим средним."""
        cluster = self.clusters_collection.find_one({"cluster_id": cluster_id})
        current_time = datetime.utcnow()
        
        if cluster:
            old_centroid = np.array(cluster["centroid"])
            size = cluster["size"]
            
            # Экспоненциальное скользящее среднее для лучшей адаптации
            alpha = 1.0 / (size + 1)
            new_centroid = old_centroid * (1 - alpha) + new_embedding * alpha
            
            update_result = self.clusters_collection.update_one(
                {"cluster_id": cluster_id},
                {"$set": {
                    "centroid": new_centroid.tolist(),
                    "size": size + 1,
                    "updated_at": current_time
                }}
            )
            logger.debug(f"Updated cluster {cluster_id}, size: {size + 1}")
        else:
            self.clusters_collection.insert_one({
                "cluster_id": cluster_id,
                "centroid": new_embedding.tolist(),
                "size": 1,
                "created_at": current_time,
                "updated_at": current_time
            })
            logger.info(f"Created new cluster {cluster_id}")
    
    def find_best_cluster_vector_search(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Ищет ближайший кластер через векторный поиск MongoDB."""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "centroid",
                        "queryVector": embedding.tolist(),
                        "numCandidates": 10,
                        "limit": 1
                    }
                },
                {
                    "$project": {
                        "cluster_id": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.clusters_collection.aggregate(pipeline))
            if results:
                best_match = results[0]
                return best_match["cluster_id"], best_match["score"]
                
        except Exception as e:
            logger.warning(f"Vector search failed: {e}. Using fallback method.")
        
        # Fallback: ручной поиск по всем кластерам
        return self._find_best_cluster_fallback(embedding)
    
    def _find_best_cluster_fallback(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Ручной поиск ближайшего кластера."""
        clusters = list(self.clusters_collection.find({}, {"cluster_id": 1, "centroid": 1}))
        
        if not clusters:
            return None, 0.0
        
        best_cluster_id = None
        max_similarity = -1.0
        
        for cluster in clusters:
            cluster_embedding = np.array(cluster["centroid"])
            similarity = self._calculate_cosine_similarity(embedding, cluster_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_cluster_id = cluster["cluster_id"]
        
        return best_cluster_id, max_similarity
    
    def get_next_cluster_id(self) -> int:
        """Генерирует следующий ID кластера."""
        last_cluster = self.clusters_collection.find_one(
            {}, 
            sort=[("cluster_id", -1)]
        )
        return (last_cluster["cluster_id"] + 1) if last_cluster else 1
    
    def assign_clusters_batch(self, texts: List[str]) -> Tuple[List[int], List[np.ndarray]]:
        """Присваивает кластеры списку текстов батчем."""
        if not texts:
            return [], []
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        assigned_clusters = []
        
        for i, embedding in enumerate(embeddings):
            cluster_id, similarity = self.find_best_cluster_vector_search(embedding)
            
            if similarity >= self.similarity_threshold:
                assigned_clusters.append(cluster_id)
                self.update_cluster_centroid(cluster_id, embedding)
                logger.debug(f"Text {i+1} assigned to existing cluster {cluster_id}, similarity: {similarity:.3f}")
            else:
                new_cluster_id = self.get_next_cluster_id()
                assigned_clusters.append(new_cluster_id)
                self.update_cluster_centroid(new_cluster_id, embedding)
                logger.info(f"Text {i+1} assigned to new cluster {new_cluster_id}, similarity: {similarity:.3f}")
        
        return assigned_clusters, embeddings
    
    def process_batch(self) -> int:
        """Обрабатывает все новые новости без cluster_id."""
        try:
            new_docs = list(self.news_collection.find(
                {"clustered": False}, 
                limit=1000  # Ограничение для избежания memory issues
            ))
            
            if not new_docs:
                logger.info("No new news to process")
                return 0
            
            texts = [doc["text"] for doc in new_docs]
            assigned_clusters, embeddings = self.assign_clusters_batch(texts)
            
            bulk_updates = []
            current_time = datetime.utcnow()
            
            for doc, cluster_id, embedding in zip(new_docs, assigned_clusters, embeddings):
                bulk_updates.append(UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "cluster_id": cluster_id,
                        "embedding": embedding.tolist(),
                        "clustered": True,
                        "updated_at": current_time
                    }}
                ))
            
            if bulk_updates:
                result = self.news_collection.bulk_write(bulk_updates, ordered=False)
                logger.info(f"Processed {len(bulk_updates)} news items in batch")
                return len(bulk_updates)
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
        return 0
    
    def process_single_news(self, text: str, **additional_fields) -> int:
        """Обрабатывает одну новость в реальном времени."""
        try:
            embedding = self.model.encode([text])[0]
            cluster_id, similarity = self.find_best_cluster_vector_search(embedding)
            current_time = datetime.utcnow()
            
            if similarity >= self.similarity_threshold:
                self.update_cluster_centroid(cluster_id, embedding)
                logger.debug(f"News assigned to existing cluster {cluster_id}, similarity: {similarity:.3f}")
            else:
                cluster_id = self.get_next_cluster_id()
                self.update_cluster_centroid(cluster_id, embedding)
                logger.info(f"News assigned to new cluster {cluster_id}, similarity: {similarity:.3f}")
            
            news_doc = {
                "text": text,
                "embedding": embedding.tolist(),
                "cluster_id": cluster_id,
                "clustered": True,
                "created_at": current_time,
                "updated_at": current_time,
                **additional_fields
            }
            
            self.news_collection.insert_one(news_doc)
            logger.info(f"Single news processed, cluster_id: {cluster_id}")
            return cluster_id
            
        except Exception as e:
            logger.error(f"Error processing single news: {e}")
            raise
    
    def get_cluster_stats(self) -> dict:
        """Возвращает статистику по кластерам."""
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_clusters": {"$sum": 1},
                    "total_news": {"$sum": "$size"},
                    "avg_cluster_size": {"$avg": "$size"},
                    "max_cluster_size": {"$max": "$size"},
                    "min_cluster_size": {"$min": "$size"}
                }
            }
        ]
        
        stats = list(self.clusters_collection.aggregate(pipeline))
        return stats[0] if stats else {}
    
    def run_continuous_processing(self):
        """Запускает непрерывную обработку новостей."""
        logger.info("Starting continuous news clustering...")
        
        while True:
            try:
                processed_count = self.process_batch()
                if processed_count > 0:
                    stats = self.get_cluster_stats()
                    logger.info(f"Cluster stats: {stats}")
                
                time.sleep(self.batch_interval)
                
            except KeyboardInterrupt:
                logger.info("Processing stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous processing: {e}")
                time.sleep(self.batch_interval)  # Продолжаем после ошибки
    
    def close(self):
        """Закрывает соединения."""
        self.client.close()


if __name__ == "__main__":
    MONGO_URI = os.getenv('MONGO')
    
    clusterer = NewsClusterer(MONGO_URI)
    
    try:
        # Пример обработки одной новости
        clusterer.process_single_news("Рынок акций показал рост сегодня")
        
        # Запуск непрерывной обработки
        # clusterer.run_continuous_processing()
        
    finally:
        clusterer.close()