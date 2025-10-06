import requests
import json
import time
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class OpenRouterClient:
    """
    Клиент для работы с OpenRouter API
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-4o"):
        """
        Инициализация клиента
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("API ключ не найден. Укажите в .env файле или передайте в конструктор")

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(
            self,
            messages: List[Dict[str, str]],
            model: str = "openai/gpt-4o",
            max_tokens: int = 1500,
            temperature: float = 0.7,
            max_retries: int = 3,
            timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Базовый метод для отправки запросов к API
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=self.base_url,
                    headers=self.default_headers,
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    data = response.json()

                    if ('choices' in data and len(data['choices']) > 0 and
                            'message' in data['choices'][0] and
                            'content' in data['choices'][0]['message']):

                        return {
                            "success": True,
                            "content": data['choices'][0]['message']['content'],
                            "model": data.get('model', 'unknown'),
                            "usage": data.get('usage', {}),
                            "full_response": data
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Неверный формат ответа от API",
                            "response": data
                        }

                elif response.status_code == 401:
                    return {
                        "success": False,
                        "error": "Ошибка авторизации: неверный API ключ",
                        "status_code": 401
                    }
                elif response.status_code == 429:
                    retry_after = response.headers.get('Retry-After', 60)
                    return {
                        "success": False,
                        "error": "Превышен лимит запросов. Попробуйте позже.",
                        "status_code": 429,
                        "retry_after": retry_after
                    }
                elif 400 <= response.status_code < 500:
                    return {
                        "success": False,
                        "error": f"Ошибка клиента: {response.status_code}",
                        "status_code": response.status_code,
                        "details": response.text
                    }
                elif 500 <= response.status_code < 600:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Серверная ошибка после {max_retries} попыток",
                            "status_code": response.status_code,
                            "details": response.text
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Неизвестная ошибка HTTP: {response.status_code}",
                        "status_code": response.status_code,
                        "details": response.text
                    }

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Таймаут запроса после {max_retries} попыток"
                    }

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Ошибка подключения после {max_retries} попыток"
                    }

            except requests.exceptions.RequestException as e:
                return {
                    "success": False,
                    "error": f"Ошибка сети: {str(e)}"
                }

            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Ошибка декодирования JSON: {str(e)}",
                    "response_text": response.text if 'response' in locals() else 'N/A'
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Неожиданная ошибка: {str(e)}"
                }

        return {
            "success": False,
            "error": "Превышено максимальное количество попыток"
        }

    def _extract_entities(self, content: str) -> Dict[str, Any]:
        """
        Надежное извлечение сущностей через отдельный запрос к AI
        """
        entity_prompt = f"""
        Ты - финансовый аналитик. Извлеки ВСЕ сущности из текста новости и определи сентимент.

        ТЕКСТ:
        {content}

        ИЗВЛЕКИ ТОЧНО СЛЕДУЮЩИЕ СУЩНОСТИ:

        1. КОМПАНИИ - все упомянутые компании
        2. ТИКЕРЫ - биржевые тикеры для КАЖДОЙ найденной компании
        3. СТРАНЫ - все упомянутые страны
        4. СЕКТОРА - отрасли экономики
        5. ВАЛЮТЫ - валюты
        6. ПЕРСОНЫ - упомянутые люди
        7. СЕНТИМЕНТ_SCORE - числовая оценка от -1.0 до 1.0

        ОСОБОЕ ВНИМАНИЕ ТИКЕРАМ:
        - Сбербанк → SBER
        - Газпром → GAZP  
        - Лукойл → LKOH
        - Роснефть → ROSN
        - Норникель → GMKN

        ФОРМАТ ОТВЕТА - ТОЛЬКО JSON:
        {{
            "companies": ["Сбербанк", "Газпром"],
            "tickers": ["SBER", "GAZP"],
            "countries": ["Россия"],
            "sectors": ["финансы", "нефтегазовый"],
            "currencies": ["RUB"],
            "people": [],
            "sentiment_score": 0.7
        }}

        КРИТИЧЕСКИ ВАЖНО:
        - Для КАЖДОЙ компании ДОЛЖЕН быть тикер
        - Определи сентимент на основе общего тона новости
        - Ответ ТОЛЬКО JSON, без лишнего текста
        """

        messages = [{"role": "user", "content": entity_prompt}]
        response = self._make_request(
            messages,
            max_tokens=1000,
            temperature=0.1,
            model="openai/gpt-4o"
        )

        if response.get("success"):
            try:
                content = response["content"].strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                entities = json.loads(content)

                # Валидация
                if entities.get("companies") and not entities.get("tickers"):
                    entities["tickers"] = ["UNKNOWN"] * len(entities["companies"])

                return entities

            except Exception:
                return self._create_empty_entities()
        else:
            return self._create_empty_entities()

    def _create_empty_entities(self) -> Dict[str, List[str]]:
        """Создает пустую структуру сущностей"""
        return {
            "companies": [],
            "tickers": [],
            "countries": [],
            "sectors": [],
            "currencies": [],
            "people": [],
            "sentiment": "neutral",
            "sentiment_score": 0.0
        }

    def _parse_news_content(self, content):
        """
        Парсит содержимое новости и извлекает структурированные данные + сущности
        """
        try:
            if isinstance(content, dict):
                parsed_data = {
                    "title": content.get("title", ""),
                    "summary": content.get("lead", content.get("content", ""))[:200] + "..."
                    if content.get("lead") or content.get("content")
                    else "Нет содержимого",
                    "key_points": content.get("key_points", content.get("bullets", [])),
                    "impact": "neutral",
                    "sources": content.get("sources", [])
                }

                # Добавляем сущности если они есть в контенте
                if "entities" in content:
                    parsed_data["entities"] = content["entities"]
                else:
                    # Извлекаем сущности из текста
                    text_for_entities = content.get("lead", "") + " " + content.get("content", "")
                    if text_for_entities.strip():
                        parsed_data["entities"] = self._extract_entities(text_for_entities)

                return parsed_data

            if isinstance(content, str):
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group())
                        return self._parse_news_content(parsed_json)
                    except:
                        pass

                summary = content[:200] + "..." if len(content) > 200 else content

                lines = content.split('\n')
                key_points = []
                for line in lines:
                    line = line.strip()
                    if line.startswith(('•', '-', '*')) and len(line) > 2:
                        key_points.append(line[1:].strip())

                if not key_points:
                    sentences = re.split(r'[.!?]+', content)
                    key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:5]

                parsed_data = {
                    "title": "",
                    "summary": summary,
                    "key_points": key_points,
                    "impact": "neutral",
                    "sources": []
                }

                # Добавляем извлечение сущностей для текста
                if content.strip():
                    parsed_data["entities"] = self._extract_entities(content)

                return parsed_data

            return {
                "title": "",
                "summary": "Не удалось проанализировать содержимое",
                "key_points": [],
                "impact": "unknown",
                "sources": [],
                "entities": self._create_empty_entities()
            }

        except Exception:
            return {
                "title": "Анализ новости",
                "summary": "Ошибка анализа",
                "key_points": ["Не удалось проанализировать содержимое"],
                "impact": "unknown",
                "sources": [],
                "entities": self._create_empty_entities()
            }


class NewsProcessor:
    """
    Обработчик готовых новостных текстов для создания черновиков RADAR
    """

    def __init__(self, openrouter_client: OpenRouterClient):
        self.client = openrouter_client

    def create_radar_draft_from_text(self, news_text: str, topic: str = "") -> Dict[str, Any]:
        """
        Создает черновик RADAR с гарантированным извлечением сущностей через отдельный запрос
        """
        try:
            # Основной запрос для анализа содержания
            prompt = self._create_analysis_prompt(news_text, topic)
            messages = [{"role": "user", "content": prompt}]
            response = self.client._make_request(messages, max_tokens=2500)

            if response and response.get("success"):
                result = self._process_analysis_response(response["content"], news_text)

                # ОТДЕЛЬНЫЙ ЗАПРОС ДЛЯ СУЩНОСТЕЙ - гарантия качества
                entities = self.client._extract_entities(news_text)
                result["entities"] = entities

                # Обновляем parsed_content
                if "parsed_content" in result:
                    result["parsed_content"]["entities"] = entities

                return result
            else:
                return self._create_fallback_draft(news_text, topic)

        except Exception:
            return self._create_fallback_draft(news_text, topic)

    def _create_analysis_prompt(self, news_text: str, topic: str) -> str:
        """
        Создает промпт для анализа с расширенными метриками
        """
        # Предварительный расчет статистик текста
        orig_words = len(news_text.split())
        orig_chars = len(news_text)

        return f"""Ты - финансовый аналитик сервиса RADAR. Проанализируй новость и создай детализированный черновик.

ТЕМА: {topic if topic else 'Не указана'}

ТЕКСТ НОВОСТИ:
{news_text}

СОЗДАЙ ЧЕРНОВИК В ФОРМАТЕ JSON:

{{
    "title": "Заголовок",
    "lead": "Краткое описание 1-2 предложения", 
    "key_points": ["факт1", "факт2", "факт3"],
    "impact_analysis": "Анализ влияния на рынки",
    "sentiment_score": 0.85,

    "hottness": {{
        "score": 0.8,
        "reasoning": "Почему новость горячая"
    }},

    "why_now": "1-2 фразы о важности и новизне",

    "timeline": {{
        "processing_start": "{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "processing_end": "будет заполнено автоматически"
    }},

    "statistics": {{
        "orig_words": {orig_words},
        "orig_chars": {orig_chars},
        "summ_words": 0,
        "summ_chars": 0, 
        "ratio_words": 0.0,
        "ratio_chars": 0.0,
        "reduction_percent": 0.0,
        "quality_score": 0.0,
        "words_per_sentence": 0.0,
        "density_score": 0.0,
        "unique_words_ratio": 0.0,
        "info_density": 0.0,
        "readability_score": 0.0
    }},

    "sources": ["Источник"],
    "confidence": 0.95,

    "entities": {{
        "companies": ["Сбербанк", "Газпром"],
        "tickers": ["SBER", "GAZP"],
        "countries": ["Россия"], 
        "sectors": ["финансы", "нефтегазовый"],
        "currencies": ["RUB"],
        "people": []
    }}
}}

ДЕТАЛЬНЫЕ ТРЕБОВАНИЯ:

1. HOTTNESS - насколько новость горячая:
   - score: 0.0-1.0 (1.0 - самая горячая)
   - reasoning: объяснение почему новость важна для широкой аудитории

2. WHY_NOW - актуальность:
   - Новизна информации
   - Подтверждения или опровержения
   - Масштаб затронутых активов/рынков
   - Формат: 1-2 короткие фразы

3. TIMELINE - хронология:
   - processing_start: время начала обработки (уже заполнено)
   - processing_end: НЕ заполняй - будет добавлено автоматически

4. СЕНТИМЕНТ - ТОЛЬКО ЧИСЛО:
   - sentiment_score: числовая оценка от -1.0 до 1.0
   - НЕ включай текстовое поле "sentiment"
   - -1.0 до -0.7: сильно негативный
   - -0.7 до -0.3: умеренно негативный  
   - -0.3 до 0.3: нейтральный
   - 0.3 до 0.7: умеренно позитивный
   - 0.7 до 1.0: сильно позитивный

5. СУЩНОСТИ - как ранее:
   - companies: все упомянутые компании
   - tickers: биржевые тикеры для каждой компании
   - countries: все упомянутые страны
   - sectors: отрасли экономики
   - currencies: валюты
   - people: упомянутые люди

ПРИМЕР HOTTNESS:
- Новость о ключевой ставке ЦБ → score: 0.9
- Отчетность крупной компании → score: 0.7
- Техническая новость → score: 0.3

ОТВЕТ ТОЛЬКО В JSON ФОРМАТЕ БЕЗ ЛЮБЫХ ДОПОЛНИТЕЛЬНЫХ КОММЕНТАРИЕВ"""

    def _calculate_text_statistics(self, original_text: str, summary_text: str) -> Dict[str, float]:
        """
        Расчет подробных статистик текста
        """
        import re
        from collections import Counter

        # Базовые метрики
        orig_words = len(original_text.split())
        orig_chars = len(original_text)
        summ_words = len(summary_text.split())
        summ_chars = len(summary_text)

        # Коэффициенты сжатия
        ratio_words = summ_words / orig_words if orig_words > 0 else 0
        ratio_chars = summ_chars / orig_chars if orig_chars > 0 else 0
        reduction_percent = (1 - ratio_words) * 100

        # Анализ качества
        orig_sentences = len(re.split(r'[.!?]+', original_text))
        summ_sentences = len(re.split(r'[.!?]+', summary_text))

        words_per_sentence = summ_words / summ_sentences if summ_sentences > 0 else 0

        # Плотность информации (уникальные слова)
        orig_word_count = Counter(original_text.lower().split())
        summ_word_count = Counter(summary_text.lower().split())

        unique_words_ratio = len(summ_word_count) / summ_words if summ_words > 0 else 0

        # Сложные метрики
        density_score = (summ_words / orig_words) * (
                    len(set(summary_text.split())) / summ_words) if summ_words > 0 else 0
        info_density = summ_words / orig_words if orig_words > 0 else 0

        # Оценка читаемости (упрощенная)
        avg_word_length = sum(len(word) for word in summary_text.split()) / summ_words if summ_words > 0 else 0
        readability_score = max(0, min(1, (20 - words_per_sentence) / 15 * (10 - avg_word_length) / 8))

        # Общая оценка качества
        quality_score = min(1.0, (
                readability_score * 0.3 +
                min(1.0, unique_words_ratio * 2) * 0.3 +
                min(1.0, 1 - abs(0.2 - ratio_words)) * 0.4  # Идеальное сжатие ~20%
        ))

        return {
            "orig_words": orig_words,
            "orig_chars": orig_chars,
            "summ_words": summ_words,
            "summ_chars": summ_chars,
            "ratio_words": round(ratio_words, 3),
            "ratio_chars": round(ratio_chars, 3),
            "reduction_percent": round(reduction_percent, 1),
            "quality_score": round(quality_score, 3),
            "words_per_sentence": round(words_per_sentence, 1),
            "density_score": round(density_score, 3),
            "unique_words_ratio": round(unique_words_ratio, 3),
            "info_density": round(info_density, 3),
            "readability_score": round(readability_score, 3)
        }

    def _process_analysis_response(self, ai_response: str, original_text: str) -> Dict[str, Any]:
        """
        Обрабатывает ответ AI с расчетом статистик
        """
        try:
            cleaned_content = ai_response.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            result = json.loads(cleaned_content)

            # РАСЧЕТ СТАТИСТИК
            summary_text = result.get("lead", "") + " " + " ".join(result.get("key_points", []))
            stats = self._calculate_text_statistics(original_text, summary_text)
            result["statistics"] = stats

            # ОБНОВЛЕНИЕ TIMELINE
            if "timeline" in result:
                result["timeline"]["processing_end"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Добавляем мета-информацию
            result["original_text_preview"] = original_text[:100] + "..." if len(original_text) > 100 else original_text
            result["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            result["parsed_content"] = self.client._parse_news_content(result)

            return result

        except json.JSONDecodeError:
            return self._create_fallback_draft(original_text, topic="Анализ текста")
        except Exception:
            return self._create_fallback_draft(original_text, topic="Анализ текста")

    def _create_fallback_draft(self, text: str, topic: str) -> Dict[str, Any]:
        """
        Создает резервный черновик с извлечением сущностей через AI
        """
        entities = self.client._extract_entities(text)

        # Расчет статистик для fallback
        summary_text = text[:150] + "..." if len(text) > 150 else text
        stats = self._calculate_text_statistics(text, summary_text)

        return {
            "title": f"Анализ: {topic}" if topic else "Анализ новости",
            "lead": summary_text,
            "key_points": ["Не удалось выполнить глубокий анализ"],
            "impact_analysis": "Требуется дополнительный анализ",
            "sentiment_score": entities.get("sentiment_score", 0.0),
            "hottness": {
                "score": 0.3,
                "reasoning": "Ограниченный анализ из-за ошибки обработки"
            },
            "why_now": "Требуется дополнительный анализ для определения актуальности",
            "timeline": {
                "processing_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_end": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": stats,
            "sources": ["Исходный текст"],
            "confidence": 0.0,
            "entities": entities,
            "original_text_preview": text[:100] + "..." if len(text) > 100 else text,
            "parsed_content": {
                "title": f"Анализ: {topic}" if topic else "Анализ новости",
                "summary": summary_text,
                "key_points": ["Не удалось выполнить глубокий анализ"],
                "impact": "unknown",
                "sources": ["Исходный текст"],
                "entities": entities
            }
        }


def convert_to_target_format(ai_result: Dict[str, Any], original_topic: str = "", original_text: str = "") -> Dict[str, Any]:
    """
    Конвертирует результат AI-анализа в целевой формат RADAR
    """
    # Извлекаем основные компоненты из существующего результата
    draft_data = ai_result.get("draft", {})
    if not draft_data:
        # Если структура другая, пытаемся извлечь из корневых полей
        draft_data = {
            "title": ai_result.get("title", ""),
            "lead": ai_result.get("lead", ai_result.get("summary", "")),
            "points": ai_result.get("key_points", ai_result.get("points", []))
        }

    # Извлекаем entities (может быть в разных местах)
    entities = ai_result.get("entities", {})
    if not entities and "parsed_content" in ai_result:
        entities = ai_result["parsed_content"].get("entities", {})

    # Извлекаем sentiment (может быть под разными именами)
    sentiment = ai_result.get("sentiment_score",
                              ai_result.get("sentiment",
                                            ai_result.get("hottness", {}).get("sentiment_score", 0.0)))

    # Извлекаем hotness (может быть под разными именами)
    hotness_data = ai_result.get("hottness", {})
    if not hotness_data:
        hotness_data = ai_result.get("hotness", {})

    hotness_score = hotness_data.get("score",
                                     ai_result.get("hotness_score",
                                                   ai_result.get("hottness_score", 0.5)))

    # Извлекаем why_hot и why_now
    why_hot = ai_result.get("why_hot",
                            hotness_data.get("reasoning",
                                             ai_result.get("reasoning", "")))

    why_now = ai_result.get("why_now", "")

    # Извлекаем impact_analysis
    impact_analysis = ai_result.get("impact_analysis",
                                    ai_result.get("impact",
                                                  "Требуется дополнительный анализ"))

    # Обрабатываем statistics
    stats = ai_result.get("statistics", {})
    target_stats = {
        "compression": stats.get("compression", stats.get("ratio_chars", 0.5)),
        "quality": stats.get("quality", stats.get("quality_score", 1.0)),
        "processing_started": stats.get("processing_started", datetime.now()),
        "processing_ended": stats.get("processing_ended", datetime.now())
    }

    # Обрабатываем entities для гарантии наличия всех полей
    target_entities = {
        "companies": entities.get("companies", []),
        "tickers": entities.get("tickers", []),
        "countries": entities.get("countries", []),
        "sectors": entities.get("sectors", []),
        "currencies": entities.get("currencies", [])
    }

    # Извлекаем links/sources
    links = ai_result.get("links",
                          ai_result.get("sources",
                                        ai_result.get("parsed_content", {}).get("sources", [])))

    # Определяем datetime
    news_datetime = ai_result.get("datetime",
                                  ai_result.get("processed_at",
                                                ai_result.get("timeline", {}).get("processing_start",
                                                                                  datetime.now())))

    # Если datetime строка, конвертируем в datetime объект
    if isinstance(news_datetime, str):
        try:
            news_datetime = datetime.fromisoformat(news_datetime.replace('Z', '+00:00'))
        except:
            news_datetime = datetime.now()

    # Определяем is_short (менее 500 символов) - ИСПРАВЛЕННАЯ ВЕРСИЯ
    text_for_length_check = original_text or ai_result.get("original_text", "") or ""
    is_short = len(text_for_length_check) < 500 if text_for_length_check else False

    # Собираем финальный результат в ЦЕЛЕВОМ формате
    target_format = {
        "topic": original_topic or ai_result.get("topic", "Анализ новости"),
        "draft": {
            "title": draft_data.get("title", "Заголовок не определен"),
            "lead": draft_data.get("lead", "Описание не предоставлено"),
            "points": draft_data.get("points", draft_data.get("key_points", []))
        },
        "sentiment": float(sentiment),
        "hotness": float(hotness_score),
        "why_hot": why_hot,
        "why_now": why_now,
        "impact_analysis": impact_analysis,
        "statistics": target_stats,
        "entities": target_entities,
        "datetime": news_datetime,
        "links": links,
        "is_short": is_short  # ← НОВОЕ ПОЛЕ
    }
    return target_format


def convert_news_processor_result(processor_result: Dict[str, Any], topic: str = "") -> Dict[str, Any]:
    """
    Специализированная конвертация для результата из вашего NewsProcessor
    """
    return convert_to_target_format(processor_result, topic)


def process_text_to_target_format(text: str, topic: str = "") -> Dict[str, Any]:
    """
    Быстрая обработка текста с immediate конвертацией в целевой формат
    """
    client = OpenRouterClient()
    processor = NewsProcessor(client)
    result = processor.create_radar_draft_from_text(text, topic)
    return convert_to_target_format(result, topic)


def main_with_conversion():
    """Демонстрация работы с конвертацией"""
    try:
        client = OpenRouterClient()
        processor = NewsProcessor(client)
        print("Процессор новостей инициализирован!\n")
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return

    sample_text = """Сегодня на Московской бирже акции Сбербанка показали рост на 2.3%, достигнув отметки в 285 рублей за бумагу. 
    Аналитики связывают это с публикацией положительной отчетности банка за второй квартал. 
    Выручка увеличилась на 15% по сравнению с аналогичным периодом прошлого года."""

    # Получаем результат от вашего существующего процессора
    original_result = processor.create_radar_draft_from_text(sample_text, "Котировки акций")

    # Конвертируем в нужный формат
    final_result = convert_to_target_format(original_result, "Котировки акций")

    # Теперь можно сохранять в MongoDB
    print("РЕЗУЛЬТАТ В ЦЕЛЕВОМ ФОРМАТЕ:")
    print(json.dumps(final_result, indent=2, default=str))

    # Для вставки в MongoDB:
    # collection.insert_one(final_result)


def _print_radar_result(result):
    """Выводит результат RADAR с расширенными метриками"""
    print("ЗАГОЛОВОК:", result.get("title", "Нет заголовка"))
    print("ЛИД:", result.get("lead", "Нет лида"))

    # СЕНТИМЕНТ
    sentiment_score = result.get("sentiment_score", 0)
    print(f"СЕНТИМЕНТ: {sentiment_score}")

    # HOTTNESS
    hottness = result.get("hottness", {})
    if hottness:
        score = hottness.get("score", 0)
        print(f"ГОРЯЧЕСТЬ: {score}")
        if hottness.get("reasoning"):
            print(f"Причина: {hottness['reasoning']}")

    # WHY_NOW
    why_now = result.get("why_now", "")
    if why_now:
        print(f"ПОЧЕМУ СЕЙЧАС: {why_now}")

    # TIMELINE
    timeline = result.get("timeline", {})
    if timeline:
        print("ТАЙМЛАЙН:")
        if timeline.get("processing_start"):
            print(f" Обработка начата: {timeline['processing_start']}")
        if timeline.get("processing_end"):
            print(f" Обработка завершена: {timeline['processing_end']}")

    print("АНАЛИЗ ВЛИЯНИЯ:", result.get("impact_analysis", "Нет анализа"))

    # СТАТИСТИКИ
    stats = result.get("statistics", {})
    if stats:
        print("СТАТИСТИКИ ТЕКСТА:")
        print(f"   Исходный текст: {stats.get('orig_words', 0)} слов, {stats.get('orig_chars', 0)} символов")
        print(f"   Сводка: {stats.get('summ_words', 0)} слов, {stats.get('summ_chars', 0)} символов")
        print(f"   Сжатие: {stats.get('reduction_percent', 0)}%")
        print(f"   Качество: {stats.get('quality_score', 0)}")
        print(f"   Плотность инфо: {stats.get('info_density', 0)}")

    # КЛЮЧЕВЫЕ ПУНКТЫ
    key_points = result.get("key_points", [])
    if key_points:
        print("КЛЮЧЕВЫЕ ПУНКТЫ:")
        for i, point in enumerate(key_points, 1):
            print(f"   {i}. {point}")

    # СУЩНОСТИ
    entities = result.get("entities", {})
    if entities:
        print("СУЩНОСТИ:")
        if entities.get("companies"):
            print(f"   Компании: {', '.join(entities['companies'])}")
        if entities.get("tickers"):
            print(f"   Тикеры: {', '.join(entities['tickers'])}")
        if entities.get("countries"):
            print(f"   Страны: {', '.join(entities['countries'])}")
        if entities.get("sectors"):
            print(f"   Сектора: {', '.join(entities['sectors'])}")
        if entities.get("currencies"):
            print(f"   Валюты: {', '.join(entities['currencies'])}")
        if entities.get("people"):
            print(f"   Персоны: {', '.join(entities['people'])}")

    sources = result.get("sources", [])
    if sources:
        print("ИСТОЧНИКИ:", ", ".join(sources))


def main():
    """Демонстрация обработки готовых текстов"""
    try:
        client = OpenRouterClient()
        processor = NewsProcessor(client)
        print("Процессор новостей инициализирован!\n")
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return

    sample_texts = [
        {
            "topic": "Котировки акций",
            "text": """Сегодня на Московской бирже акции Сбербанка показали рост на 2.3%, достигнув отметки в 285 рублей за бумагу. Аналитики связывают это с публикацией положительной отчетности банка за второй квартал. Выручка увеличилась на 15% по сравнению с аналогичным периодом прошлого года. Одновременно акции Газпрома снизились на 1.1%."""
        }
    ]

    print("РЕЗУЛЬТАТ АНАЛИЗА НОВОСТЕЙ")
    print("=" * 60)

    for sample in sample_texts:
        print(f"\nТЕМА: {sample['topic']}")
        print(f"ТЕКСТ: {sample['text'][:100]}...")

        result = processor.create_radar_draft_from_text(sample['text'], sample['topic'])

        if result:
            _print_radar_result(result)
        else:
            print("Не удалось создать черновик")

        print("-" * 50)


def process_single_text(text: str, topic: str = "") -> Dict[str, Any]:
    """
    Быстрая обработка одного текста
    """
    client = OpenRouterClient()
    processor = NewsProcessor(client)
    return processor.create_radar_draft_from_text(text, topic)


def process_single_text_to_target_format(text: str, topic: str = "") -> Dict[str, Any]:
    """
    Быстрая обработка одного текста с конвертацией в целевой формат
    """
    client = OpenRouterClient()
    processor = NewsProcessor(client)
    result = processor.create_radar_draft_from_text(text, topic)
    return convert_to_target_format(result, topic, text)


def demo_full_pipeline():
    """
    Демонстрация полного пайплайна обработки
    """
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ ПОЛНОГО ПАЙПЛАЙНА RADAR")
    print("=" * 70)

    # Пример новости
    sample_news = """
    Компания Apple представила новые модели iPhone 16 с революционной системой искусственного интеллекта. 
    Акции компании выросли на 5% в ходе торгов на NASDAQ. Аналитики ожидают дальнейшего роста котировок 
    в связи с высоким спросом на новые устройства. Одновременно конкуренты Samsung и Xiaomi анонсировали 
    ответные продукты, что может привести к обострению конкуренции на рынке смартфонов.
    """

    print("\n1. ОБРАБОТКА ЧЕРЕЗ NewsProcessor:")
    print("-" * 40)

    # Обработка через оригинальный процессор
    original_result = process_single_text(sample_news, "Технологии")
    if original_result:
        _print_radar_result(original_result)

    print("\n2. КОНВЕРТАЦИЯ В ЦЕЛЕВОЙ ФОРМАТ:")
    print("-" * 40)

    # Конвертация в целевой формат
    target_result = convert_to_target_format(original_result, "Технологии")
    print("Структура целевого формата:")
    for key, value in target_result.items():
        if key == "draft":
            print(f"  draft:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value[:50]}..." if isinstance(sub_value, str) and len(
                    sub_value) > 50 else f"    {sub_key}: {sub_value}")
        elif key == "statistics":
            print(f"  statistics: [данные статистики]")
        elif key == "entities":
            print(f"  entities: [извлеченные сущности]")
        else:
            print(f"  {key}: {value}")

    print("\n3. JSON ДЛЯ MONGODB:")
    print("-" * 40)
    print(json.dumps(target_result, indent=2, default=str, ensure_ascii=False))

    return target_result


def save_to_mongodb_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Подготавливает данные для сохранения в MongoDB
    """
    # Конвертируем datetime в строку для JSON сериализации
    mongodb_data = data.copy()

    if "datetime" in mongodb_data and isinstance(mongodb_data["datetime"], datetime):
        mongodb_data["datetime"] = mongodb_data["datetime"].isoformat()

    if "statistics" in mongodb_data:
        stats = mongodb_data["statistics"]
        if "processing_started" in stats and isinstance(stats["processing_started"], datetime):
            stats["processing_started"] = stats["processing_started"].isoformat()
        if "processing_ended" in stats and isinstance(stats["processing_ended"], datetime):
            stats["processing_ended"] = stats["processing_ended"].isoformat()

    return mongodb_data


def batch_process_texts(texts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Пакетная обработка нескольких текстов
    """
    client = OpenRouterClient()
    processor = NewsProcessor(client)
    results = []

    for i, item in enumerate(texts, 1):
        print(f"Обработка текста {i}/{len(texts)}...")
        try:
            text = item.get("text", "")
            topic = item.get("topic", "")

            result = processor.create_radar_draft_from_text(text, topic)
            target_format = convert_to_target_format(result, topic)
            results.append(target_format)

            print(f"✓ Текст {i} обработан успешно")

        except Exception as e:
            print(f"✗ Ошибка обработки текста {i}: {e}")
            # Создаем fallback результат
            fallback_result = {
                "topic": topic,
                "draft": {
                    "title": f"Ошибка обработки: {topic}",
                    "lead": "Не удалось обработать текст",
                    "points": ["Ошибка в процессе анализа"]
                },
                "sentiment": 0.0,
                "hotness": 0.1,
                "why_hot": "Ошибка обработки",
                "why_now": "Не определено",
                "impact_analysis": "Требуется повторная обработка",
                "statistics": {
                    "compression": 0.1,
                    "quality": 0.1,
                    "processing_started": datetime.now(),
                    "processing_ended": datetime.now()
                },
                "entities": {
                    "companies": [],
                    "tickers": [],
                    "countries": [],
                    "sectors": [],
                    "currencies": []
                },
                "datetime": datetime.now(),
                "links": []
            }
            results.append(fallback_result)

    return results


if __name__ == "__main__":
    # Запуск демонстрации
    print("Запуск RADAR News Processor...")

    # Демонстрация полного пайплайна
    final_result = demo_full_pipeline()

    print("\n" + "=" * 70)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ ДЛЯ MONGODB:")
    print("=" * 70)

    # Пример для MongoDB
    mongodb_data = save_to_mongodb_format(final_result)
    print("Данные готовы для вставки в MongoDB:")
    print(f"collection.insert_one({json.dumps(mongodb_data, indent=2, default=str, ensure_ascii=False)})")

    print("\n" + "=" * 70)
    print("БЫСТРЫЕ ФУНКЦИИ ДОСТУПНЫ:")
    print("=" * 70)
    print("1. process_single_text(text, topic) - базовая обработка")
    print("2. process_single_text_to_target_format(text, topic) - обработка с конвертацией")
    print("3. batch_process_texts([texts]) - пакетная обработка")
    print("4. convert_to_target_format(ai_result, topic) - конвертация существующего результата")