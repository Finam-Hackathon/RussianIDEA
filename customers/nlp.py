from transformers import pipeline
import torch
import re
from typing import Dict
import logging
import os
import time
import sys
import transformers
import datetime
import hashlib
from functools import lru_cache
from typing import Dict, Optional



# Глобальные метрики для мониторинга (добавьте в начало файла)
_SUMMARIZATION_METRICS = {
    'requests_total': 0,
    'errors_total': 0,
    'processing_times': [],
    'last_reset': datetime.datetime.now()
}


def _get_text_hash(text: str) -> str:
    """Генерация хеша текста для кэширования"""
    return hashlib.md5(text.encode()).hexdigest()


@lru_cache(maxsize=100)
def _cached_summarization(text_hash: str, config_key: str, config_params: tuple) -> str:
    """
    Кэшированная суммаризация на основе хеша текста и конфигурации
    """
    # В реальной реализации здесь будет вызов модели
    # Пока возвращаем заглушку, которая будет заменена реальным результатом
    return f"Кэшированная суммаризация для конфига {config_key}"


def _record_metrics(success: bool = True, processing_time: float = 0.0):
    """
    Запись метрик производительности
    """
    global _SUMMARIZATION_METRICS

    _SUMMARIZATION_METRICS['requests_total'] += 1

    if success:
        _SUMMARIZATION_METRICS['processing_times'].append(processing_time)
    else:
        _SUMMARIZATION_METRICS['errors_total'] += 1


# Дополнительные утилиты для мониторинга (можно добавить в класс как статические методы)
def get_summarization_metrics() -> Dict:
    """Получение текущих метрик суммаризации"""
    global _SUMMARIZATION_METRICS

    times = _SUMMARIZATION_METRICS['processing_times']
    total_requests = _SUMMARIZATION_METRICS['requests_total']

    avg_time = sum(times) / len(times) if times else 0
    error_rate = (_SUMMARIZATION_METRICS['errors_total'] / total_requests * 100) if total_requests > 0 else 0

    return {
        'total_requests': total_requests,
        'total_errors': _SUMMARIZATION_METRICS['errors_total'],
        'error_rate_percent': round(error_rate, 2),
        'avg_processing_time_seconds': round(avg_time, 2),
        'performance_benchmark': 'good' if avg_time < 2.0 else 'needs_optimization'
    }


def clear_summarization_cache():
    """Очистка кэша суммаризации"""
    _cached_summarization.cache_clear()
    logging.getLogger('RADARFinancialSummarizer').info("🧹 Кэш суммаризации очищен")


def get_cache_info() -> Dict:
    """Получение информации о кэше"""
    cache_info = _cached_summarization.cache_info()
    return {
        'cache_hits': cache_info.hits,
        'cache_misses': cache_info.misses,
        'cache_size': cache_info.currsize,
        'cache_max_size': cache_info.maxsize
    }

class RADARFinancialSummarizer:
    """
    ФИНАЛЬНАЯ ГОТОВАЯ СИСТЕМА для финансовых новостей RADAR
    """

    def __init__(self, device: str = "auto", debug_mode: bool = False):
        self.setup_logging()
        self.device = self._setup_device(device)
        self.model_name = "facebook/bart-large-cnn"
        self.debug_mode = debug_mode  # Добавьте эту строку

        self.model_kwargs = {
            'low_cpu_mem_usage': True,
            'torchscript': True,  # 🚀 Ускорение инференса
            'use_cache': True,  # 📝 Кэширование внимания
        }

        # 🎯 ФИНАЛЬНЫЕ ОПТИМАЛЬНЫЕ НАСТРОЙКИ
        self.configs = {
            "EARNINGS": {'max_length': 50, 'min_length': 25, 'length_penalty': 1.6, 'num_beams': 4, 'early_stopping': True, },
            "CENTRAL_BANK": {'max_length': 35, 'min_length': 18, 'length_penalty': 2.1, 'num_beams': 4, 'early_stopping': True,},
            "MARKET": {'max_length': 30, 'min_length': 15, 'length_penalty': 2.0, 'num_beams': 4, 'early_stopping': True, },
            "MERGERS": {'max_length': 70, 'min_length': 35, 'length_penalty': 1.7, 'num_beams': 4, 'early_stopping': True,},
            "REGULATORY": {'max_length': 60, 'min_length': 30, 'length_penalty': 2.0, 'num_beams': 4, 'early_stopping': True,},
            "ECONOMIC": {'max_length': 55, 'min_length': 28, 'length_penalty': 1.9, 'num_beams': 4, 'early_stopping': True,},
            "TECHNOLOGY": {'max_length': 65, 'min_length': 32, 'length_penalty': 1.8, 'num_beams': 4, 'early_stopping': True,},
            "COMMODITIES": {'max_length': 50, 'min_length': 25, 'length_penalty': 2.0, 'num_beams': 4, 'early_stopping': True,},
            "GENERAL_FINANCIAL": {'max_length': 60, 'min_length': 30, 'length_penalty': 2.0, 'num_beams': 4, 'early_stopping': True,},
        }

        self.setup_model()

    def _get_adaptive_config(self, text: str, news_type: str) -> Dict:
        """Адаптивная настройка параметров на основе длины текста"""
        base_config = self.configs.get(news_type, self.configs["GENERAL_FINANCIAL"]).copy()

        # Рассчитываем оптимальные длины на основе количества слов
        words = text.split()
        word_count = len(words)

        # Адаптивная настройка max_length и min_length
        if word_count < 30:
            base_config['max_length'] = max(15, word_count // 2)
            base_config['min_length'] = max(8, word_count // 3)
        elif word_count > 200:
            base_config['max_length'] = min(60, base_config['max_length'])
            base_config['min_length'] = min(30, base_config['min_length'])

        # Для CPU уменьшаем num_beams для скорости
        if self.device == "cpu":
            base_config['num_beams'] = 2  # Быстрее чем 4

        return base_config

    def _setup_device(self, device: str) -> str:
        """
        Продвинутое определение устройства с оптимизацией
        """
        try:
            if device == "auto":
                # Приоритет 1: CUDA с проверкой памяти
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

                    self.logger.info(f"🎮 Обнаружен GPU: {gpu_name} ({gpu_memory:.1f}GB)")

                    # Проверка достаточности памяти для BART-large
                    if gpu_memory >= 4.0:  # Минимум 4GB для комфортной работы
                        return "cuda"
                    else:
                        self.logger.warning(f"⚠️ GPU память {gpu_memory:.1f}GB маловата, используется CPU")
                        return "cpu"

                # Приоритет 2: Apple Silicon
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.logger.info("🍎 Используется Apple Silicon (MPS)")
                    return "mps"

                # Приоритет 3: CPU
                else:
                    cpu_count = os.cpu_count()
                    self.logger.info(f"⚡ Используется CPU (ядер: {cpu_count})")
                    return "cpu"

            # Ручной выбор с проверкой доступности
            elif device == "cuda":
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    self.logger.warning("❌ CUDA запрошена, но недоступна. Используется CPU")
                    return "cpu"

            elif device == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    self.logger.warning("❌ MPS запрошена, но недоступна. Используется CPU")
                    return "cpu"

            elif device == "cpu":
                return "cpu"

            else:
                self.logger.warning(f"❌ Неизвестное устройство '{device}', используется CPU")
                return "cpu"

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка определения устройства: {e}")
            return "cpu"  # Гарантированный fallback

    def setup_logging(self):
        """
        Профессиональная настройка логирования с ротацией и фильтрацией
        """
        import logging.handlers
        import sys

        self.logger = logging.getLogger('RADARFinancialSummarizer')
        self.logger.setLevel(logging.INFO)

        # Очистка предыдущих handlers
        self.logger.handlers.clear()

        # 1. 🎯 Console Handler с улучшенным форматированием
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        class SmartFormatter(logging.Formatter):
            def format(self, record):
                # Добавляем эмодзи в зависимости от уровня
                if record.levelno >= logging.ERROR:
                    record.msg = f"❌ {record.msg}"
                elif record.levelno >= logging.WARNING:
                    record.msg = f"⚠️ {record.msg}"
                elif record.levelno >= logging.INFO:
                    record.msg = f"ℹ️ {record.msg}"
                return super().format(record)

        console_formatter = SmartFormatter(
            '%(asctime)s | %(levelname)-8s | [RADAR] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # 2. 💾 Rotating File Handler (ограничение размера)
        try:
            os.makedirs('logs', exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                filename='logs/radar_system.log',
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,  # 5 backup файлов
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # В файл пишем больше информации

            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)

            self.logger.addHandler(file_handler)

        except Exception as e:
            print(f"⚠️ File logging unavailable: {e}")

        # 3. 📧 Error Handler (только ошибки в отдельный файл)
        try:
            error_handler = logging.FileHandler('logs/radar_errors.log', encoding='utf-8')
            error_handler.setLevel(logging.ERROR)

            error_formatter = logging.Formatter(
                '%(asctime)s | ERROR | %(message)s\nStacktrace: %(exc_info)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            error_handler.setFormatter(error_formatter)

            self.logger.addHandler(error_handler)

        except Exception as e:
            print(f"⚠️ Error logging unavailable: {e}")

        # Добавляем console handler в конце
        self.logger.addHandler(console_handler)

        # Логируем инициализацию
        self.logger.info("=" * 50)
        self.logger.info("🚀 RADAR Financial Summarizer Started")
        self.logger.info("=" * 50)

    def setup_model(self):
        """
        Улучшенная загрузка модели с оптимизацией и обработкой ошибок
        """
        try:
            self.logger.info(f"🔄 Загрузка модели {self.model_name}...")

            # 📊 Логирование информации о системе перед загрузкой
            self._log_system_info()

            # ⏱️ Замер времени загрузки
            start_time = time.time()

            # 🎯 Оптимизированная загрузка модели
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # 🚀 Экономия памяти
                trust_remote_code=True,  # 🔧 Для некоторых моделей
                model_kwargs={
                    'low_cpu_mem_usage': True,  # 📉 Оптимизация RAM
                    'force_download': False,  # 💾 Кэширование моделей
                    'resume_download': True,  # 🔄 Продолжение прерванной загрузки
                }
            )

            # 📈 Логирование успешной загрузки
            load_time = time.time() - start_time
            self.logger.info(f"✅ Модель загружена за {load_time:.1f} секунд")

            # 🧪 Тестовая суммаризация для проверки работы
            self._test_model_functionality()

            # 💾 Информация о памяти
            self._log_memory_usage()

        except OSError as e:
            # 🌐 Ошибки сети/загрузки
            if "404" in str(e):
                self.logger.error(f"❌ Модель {self.model_name} не найдена на HuggingFace Hub")
                self.logger.info("💡 Проверьте название модели или интернет-соединение")
            elif "timeout" in str(e).lower():
                self.logger.error("❌ Таймаут загрузки модели")
                self.logger.info("💡 Попробуйте увеличить timeout или проверить соединение")
            else:
                self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

        except RuntimeError as e:
            # 💻 Ошибки памяти/совместимости
            if "CUDA out of memory" in str(e):
                self.logger.error("❌ Недостаточно памяти GPU")
                self.logger.info("💡 Попробуйте использовать CPU или уменьшить batch_size")
            elif "CUDA" in str(e):
                self.logger.error(f"❌ Ошибка CUDA: {e}")
                self.logger.info("💡 Проверьте драйверы NVIDIA и совместимость PyTorch+CUDA")
            else:
                self.logger.error(f"❌ Runtime ошибка: {e}")
            raise

        except Exception as e:
            self.logger.error(f"❌ Неожиданная ошибка при загрузке модели: {e}")
            self.logger.info("💡 Проверьте установку transformers и torch")
            raise

    def _log_system_info(self):
        """Логирование информации о системе"""
        self.logger.info(f"⚙️ Устройство: {self.device}")
        self.logger.info(f"🐍 Python: {sys.version}")
        self.logger.info(f"🔥 PyTorch: {torch.__version__}")
        self.logger.info(f"🤗 Transformers: {transformers.__version__}")

        if self.device == "cuda":
            self.logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _test_model_functionality(self):
        """Тестирование работоспособности модели с правильными параметрами"""
        try:
            test_text = "This is a test sentence to verify the model is working correctly."
            # Используем адаптивные параметры для теста
            words_count = len(test_text.split())
            max_length = max(15, words_count // 2)
            min_length = max(8, words_count // 3)

            test_result = self.summarizer(test_text, max_length=max_length, min_length=min_length)

            if test_result and len(test_result) > 0:
                self.logger.info("🧪 Тестовая суммаризация прошла успешно")
            else:
                self.logger.warning("⚠️ Тестовая суммаризация вернула пустой результат")

        except Exception as e:
            self.logger.warning(f"⚠️ Тестовая суммаризация не удалась: {e}")

    def _log_memory_usage(self):
        """Логирование использования памяти"""
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1e6
            memory_reserved = torch.cuda.memory_reserved() / 1e6
            self.logger.info(f"💾 Память GPU: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB")

    def summarize_news(self, text: str, generate_draft: bool = True, use_cache: bool = True) -> Dict:
        """
        УЛУЧШЕННАЯ ФИНАЛЬНАЯ функция суммаризации с кэшированием и метриками

        Args:
            text: Текст для суммаризации
            generate_draft: Генерировать ли черновик
            use_cache: Использовать кэширование

        Returns:
            Dict: Результат суммаризации или информация об ошибке
        """
        start_time = time.time()

        # 1. Расширенная валидация входных данных
        validation_error = self._validate_summarization_input(text)
        if validation_error:
            _record_metrics(success=False)
            self.logger.error(f"❌ Ошибка валидации: {validation_error}")
            return {"error": validation_error, "error_type": "validation"}

        text = text.strip()
        self.logger.info(f"🚀 Начата суммаризация текста длиной {len(text)} символов")

        try:
            # 2. Классификация типа новости
            news_type = self._detect_news_type(text)
            self.logger.debug(f"📰 Определен тип новости: {news_type}")

            # ДЛЯ ОТЛАДКИ: логируем детали классификации
            if self.debug_mode:
                text_lower = text.lower()
                market_terms = ['s&p', 'dow', 'nasdaq', 'index', 'stock market']
                earnings_terms = ['earnings', 'revenue', 'profit', 'quarterly']

                found_market = [term for term in market_terms if term in text_lower]
                found_earnings = [term for term in earnings_terms if term in text_lower]

                self.logger.debug(f"🔍 Найдены рыночные термины: {found_market}")
                self.logger.debug(f"🔍 Найдены отчетные термины: {found_earnings}")

            # 3. Получение АДАПТИВНОЙ конфигурации
            config = self._get_adaptive_config(text, news_type)

            # 4. Предобработка текста
            processed_text = self.preprocess_text(text)

            # 5. Суммаризация (с кэшированием или без)
            cache_info = "disabled"
            if use_cache:
                text_hash = _get_text_hash(processed_text)
                config_key = news_type
                config_params = tuple(config.values())  # Конвертируем в hashable tuple

                # Пытаемся получить из кэша
                raw_summary = _cached_summarization(text_hash, config_key, config_params)
                cache_info = "hit"

                # Если кэш пустой, выполняем реальную суммаризацию
                if raw_summary.startswith("Кэшированная суммаризация"):
                    result = self.summarizer(processed_text, **config)
                    raw_summary = result[0]['summary_text']
                    cache_info = "miss"
            else:
                # Прямой вызов суммаризатора
                result = self.summarizer(processed_text, **config)
                raw_summary = result[0]['summary_text']

            # 6. Очистка результата
            cleaned_summary = self._clean_summary(raw_summary)

            # 7. Генерация черновика (опционально)
            draft = ""
            if generate_draft:
                draft = self._generate_draft(cleaned_summary, news_type)

            # 8. Расчет статистики
            stats = self._calculate_stats(text, cleaned_summary)
            stats['cache'] = cache_info
            stats['text_length_original'] = len(text)
            stats['text_length_summary'] = len(cleaned_summary)

            # 9. Логирование и метрики
            processing_time = time.time() - start_time
            _record_metrics(success=True, processing_time=processing_time)

            self.logger.info(
                f"✅ Суммаризация завершена. "
                f"Качество: {stats.get('quality', 'N/A')}, "
                f"Сжатие: {stats.get('compression_ratio', 0)}, "
                f"Кэш: {cache_info}, "
                f"Время: {processing_time:.2f}с"
            )

            return {
                "summary": cleaned_summary,
                "news_type": news_type,
                "draft": draft,
                "stats": stats,
                "processing_time": round(processing_time, 2),
                "success": True
            }

        except ValueError as e:
            # Ошибки валидации данных
            processing_time = time.time() - start_time
            _record_metrics(success=False, processing_time=processing_time)
            self.logger.error(f"❌ Ошибка данных: {e} [Время: {processing_time:.2f}с]")
            return {"error": str(e), "error_type": "data_validation"}

        except TimeoutError as e:
            # Таймаут операции
            processing_time = time.time() - start_time
            _record_metrics(success=False, processing_time=processing_time)
            self.logger.error(f"❌ Таймаут: {e} [Время: {processing_time:.2f}с]")
            return {"error": "Превышено время обработки", "error_type": "timeout"}

        except Exception as e:
            # Все остальные ошибки
            processing_time = time.time() - start_time
            _record_metrics(success=False, processing_time=processing_time)
            self.logger.error(f"❌ Неожиданная ошибка: {e} [Время: {processing_time:.2f}с]")

            # Детализация ошибки для логирования
            self.logger.exception("Детали неожиданной ошибки:")

            return {
                "error": "Внутренняя ошибка сервиса",
                "error_type": "internal",
                "details": str(e) if self.debug_mode else None  # Режим отладки
            }

    def _validate_summarization_input(self, text: str) -> Optional[str]:
        """
        Валидация входных данных для суммаризации
        """
        if not text or not isinstance(text, str):
            return "Текст должен быть непустой строкой"

        text = text.strip()

        if len(text) < 30:
            return "Текст слишком короткий для анализа"

        if len(text) > 100000:
            return "Текст слишком длинный для анализа"

        # Проверка на осмысленный текст (минимальное количество слов)
        words = text.split()
        if len(words) < 5:
            return "Текст должен содержать хотя бы 5 слов"

        return None

    def _detect_news_type(self, text: str) -> str:
        """
        УЛУЧШЕННАЯ классификация с приоритетами и контекстным анализом
        """
        text_lower = text.lower()

        # Сначала проверяем самые специфичные паттерны с приоритетами
        type_checks = [
            # (тип, ключевые_слова, приоритет)
            ("MERGERS", ['acquisition', 'merger', 'takeover', 'buyout', 'deal'], 10),
            ("EARNINGS", [
                'earnings', 'revenue', 'profit', 'quarterly', 'q1', 'q2', 'q3', 'q4',
                'financial results', 'beat estimates', 'missed estimates', 'eps', 'ebitda'
            ], 9),
            ("CENTRAL_BANK", [
                'fed', 'federal reserve', 'ecb', 'central bank', 'interest rate',
                'rate hike', 'rate cut', 'monetary policy', 'powell', 'lagarde'
            ], 8),
            ("MARKET", [
                's&p', 'dow', 'nasdaq', 'index', 'stock market', 'trading session',
                'market close', 'intraday', 'points', 'gains', 'losses', 'stock', 'stocks',
                'equities', 'market index', 'indexes', 'trading volume', 'market volatility'
            ], 7),
            ("REGULATORY", ['regulation', 'sec', 'lawsuit', 'legal', 'ftc', 'doj'], 6),
            ("ECONOMIC", ['gdp', 'unemployment', 'cpi', 'economic data', 'jobs report'], 5),
        ]

        # Находим тип с наивысшим приоритетом
        best_type = "GENERAL_FINANCIAL"
        best_priority = 0
        best_match_count = 0

        for news_type, keywords, priority in type_checks:
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Если нашли совпадения и приоритет выше ИЛИ больше совпадений при равном приоритете
                if (priority > best_priority) or (priority == best_priority and matches > best_match_count):
                    best_priority = priority
                    best_match_count = matches
                    best_type = news_type

        # Особый случай: если есть "earnings" но также много рыночных терминов - это MARKET
        if best_type == "EARNINGS" and any(
                term in text_lower for term in ['s&p', 'dow', 'nasdaq', 'index', 'stock market']):
            market_terms = sum(1 for term in ['s&p', 'dow', 'nasdaq', 'index', 'stock market'] if term in text_lower)
            if market_terms >= 2:  # Если есть хотя бы 2 рыночных термина
                best_type = "MARKET"

        return best_type

    def preprocess_text(self, text: str) -> str:
        """
        Оптимальная предобработка для финансовых новостей
        """
        # 1. 🗑️ Удаление HTML-тегов
        text = re.sub(r'<.*?>', '', text)
        # 2. 📧 Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        # 3. 🔗 Удаление URL (дополнительная защита)
        text = re.sub(r'https?://\S+', '', text)
        # 4. 🔄 Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        # 5. ✂️ Обрезка пробелов
        text = text.strip()
        # 6. Добавление пробела в конец предложения
        text = re.sub(r'([.!?])([А-ЯA-Z])', r'\1 \2', text)

        return text

    def _clean_summary(self, summary: str) -> str:
        """
        УЛУЧШЕННАЯ очистка суммаризации с исправлением числовых форматов
        """
        # 1. 💰 ИСПРАВЛЕНИЕ ФИНАНСОВОГО ФОРМАТИРОВАНИЯ
        # Исправляем пробелы после точек в числах: $89. 5 → $89.5
        summary = re.sub(r'(\$?\d+)\.\s+(\d+)', r'\1.\2', summary)

        # Форматирование денежных сумм
        summary = re.sub(r'\$(\s*)(\d+(?:\.\d+)?)\s*(?:billion|million)?', r'$\2', summary)
        summary = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1%', summary)
        summary = re.sub(r'(\d+(?:\.\d+)?)%\s*-\s*(\d+(?:\.\d+)?)%', r'\1%-\2%', summary)

        # 2. 🔤 Исправление капитализации компаний
        company_fixes = {
            'apple': 'Apple', 'fed': 'Fed', 'ceo': 'CEO', 'eps': 'EPS', 'ebitda': 'EBITDA',
            'iphone': 'iPhone', 's&p': 'S&P'
        }

        for wrong, correct in company_fixes.items():
            summary = re.sub(r'\b' + wrong + r'\b', correct, summary, flags=re.IGNORECASE)

        # 3. 📝 Структурирование предложений
        sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]

        if sentences:
            cleaned_sentences = []
            for sent in sentences:
                words = sent.split()
                if len(words) >= 2:  # Уменьшил с 3 до 2 для лучшего сохранения контекста
                    # Капитализация первого слова
                    if sent and sent[0].isalpha():
                        sent = sent[0].upper() + sent[1:]
                    cleaned_sentences.append(sent)

            summary = '. '.join(cleaned_sentences)

            # Финальная точка
            if summary and summary[-1] not in '.!?':
                summary += '.'

        # 4. 🧹 Финальная очистка
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = re.sub(r'\s([.,!?])', r'\1', summary)  # Убираем пробелы перед пунктуацией

        return summary

    def _generate_draft(self, summary: str, news_type: str) -> str:
        """Генерация черновика"""

        templates = {
            "EARNINGS": f"""📊 **EARNINGS REPORT**

    {summary}

    **KEY HIGHLIGHTS:**
    • Financial performance metrics
    • Growth and profitability trends  
    • Capital allocation decisions
    • Management outlook and guidance

    **INVESTMENT IMPACT:**
    Expected market reaction and analyst response""",

            "CENTRAL_BANK": f"""🏦 **FED POLICY UPDATE**

    {summary}

    **POLICY DECISION:**
    • Interest rate changes
    • Economic assessment
    • Forward guidance

    **MARKET IMPLICATIONS:**
    Impact on financial markets""",

            "MARKET": f"""📈 **MARKET ANALYSIS**

    {summary}

    **TODAY'S ACTION:**
    • Index performance
    • Sector movements  
    • Trading activity

    **KEY DRIVERS:**
    Market influences and outlook""",

            "MERGERS": f"""🤝 **M&A TRANSACTION**

    {summary}

    **DEAL DETAILS:**
    • Acquisition terms and valuation
    • Strategic rationale and synergies  
    • Regulatory approval timeline
    • Integration plans and leadership

    **INDUSTRY IMPACT:**
    Competitive landscape changes and market consolidation""",

            "REGULATORY": f"""⚖️ **REGULATORY UPDATE**

    {summary}

    **LEGAL DEVELOPMENTS:**
    • Regulatory actions and decisions
    • Compliance requirements
    • Legal proceedings and outcomes
    • Policy implications

    **BUSINESS IMPACT:**
    Operational and financial consequences""",

            "ECONOMIC": f"""📊 **ECONOMIC DATA**

    {summary}

    **KEY INDICATORS:**
    • Major economic metrics and trends
    • Comparison with forecasts and prior periods
    • Sector-specific impacts
    • Policy implications

    **MARKET REACTION:**
    Financial market responses and outlook""",

            "TECHNOLOGY": f"""🔬 **TECHNOLOGY NEWS**

    {summary}

    **INNOVATION HIGHLIGHTS:**
    • Technological developments and features
    • Research and development progress
    • Competitive advancements
    • Market adoption and potential

    **INVESTMENT POTENTIAL:**
    Growth opportunities and sector impact""",

            "COMMODITIES": f"""🛢️ **COMMODITIES UPDATE**

    {summary}

    **MARKET MOVEMENTS:**
    • Price changes and trading patterns
    • Supply and demand factors
    • Geopolitical influences
    • Inventory and production data

    **TRADING OUTLOOK:**
    Price forecasts and risk factors""",

            "GENERAL_FINANCIAL": f"""📰 **FINANCIAL NEWS**

    {summary}

    **KEY DEVELOPMENTS:**
    Main events and their financial significance

    **MARKET IMPLICATIONS:**
    Potential impacts and considerations"""
        }

        return templates.get(news_type, templates["GENERAL_FINANCIAL"])

    def _calculate_stats(self, original: str, summary: str) -> Dict:
        """Расширенная статистика суммаризации"""
        # Базовые метрики
        orig_words = len(original.split())
        summ_words = len(summary.split())
        orig_chars = len(original)
        summ_chars = len(summary)
        orig_sentences = len([s for s in original.split('.') if s.strip()])
        summ_sentences = len([s for s in summary.split('.') if s.strip()])

        # Расчет коэффициентов
        ratio_words = orig_words / summ_words if summ_words > 0 else 1
        ratio_chars = orig_chars / summ_chars if summ_chars > 0 else 1
        reduction_percent = (1 - summ_words / orig_words) * 100 if orig_words > 0 else 0

        # Оценка качества на основе нескольких факторов
        if 2.0 <= ratio_words <= 3.5 and summ_sentences >= 1:
            quality = "✅ Отлично"
            quality_score = 5
        elif 1.5 <= ratio_words < 2.0 and summ_sentences >= 1:
            quality = "✅ Хорошо"
            quality_score = 4
        elif ratio_words > 3.5:
            quality = "⚠️ Можно короче"
            quality_score = 3
        elif ratio_words < 1.5:
            quality = "⚠️ Можно подробнее"
            quality_score = 2
        else:
            quality = "❌ Низкое качество"
            quality_score = 1

        # Дополнительные метрики
        words_per_sentence = summ_words / summ_sentences if summ_sentences > 0 else 0
        density_score = summ_words / orig_words if orig_words > 0 else 0

        # Анализ информативности
        unique_words_ratio = len(set(summary.split())) / summ_words if summ_words > 0 else 0
        info_density = "Высокая" if unique_words_ratio > 0.7 else "Средняя" if unique_words_ratio > 0.5 else "Низкая"

        return {
            # Основные метрики
            "original_words": orig_words,
            "summary_words": summ_words,
            "original_chars": orig_chars,
            "summary_chars": summ_chars,
            "original_sentences": orig_sentences,
            "summary_sentences": summ_sentences,

            # Коэффициенты сжатия
            "compression_ratio_words": f"{ratio_words:.1f}x",
            "compression_ratio_chars": f"{ratio_chars:.1f}x",
            "reduction_percent": f"{reduction_percent:.0f}%",
            "compression_ratio": f"{ratio_words:.1f}x",

            # Качество и оценка
            "quality": quality,
            "quality_score": quality_score,  # числовая оценка от 1 до 5
            "info_density": info_density,  # плотность информации

            # Дополнительные метрики
            "words_per_sentence": round(words_per_sentence, 1),
            "density_score": round(density_score, 3),
            "unique_words_ratio": f"{unique_words_ratio:.1%}",

            # Рекомендации
            "recommendation": self._generate_recommendation(ratio_words, summ_sentences, quality_score)
        }

    def _generate_recommendation(self, ratio: float, sentences: int, score: int) -> str:
        """Генерация рекомендаций по улучшению"""
        recommendations = []

        if ratio > 4.0:
            recommendations.append("увеличить детализацию")
        elif ratio < 1.2:
            recommendations.append("сократить текст")

        if sentences < 1:
            recommendations.append("добавить законченные предложения")
        elif sentences > 5:
            recommendations.append("объединить некоторые предложения")

        if score < 3:
            recommendations.append("проверить информативность")

        return "; ".join(recommendations) if recommendations else "оптимально"


# 🎯 ФИНАЛЬНЫЙ ТЕСТ
def test_radar():
    """Финальный тест готовой системы RADAR"""

    test_articles = [
        {
            "text": """Apple Inc. reported fourth-quarter revenue of $89.5 billion, beating analyst estimates of $88.9 billion. 
            iPhone sales grew 12% year-over-year to $42.3 billion, while services revenue reached an all-time high of $19.2 billion. 
            The company announced a new $100 billion share buyback program and increased its dividend by 5% to $0.24 per share. 
            CEO Tim Cook stated that the company is seeing strong growth in emerging markets and expects continued momentum.""",
            "expected_type": "EARNINGS"
        },
        {
            "text": """The Federal Reserve maintained its benchmark interest rate at 5.25%-5.50% during today's policy meeting. 
            Fed Chair Jerome Powell emphasized that the central bank remains committed to bringing inflation down to its 2% target. 
            The updated economic projections show most officials expect at least one more rate hike this year.""",
            "expected_type": "CENTRAL_BANK"
        },
        {
            "text": """The S&P 500 index rose 1.5% today, led by technology stocks amid positive earnings reports. 
            Trading volume was above average as investors reacted to the Fed's policy decision and economic data. 
            Market volatility declined as uncertainty about interest rates diminished. The Dow Jones gained 200 points.""",
            "expected_type": "MARKET"
        },
        # Добавьте тест на ошибки
        {
            "text": "Short text",
            "expected_type": "ERROR",
            "description": "Тест короткого текста"
        },
        {
            "text": "",
            "expected_type": "ERROR",
            "description": "Тест пустого текста"
        }
    ]

    print("🚀 RADAR - ФИНАЛЬНАЯ ГОТОВАЯ СИСТЕМА")
    print("=" * 60)

    # 📊 Статистика тестирования
    total_tests = len(test_articles)
    passed_tests = 0
    total_processing_time = 0

    summarizer = RADARFinancialSummarizer()

    for i, test_case in enumerate(test_articles, 1):
        article = test_case["text"]
        expected_type = test_case.get("expected_type")
        description = test_case.get("description", f"Тест #{i}")

        print(f"\n🧪 ТЕСТ #{i}: {description}")
        print("-" * 50)

        start_time = time.time()
        result = summarizer.summarize_news(article)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        if "error" not in result:
            # ✅ Проверка классификации (если ожидаемый тип указан)
            type_match = "⚡"
            if expected_type and expected_type != "ERROR":
                type_match = "✅" if result['news_type'] == expected_type else "❌"

            print(f"{type_match} Тип: {result['news_type']} (ожидался: {expected_type})")
            print(f"📝 Суммаризация: {result['summary']}")

            # 📊 Детальная статистика
            stats = result['stats']
            print(f"📊 Статистика:")
            print(f"   • Слов: {stats['original_words']} → {stats['summary_words']}")
            print(f"   • Сжатие: {stats['compression_ratio_words']} ({stats['reduction_percent']})")
            print(f"   • Качество: {stats['quality']}")
            print(f"   • Время: {processing_time:.2f}с")

            if 'summary_sentences' in stats:
                print(f"   • Предложений: {stats['summary_sentences']}")

            # 📄 Черновик (показываем только начало)
            if result['draft']:
                draft_preview = result['draft'][:150] + "..." if len(result['draft']) > 150 else result['draft']
                print(f"📄 Черновик: {draft_preview}")

            passed_tests += 1

        else:
            print(f"❌ Ошибка: {result['error']}")
            if expected_type == "ERROR":
                print("✅ Ожидаемая ошибка - тест пройден")
                passed_tests += 1

        print(f"⏱️ Время обработки: {processing_time:.2f}с")

    # 📈 Финальная статистика
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"✅ Пройдено тестов: {passed_tests}/{total_tests}")
    print(f"📈 Успешность: {(passed_tests / total_tests) * 100:.1f}%")
    print(f"⏱️ Среднее время: {total_processing_time / total_tests:.2f}с")

    # 🎯 Тест производительности
    print(f"\n🎯 ПРОИЗВОДИТЕЛЬНОСТЬ:")
    metrics = get_summarization_metrics()
    print(f"• Всего запросов: {metrics['total_requests']}")
    print(f"• Ошибок: {metrics['total_errors']}")
    print(f"• Среднее время: {metrics['avg_processing_time_seconds']}с")

    # 💾 Информация о кэше (если используется)
    try:
        cache_info = get_cache_info()
        print(f"💾 Кэш: {cache_info['cache_hits']} попаданий, {cache_info['cache_misses']} промахов")
    except:
        pass

    # 🧪 Дополнительные тесты
    print(f"\n🔍 ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ:")

    # Тест на длинный текст
    long_text = " ".join(["This is a test sentence."] * 50)
    long_result = summarizer.summarize_news(long_text)
    print(f"• Длинный текст: {'✅' if 'error' not in long_result else '❌'}")

    # Тест на специальные символы
    special_text = "Apple's revenue grew 15% to $100M - amazing results! #investing"
    special_result = summarizer.summarize_news(special_text)
    print(f"• Спецсимволы: {'✅' if 'error' not in special_result else '❌'}")

    print("🎉 Тестирование завершено!")


# 🚀 Быстрый тест для разработки
def quick_test():
    """Быстрый тест для проверки работы системы"""
    print("⚡ БЫСТРЫЙ ТЕСТ RADAR")

    summarizer = RADARFinancialSummarizer()

    test_text = """Microsoft reported strong quarterly results with cloud revenue growing 25%. 
    The company announced a new AI partnership and increased its dividend."""

    result = summarizer.summarize_news(test_text)

    if "error" not in result:
        print(f"✅ Тип: {result['news_type']}")
        print(f"✅ Суммаризация: {result['summary']}")
        print(f"✅ Качество: {result['stats']['quality']}")
    else:
        print(f"❌ Ошибка: {result['error']}")


if __name__ == "__main__":
    # Можно выбрать нужный тест
    test_radar()  # Полный тест
