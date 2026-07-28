# 🛡️ VibeShield

**Intelligent middleware for [VibeMarketolog Agent API](https://lk.vibemarketolog.ru/docs/agent-api) — agent loop protection, semantic deduplication cache, and cost guardrails.**

> Предложение для позиции AI-разработчика в Вайб-Маркетолог.

---

## Проблема

Когда AI-агенты (Claude через MCP, ChatGPT, LangChain-пайплайны) работают с генеративными API, возникают три системные проблемы:

| Проблема | Последствие | Кто страдает |
|----------|------------|--------------|
| **Агентные петли** — агент зацикливается на одном вызове `/generate` | Баланс пользователя обнуляется за минуты | Пользователь MCP-коннектора |
| **Дубликаты генераций** — промпты «закат над морем» и «закат на море, 4К» дают одинаковый результат | 20–40% upstream-расходов тратится впустую | Платформа (маржа) |
| **Отсутствие budget caps** — нет механизма остановки при превышении бюджета сессии | Неконтролируемые списания | Пользователь и платформа |

**Ни один AI-агрегатор** (OpenRouter, VseGPT, Bothub, fal.ai) не решает эти задачи на уровне middleware.

---

## Решение

VibeShield — middleware-слой, который встаёт **между агентом и API** без изменения существующей инфраструктуры:

```
AI Agent (Claude MCP / ChatGPT / LangChain)
    │
    ▼
┌─────────────────────────────────────┐
│          VibeShield                 │
│  ┌─────────────┐ ┌──────────────┐  │
│  │  Circuit     │ │  Semantic    │  │
│  │  Breaker     │ │  Cache       │  │
│  └─────────────┘ └──────────────┘  │
│  ┌─────────────────────────────┐   │
│  │  Cost Estimator & Guards    │   │
│  └─────────────────────────────┘   │
└──────────────────┬──────────────────┘
                   │
                   ▼
        VibeMarketolog Agent API
```

---

## Модули

### 1. 🔌 Circuit Breaker — защита от агентных петель

Отслеживает паттерны вызовов в реальном времени и блокирует деструктивное поведение:

- **Duplicate Detection** — если агент вызвал `/generate` с идентичными параметрами 3 раза подряд → стоп
- **Semantic Loop Detection** — если текстовая similarity между промптами > 92% → стоп  
- **Budget Cap** — если сумма сессии превышает лимит → стоп
- **Rate Limiting** — если > 20 вызовов в минуту → стоп
- **Corrective Injection** — генерирует текст для инъекции в контекст агента, чтобы вывести его из петли

```python
from vibe_shield.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(budget_cap=200.0, max_duplicates=3)

# Перед каждым вызовом API:
if breaker.check("/generate", payload, cost=estimated_cost):
    result = await client.generate(payload)
    breaker.record_success("/generate", payload, cost=result.cost)
else:
    # Агент зациклился — отправляем corrective message
    print(breaker.get_corrective_message())
```

### 2. 💾 Semantic Cache — экономия на дубликатах

Определяет семантически идентичные промпты и возвращает кэшированный результат:

- **Exact Match** — O(1) по хешу `(model + prompt + params)`
- **Fuzzy Match** — SequenceMatcher для обнаружения перефразирований (upgradeable до sentence-transformers / CLIP)
- **Per-Model Isolation** — кэш `z-image` не пересекается с `grok-ttv-10`
- **TTL 24h** — совпадает с кэшем самого VibeMarketolog API
- **LRU Eviction** — ограничение по памяти

```python
from vibe_shield.semantic_cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.90)

# Prompt 1: "красивый закат над морем" → генерация, 15₽
cache.put("z-image", "красивый закат над морем", result, cost=15.0)

# Prompt 2: "красивый закат на море, высокое качество" → cache hit, 0₽!
cached = cache.get("z-image", "красивый закат на море, высокое качество")
# → returns cached result, saves 15₽
```

### 3. 🚀 VibePipeline — всё в одном

Объединяет клиент, circuit breaker и кэш в единый интерфейс:

```python
from vibe_shield.pipeline import VibePipeline

async with VibePipeline("your_token", budget_cap=300.0) as pipe:
    result = await pipe.generate_safe({
        "type": "image",
        "model": "z-image",
        "prompt": "cyberpunk cityscape at night",
        "aspect_ratio": "16:9",
    })
    
    print(f"URL: {result.display_url}")
    print(f"Cost: {result.cost}₽")
    print(f"Stats: {pipe.stats()}")
```

---

## Экономический эффект

| Метрика | Без VibeShield | С VibeShield |
|---------|---------------|-------------|
| Дубликаты (30% от объёма) | Оплачиваются полностью | Из кэша бесплатно |
| Агентные петли | Баланс → 0₽ | Стоп после 3 итераций |
| Upstream-расходы | 100% | ~70–80% (экономия 20–30%) |

---

## Установка

```bash
pip install -e .

# С поддержкой semantic embeddings (production):
pip install -e ".[semantic]"
```

## Тесты

```bash
pip install -e ".[dev]"
pytest -v
```

---

## Архитектура и дальнейшее развитие

**Реализовано (v0.1):**
- [x] Async HTTP клиент с экспоненциальным backoff и Retry-After
- [x] Circuit Breaker (4 стратегии детекции)
- [x] Semantic Cache (exact + fuzzy match, LRU, TTL)
- [x] Pipeline orchestrator с batch-генерацией
- [x] Dry-run cost estimation перед списанием
- [x] Idempotency key support

**Roadmap (v0.2+):**
- [ ] OpenAI-compatible proxy (`/v1/chat/completions` → VibeMarketolog API) для интеграции с LangChain/CrewAI
- [ ] CLIP / sentence-transformers для настоящего semantic matching
- [ ] MCP Server Tool `generate_with_shield` для встраивания в Claude Code
- [ ] Redis-backed distributed cache для multi-instance deployments
- [ ] Prometheus metrics exporter
- [ ] Webhook listener для push-based результатов

---

## Лицензия

MIT

## Автор

**Максим Ли** — [GitHub](https://github.com/twaise) · [Telegram](https://t.me/twaise12)
