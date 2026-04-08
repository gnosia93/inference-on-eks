

### 캐싱전략 ###
같은 질문이 반복될 때 LLM을 다시 호출하지 않고 캐시된 응답을 반환한다.
```python
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
import redis

# Redis 캐시 설정
set_llm_cache(RedisCache(redis_client=redis.Redis()))

# 이후 같은 프롬프트로 호출하면 캐시에서 반환
# → LLM 호출 안 함 → 토큰 비용 0, 응답 즉시
```

### 모델 라우팅 ###
```python
def route_to_model(query: str):
    # 간단한 분류기로 질문 복잡도 판단
    classifier_response = small_llm.invoke(
        f"이 질문이 단순한지 복잡한지 판단해: {query}\n답변: simple 또는 complex"
    )
    if "simple" in classifier_response:
        return small_llm   # 3B 모델 - 빠르고 저렴
    else:
        return large_llm   # 27B 모델 - 느리지만 정확
```
실제로는 분류기 자체도 비용이므로, 프롬프트 길이나 키워드 기반으로 단순하게 라우팅하는 경우도 많다.
셀프 호스팅(vLLM)이면 API 비용은 없지만 GPU 시간이 비용이라, 작은 모델로 처리할 수 있는 건 작은 모델로 보내서 GPU 리소스를 아끼는 게 핵심이다.
