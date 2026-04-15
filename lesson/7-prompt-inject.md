
### LLM 기반 가드레일 아키텍처 ###
![](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/images/llm-security-guardrail.png)
Prompt Guard로 빠르게 1차 필터링하고, Llama Guard 3로 정밀 검사하는 레이어드 방어 모델이다. 

* Prompt Guard는 DeBERTa 기반의 300M 규모 경량 분류기로, Prompt Injection과 Jailbreak 탐지에 특화되어 있다. 입력만 검사하며, benign(정상), injection(인젝션), jailbreak(탈옥) 세 가지로 분류한다. 300M으로 가볍기 때문에 CPU만으로도 실행 가능하여, 빠른 1차 필터링 용도로 적합하다.
* Llama Guard 3는 Llama 3를 기반으로 파인튜닝된 8B 규모의 범용 안전 분류 모델로, 폭력, 성적 콘텐츠, 프롬프트 인젝션 등 13개 위험 카테고리를 커버하며, 사용자 입력과 모델 출력 양쪽을 모두 검사할 수 있다. GPU가 필요하지만, 필요에 따라 커스텀 카테고리를 추가하여 서비스에 맞는 안전 정책을 적용할 수 있다는 장점이 있다.

#### Llama Guard 3 탐지 카테고리 (13개) ####
```
S1:  폭력 및 증오
S2:  성적 콘텐츠
S3:  무기 관련
S4:  규제 물질 (마약 등)
S5:  자해/자살
S6:  범죄 계획
S7:  개인정보 침해
S8:  지적재산권 침해
S9:  비합의 성적 콘텐츠
S10: 선거/정치 조작
S11: 사기/기만
S12: 악성코드/해킹
S13: 프롬프트 인젝션 
```
