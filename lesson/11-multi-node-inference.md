## 멀티 노드 인프런스 최적화 ##
단일 노드에 담기 어려운 대규모 모델을 여러 노드에 걸쳐 서빙할 때, 노드 간 통신 오버헤드를 최소화하고 GPU 활용률을 최대화하는 것이 핵심이다. 모델 크기, 지연시간 요구사항, 처리량 목표에 따라 병렬화 전략과 최적화 기법의 조합이 달라진다. 일반적으로 노드 내부는 TP, 노드 간은 PP를 사용하는 하이브리드 구성을 적용한다.

### 텐서 병렬리즘 (Tensor Parallelism, TP) ###

* 레이어의 가중치 텐서를 행또는 열 방향으로 분할해서 여러 GPU에 나눔
* 레이어 내부에서 GPU 간 통신(all-reduce)이 발생하므로 노드 내부(NVLink 등 고속 인터커넥트)에서 주로 사용

### 파이프라인 병렬리즘 (Pipeline Parallelism, PP) ###

* 모델의 레이어를 순차적으로 여러 노드에 배치
* 노드 간 통신량이 TP보다 적어서 멀티노드에 적합. 다만 파이프라인 버블(유휴 시간)이 발생하므로, 마이크로배칭으로 이를 줄이는 것이 핵심.

### 실무에서 쓰이는 도구들 ###
* vLLM: PagedAttention + 텐서/파이프라인 병렬리즘 지원
* TensorRT-LLM: NVIDIA 최적화, 멀티노드 지원


### TP + PP 하이브리드 구성 시 고려사항 ###
* TP degree 선택: 노드 내 GPU 수(보통 8)를 넘지 않도록 설정. NVLink 대역폭(예: A100 NVSwitch 600GB/s)을 넘어 노드 간 TP를 걸면 all-reduce 지연이 급격히 증가.
* PP stage 분할: 레이어를 균등 분할하는 것이 기본이지만, 임베딩 레이어나 LM head가 있는 첫/마지막 스테이지는 연산량이 다르므로 불균형이 발생한다. vLLM이나 TensorRT-LLM에서는 이를 자동 밸런싱하는 옵션이 있다.
* 마이크로배치 수: PP 버블을 줄이려면 마이크로배치 수를 PP 스테이지 수의 4배 이상으로 잡는 것이 경험적으로 효과적이다. 다만 인퍼런스에서는 배치 크기가 제한적이라 트레이닝만큼 효과를 보기 어려울 수 있다.

### 노드 간 통신 최적화 ###
* NCCL 튜닝: NCCL_ALGO, NCCL_PROTO 환경변수로 통신 알고리즘/프로토콜 선택 가능. InfiniBand 환경에서는 NCCL_IB_HCA로 HCA 디바이스를 명시적으로 지정하면 성능이 개선된다.
* PP 에서의 비동기 전송: 스테이지 간 activation 전달을 비동기로 처리하면 computation과 communication을 오버랩할 수 있다.
* KV Cache 관리: 멀티노드에서 continuous batching을 쓸 때, 각 노드의 KV cache 상태를 동기화하는 오버헤드가 존재한다. vLLM의 PagedAttention이 이 부분에서 메모리 효율을 크게 높여준다.

### 추가로 고려할 만한 기법들 ###
* Speculative Decoding: 작은 draft 모델로 여러 토큰을 먼저 생성하고 큰 모델로 검증. 멀티노드 환경에서 draft 모델을 별도 노드에 배치하면 지연시간을 줄일 수 있다.
* Expert Parallelism: MoE 모델의 경우 expert를 노드별로 분산 배치하는 전략이 TP/PP와 별도로 필요하다.
* Disaggregated Prefill/Decode: prefill(프롬프트 처리)과 decode(토큰 생성)를 별도 노드 그룹에서 처리하는 아키텍처. 처리량과 지연시간을 독립적으로 스케일링할 수 있다.
