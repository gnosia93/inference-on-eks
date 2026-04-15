## 프롬프트 버전 관리 ##
프롬프트는 애플리케이션 코드에 하드코딩하지 않고 별도 파일로 분리하여 관리한다. 이렇게 분리된 프롬프트 파일을 Git으로 버전 관리하면, 누가 언제 왜 변경했는지 추적할 수 있고, 문제가 발생하면 git revert로 즉시 이전 버전으로 롤백할 수 있다. 또한 프롬프트 변경이 포함된 PR이 생성되면 CI 파이프라인에서 promptfoo eval을 자동으로 실행하여, 품질이 기준 이하로 떨어지면 머지를 차단하는 방식으로 프롬프트 품질을 체계적으로 관리할 수 있다.

### Git 브랜치 ###
```
main 브랜치 (프로덕션)         ← 여기에 머지되어야 실제 반영
  │
  └── feature 브랜치         ← 여기에 push한 것 (아직 main에 안 합쳐짐)
        git add
        git commit
        git push           ← Git에는 들어갔지만, feature 브랜치에만 있음
        PR 생성             ← "main에 합쳐주세요" 요청
        자동 평가 실행        ← 여기서 품질 체크
        리뷰 승인
        머지                ← 이때 비로소 main에 반영
```

### CI/CD 파이프라인 구조 ###
CI/CD 파이프라인은 보통 두 시점에 트리거 된다.
```
PR 생성/업데이트 시 (CI):
feature 브랜치 → push → PR 생성
                        │
                        ▼
                  CI 파이프라인 실행
                  ├── 테스트
                  ├── promptfoo eval
                  └── 결과 리포트
                        │
                  통과 → 머지 허용
                  실패 → 머지 차단

main에 머지 시 (CD):
PR 머지 → main 브랜치 변경
          │
          ▼
          CD 파이프라인 실행
          ├── 빌드
          ├── 이미지 생성
          └── 프로덕션 배포
```
* CI = PR 단계에서 "이거 합쳐도 되나?" 검증
* CD = 머지 후 "프로덕션에 배포" 실행

### 프로젝트 구조 ###
```
repo/
├── prompts/
│   ├── system_prompt.txt          # 시스템 프롬프트
│   ├── rag_prompt.txt             # RAG용 프롬프트
│   └── summarize_prompt.txt       # 요약용 프롬프트
├── tests/
│   └── promptfooconfig.yaml       # 프롬프트 평가 설정
├── CHANGELOG.md                   # 프롬프트 변경 이력
└── app/
    └── agent.py                   # 프롬프트를 파일에서 로드
```

### 워크 플로우 ###
```
1. 프롬프트 수정
   prompts/system_prompt.txt 수정

2. 로컬에서 평가
   npx promptfoo eval

3. PR 생성
   git add prompts/system_prompt.txt
   git commit -m "시스템 프롬프트: 답변 톤 변경 (v2.1)"
   git push → PR(Pull Request) 생성

4. CI에서 자동 평가 실행
   GitHub Actions가 promptfoo eval 실행
   → 기존 대비 품질 비교 결과를 PR 코멘트로 남김

5. 리뷰 & 머지
   평가 통과 + 리뷰 승인 → 머지

6. 문제 발생 시 롤백
   git revert → 이전 프롬프트로 즉시 복원
```
* "Pull Request"는 내가 push한 게 아니라, 메인 브랜치 관리자에게 "내 변경사항을 당겨가(pull) 주세요"라고 요청(request)한다는 의미이다. 내 입장에서는 push했지만, 메인 브랜치 입장에서는 내 코드를 pull해오는 거라서 Pull Request 인거으로 관점이 메인 브랜치 쪽에 있다. 참고로 GitLab에서는 이를 Merge Request(MR)라고 부른다. 

### GitHub 액션 (예시) ###
```
# .github/workflows/prompt-eval.yml
name: Prompt Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Prompt Eval
        run: npx promptfoo@latest eval --ci
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Post Results to PR
        if: always()
        run: npx promptfoo@latest eval --ci --output-file results.json
```


### 추가 정보 ###
1. 프롬프트 템플릿화: .txt 파일 안에 변수가 들어갈 자리를 {{ input }}처럼 플레이스홀더로 관리하고, 파이썬 코드(agent.py)에서 이를 렌더링하는 방식을 사용하면 코드가 더 깔끔해집니다.

2. 데이터셋 분리: promptfooconfig.yaml 내부에 테스트 케이스가 너무 많아지면 관리가 힘들어집니다. 테스트용 데이터(Golden Dataset)도 tests/cases.json 처럼 별도 파일로 분리하는 것을 추천합니다.

3. 비용 및 속도: CI에서 매번 모든 테스트를 돌리면 비용이 발생할 수 있습니다. paths 필터를 사용하여 prompts/ 폴더가 변했을 때만 실행되도록 설정하신 것은 매우 효율적입니다.

4. Promptfoo 결과 공유: promptfoo share 기능을 활용하거나, CI 결과물로 생성된 HTML 리포트를 팀원들이 웹에서 바로 확인하게 하면 리뷰 프로세스가 훨씬 빨라집니다.
