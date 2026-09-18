# YT Shorts AI · 유튜브 쇼츠 자동 편집 AI

시청자 재시청 데이터(YouTube "Most Replayed" 히트맵) 기반으로 긴 게임 영상에서
하이라이트를 탐지하고 세로형(9:16) 쇼츠로 자동 편집하는 **로컬 GPU 가속 딥러닝 멀티모달 시스템**.

> 저장소: `solidSnakesado/yt_shorts_ai` · 1인 개발
> 현재 국면: **운영(operations) — 학습 완료, round18이 최종 어댑터**

---

## 핵심 특징

- **재시청 히트맵을 정답 신호로** 학습한 하이라이트 판별 모델 (수동 라벨 의존 최소화)
- **멀티모달 하이라이트 탐지** — 프레임 + 오디오(효과음·환호·BGM 등 비언어 신호)까지 반영
- **2개 모델 스택 병렬 운영** (물리·브랜치 격리)
  - **Gemma 4 E4B 오디오 피벗**(활성) — raw audio 직접 입력, OK-rate **82.4%**
  - **Qwen2.5-VL-7B QLoRA**(동결) — 전사 텍스트 기반, round3 OK-rate 82.1%
- **적응형 리프레이밍** — YOLO 피사체 추적 크롭 + 4:3/4:5/9:16/16:9 레터박스
- **로컬 완결** — 다운로드→전사→하이라이트→편집→인코딩까지 12GB VRAM 1대에서 순차 실행

---

## 아키텍처 개요

MVA (Minimum Viable Architecture) — 계층형 + 레포지토리 패턴 + 의존성 주입 체인
(단방향: **API → 서비스 → 레포지토리 → DB**)

```
app/
├── main.py
├── core/
│   ├── config.py                 # Qwen 스택 설정 (Pydantic Settings, .env 바인딩)
│   ├── gemma_config.py           # Gemma 피벗 전용 설정 (GEMMA_* 키만 사용, Qwen과 격리)
│   ├── database.py               # 비동기 DB 연결/세션
│   ├── security.py               # JWT 인증, 비밀번호 해싱
│   ├── dependencies.py           # DI 체인
│   ├── gpu_manager.py            # GPU VRAM 관리 (Whisper/LLM/YOLO 순차 로드/언로드)
│   └── llm_server.py             # llama-server 서브프로세스 관리 (Qwen GGUF 경로)
├── api/v1/
│   ├── router.py
│   ├── projects.py               # 프로젝트 엔드포인트
│   ├── shorts.py                 # 쇼츠 엔드포인트
│   ├── system.py                 # 시스템 상태 (GPU 모니터링)
│   └── heatmap.py                # 히트맵 수집 엔드포인트
├── models/
│   └── domain.py                 # SQLModel 도메인 모델 (Project, Shorts)
├── schemas/
│   └── api.py                    # Pydantic 요청/응답 스키마
├── repositories/                 # base / project / shorts
└── services/
    ├── video_service.py          # 다운로드 + 오디오 추출 (yt-dlp, FFmpeg, 4단 포맷 폴백)
    ├── analysis_service.py       # 전사(Whisper) + 하이라이트 추출 오케스트레이션
    ├── llm_highlight_extractor.py# LLM 프롬프트/호출/파싱
    ├── vlm_client.py             # 모델 셀렉터 — GEMMA_ENABLED 분기(Gemma 우선 / Qwen 폴백)
    ├── gemma_phase_inference.py  # Gemma e2e 회귀 추론 (30s 슬라이딩 윈도우 → hook_score)
    ├── gemma_audio_extractor.py  # 구간 오디오 추출 (Gemma 입력)
    ├── phase2_inference.py       # Qwen 10초 클립 추론 (동결 스택)
    ├── frame_extractor.py        # 영상 프레임 추출 (start/end/fps 지정)
    ├── transcript_chunker.py     # 전사 청크 분할/재랭킹
    ├── editing_service.py        # 리프레이밍 + 자막 + 인코딩 오케스트레이션
    ├── reframe_engine.py         # 클립 추출 + YOLO 추적 + 적응형 크롭
    ├── letterbox_engine.py       # 콘텐츠 크롭 + 검정 여백, 캔버스 9:16 고정
    └── subtitle_generator.py     # ASS 자막 + FFmpeg 합성/인코딩 (현재 발행 경로에서 skip)

scripts/                          # 학습·데이터·추론 도구 (레포 루트 gemma_* 스크립트 포함)
    ├── gemma_e2e_model.py        # e2e 회귀 모델 (타워 동결 + 언어층 QLoRA + 회귀 헤드)
    ├── gemma_e2e_collate.py      # e2e collate (라벨 누설 방지)
    ├── gemma_e2e_train.py        # A100 학습 (랭킹 손실 + MSE, 체크포인트 회전)
    ├── gemma_e2e_infer.py        # 로컬 12GB 추론 + 분포 판정
    ├── gemma_dataset_builder.py  # 데이터셋 빌드 (pos/neg, 30초, video_id dedup)
    ├── package_dataset.py        # 병합·셔플 + 영상 단위 eval split
    ├── measure_ok_rate.py        # OK-rate 측정 (model_version별, 95% CI)
    └── gemma_ok_breakdown.py     # 점수 밴드·탐색/활용·신뢰 계층 분해
```

---

## 모델 스택 (2계통, 격리)

`.env` 토글로 Step 4 하이라이트 추출 경로를 선택합니다.

| 스택 | 브랜치 | 입력 | 런타임 | 최종 지표 | 상태 |
|---|---|---|---|---|---|
| **Gemma 4 E4B 오디오 피벗** | `feature/gemma4-audio` | 1fps 프레임 + 30s raw 오디오 | 4bit 로컬(회귀 헤드) | **OK-rate 82.4%** (652건, CI ±2.9%p) | 활성 |
| **Qwen2.5-VL-7B QLoRA** | `feature/phase2-clip-training` | 10초 클립 프레임 + 전사 | llama-server GGUF + LoRA | round3 OK-rate 82.1% | 동결·보존 |

- **모델 셀렉터**: `GEMMA_ENABLED=true` → `gemma_phase_inference` 경로 우선.
  `false` → Qwen `phase2_inference`(`LORA_PIPELINE=phase2`) 경로.
- **Gemma 추론 흐름**: 영상 → 30초 슬라이딩 윈도우 → [프레임 + 오디오] → 마지막 hidden 마스크
  평균 풀링 → 회귀 헤드 → `hook_score`. 타임스탬프는 모델 출력이 아니라 **클립 윈도우에서 재구성**.
- **격리 원칙**: Qwen 어댑터(round1/1b/2/3)와 모든 Gemma 라운드 산출물은 **삭제·덮어쓰기 금지**.

---

## 7단계 파이프라인

```
유튜브 URL 입력
  → [1] 프로젝트 생성            POST /api/v1/projects/
  → [2] 영상 다운로드             POST /api/v1/projects/{id}/download
  → [3] 음성 전사 (Whisper)      POST /api/v1/projects/{id}/transcribe
  → [4] 하이라이트 추출           POST /api/v1/projects/{id}/analyze
         (GEMMA_ENABLED=true → Gemma 회귀 / false → Qwen phase2)
  → [5] 리프레이밍               POST /api/v1/shorts/{id}/edit
         (layout="crop": YOLO 추적 / layout="letterbox": 크롭+검정 여백, 캔버스 9:16)
  → [6] 자막 합성 (ASS)          POST /api/v1/shorts/{id}/subtitle   (현재 발행 경로 skip)
  → [7] 최종 인코딩 (NVENC)      POST /api/v1/shorts/{id}/encode
  → outputs/{title}.mp4 (H.264, AAC)
```

- **발행 화질(60일차 확정)**: 전체 실행은 1080p로 다운로드 — 실측 `1728×1080 h264 + 128kbps`.
  다운로드 로그가 상시 품질 게이트로 동작해 침묵 강등이 재발하지 않습니다.
- **라벨링 경로는 480p** — 모델 입력이 336px라 프레임 품질 동일. 발행은 반드시 전체 실행(1080p) 경로.

---

## 빠른 시작

### 1) 공통 설치

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# HuggingFace 로그인
uv run hf auth login

# Node 22 활성화 (yt-dlp n-challenge 해결에 필수, 세션마다)
export NVM_DIR="$HOME/.nvm" && source "$NVM_DIR/nvm.sh" && nvm use 22

# 환경변수
cp .env.example .env
```

### 2) 추론 모델 다운로드

**Gemma 스택 (활성 · 권장)** — Q4_K_M 베이스 + mmproj Q8_0(오디오 포함):

```bash
# 로컬 추론 자산 (models/gguf/gemma4/ 및 어댑터는 학습 산출물로 보존)
uv run hf download unsloth/gemma-4-E4B-it-GGUF \
    --include "*Q4_K_M*" --local-dir ./models/gguf/gemma4/
uv run hf download unsloth/gemma-4-E4B-it-GGUF \
    --include "*mmproj*Q8_0*" --local-dir ./models/gguf/gemma4/
# .env에서 GEMMA_ENABLED=true, GEMMA_INFER_ADAPTER_DIR=<round18 최종 어댑터 경로>
```

**Qwen 스택 (동결 · 폴백)**:

```bash
uv run hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \
    --include "*Q5_K_M*" --local-dir ./models/llm/
uv run hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \
    --include "*mmproj*" --local-dir ./models/llm/
```

### 3) llama.cpp 빌드 (Qwen GGUF 경로용)

```bash
# GPU 아키텍처에 맞게 CUDA_ARCHITECTURES 조정
# RTX 5070 Ti (Blackwell): 120 / RTX 4090 (Ada): 89 / RTX 3090 (Ampere): 86
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build llama.cpp/build --config Release -j4 --target llama-server
cp llama.cpp/build/bin/llama-server ./bin/llama-server
```

> WSL에서 llama.cpp 빌드는 반드시 `-j4` 이하. 전체 병렬은 시스템 정지를 유발합니다.

### 4) 서버 실행

```bash
# 백엔드 (터미널 1)
uv run uvicorn app.main:app --reload \
    --reload-exclude "unsloth_compiled_cache/*" \
    --host 0.0.0.0 --port 8000

# 프론트엔드 (터미널 2)
python3 -m http.server 3000
# http://localhost:3000/test.html
```

> `.env`를 바꾼 뒤에는 `--reload`가 환경변수를 반영하지 못하므로 백엔드를 **재시작**해야 합니다.

---

## 학습 (완료 — 참고용)

학습 국면은 종료되었고 **round18이 최종 어댑터**입니다. 학습은 **Colab Pro+ A100 80GB**에서
진행했으며, 로컬 12GB 카드는 추론 전용입니다. 아래는 재현·이해용 요약입니다.

```bash
# 히트맵 수집 → 데이터셋 빌드 (pos/neg 1:1, 30초 클립, video_id dedup)
uv run python -m scripts.collect_heatmaps
uv run python scripts/gemma_dataset_builder.py       # + run_gemma_neg.py (네거티브)
uv run python scripts/package_dataset.py --eval-ratio 0.2   # 영상 단위 split (누수 금지)

# A100에서 e2e 회귀 학습 (랭킹 hinge + 0.3×MSE, 회귀 헤드)
python scripts/gemma_e2e_train.py

# OK-rate 측정 (Qwen vs Gemma 비교, model_version별 95% CI)
python3 measure_ok_rate.py --model_version
python3 gemma_ok_breakdown.py
```

**핵심 학습 교훈**
- `eval_loss`로는 이진 붕괴를 진단할 수 없음 — 재학습 후 반드시 추론 분포를 확인.
- LLM 텍스트 생성(CE)으로 회귀를 수행하면 상수·양극 붕괴 → **회귀 헤드 + 랭킹 손실**로 해결.
- eval split은 **영상 단위만** 허용(클립 무작위 분리는 누수).
- PLE(`embed_tokens_per_layer`)가 최대 VRAM 병목(~5.64GB) → CPU 오프로드로 해소.

---

## 시스템 요구사항

| 항목 | 추론(로컬) | 학습(클라우드, 완료) |
|------|------|------|
| GPU | RTX 5070 Ti Laptop (11.94GB) | A100 80GB (Colab Pro+) |
| RAM | 16GB+ | — |
| OS | WSL2 Ubuntu 24.04 | — |
| Python | 3.11.x (uv) | 3.11.x |
| CUDA | 12.x+ | 12.8 (torch 2.11.0+cu128) |
| FFmpeg | 6.x+ (NVENC, libass) | — |
| Node | 22 (yt-dlp) | — |

**환경 핀 (Gemma)**: `unsloth==2026.6.9`, `unsloth_zoo==2026.6.7`, `torch 2.11.0+cu128`,
로컬 `transformers 5.5.0` 고정 (미핀 설치 시 실패).

**VRAM 사용량 (순차 로딩)**

```
[Step 3] Whisper medium              ~5GB   → 언로드
[Step 4] Gemma 4 E4B 4bit(회귀 추론)   ~6-8GB → 언로드 (release_vram + gc.collect)
         / 또는 Qwen phase2 GGUF      ~5GB   → 언로드
[Step 5] YOLOv8n                     ~1GB   → 언로드
```

---

## 성능 지표

- **Gemma round18 (최종)**: OK-rate **82.4%** (652건, CI ±2.9%p)
- **Qwen 궤적(비교 기준)**: 41.0% → 50.3%(round1b) → 61.8%(round2) → **82.1%(round3)**
- **고신뢰 순도**: 87.1%(공칭) / 93.9%(엔딩 크레딧 NO 9건 제외)
- **잔여 구조적 약점**: 0.6대 점수 밴드 판별력 ~61% (가중치 문제 — 학습 없이 개선 대상)
- **측정 표준**: `OK-rate = OK / (OK + 선택NO)`, `model_version`별 그룹·95% CI.
  Spearman(rho)은 사전 필터, OK-rate 분해가 최종 판정.

---

## 개발 로드맵

| 일차 | 주요 활동 | 상태 |
|------|----------|:---:|
| 1-2 | 아키텍처 확립, DB 모델링 | ✅ |
| 3-5 | Whisper ASR, 다운로드 파이프라인 | ✅ |
| 6-7 | LLM 하이라이트 추출 | ✅ |
| 8-10 | YOLOv8 리프레이밍, FFmpeg 크롭 | ✅ |
| 11-13 | 자막 합성, 인코딩, LLM 자동 길이 판단 | ✅ |
| 14-17 | VLM 멀티모달, 청크 분할, 히트맵 수집 | ✅ |
| 21-24 | 종횡비 추천, 판별기/생성기 QLoRA (Qwen) | ✅ |
| 26 | 레터박스 엔진 도입 | ✅ |
| 34-37 | OK-rate 측정 체계, 인간 피드백 루프 | ✅ |
| 38-49 | **Gemma 4 E4B 오디오 피벗** (붕괴 규명 → 회귀 헤드 아키텍처) | ✅ |
| 50-57 | Gemma 추론 모듈, OK-rate 실측, 계층 발행, 오분류 분석 | ✅ |
| 59 | **학습 국면 종료 선언** (round18 = 최종), 포트폴리오 3종, 레터박스 개편 | ✅ |
| 60 | 발행 화질·음질 강등 사건 규명·해결 (`player_client=web` 제거) | ✅ |
