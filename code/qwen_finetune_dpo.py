# qwen_finetune_dpo.py
# Qwen3.5-27B DPO (Direct Preference Optimization) 파인튜닝 예제

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# ============================================================
# 1. 모델 로드
# ============================================================
model_name = "Qwen/Qwen3.5-27B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
#    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. LoRA 설정
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# 3. DPO 학습 데이터 (chosen vs rejected 쌍)
# ============================================================
dpo_data = [
    {
        "prompt": "GPU OOM이 발생했을 때 해결 방법을 알려줘",
        "chosen": "GPU OOM 해결 방법: 1) 배치 크기 줄이기 + Gradient Accumulation으로 effective batch size 유지 2) Mixed Precision(BF16) 사용으로 메모리 50% 절감 3) Activation Checkpointing 적용 4) 모델이 단일 GPU에 안 들어가면 ZeRO Stage 3 또는 FSDP 사용",
        "rejected": "GPU를 더 사세요. 메모리가 부족하면 더 큰 GPU를 쓰면 됩니다."
    },
    {
        "prompt": "Kubernetes Pod가 CrashLoopBackOff 상태일 때 디버깅 방법은?",
        "chosen": "CrashLoopBackOff 디버깅: 1) kubectl logs <pod> --previous로 이전 크래시 로그 확인 2) kubectl describe pod <pod>로 이벤트 확인 3) 리소스 제한(memory limit) 초과 여부 확인 4) liveness/readiness probe 설정 확인 5) 이미지 pull 실패 여부 확인",
        "rejected": "Pod를 삭제하고 다시 만드세요."
    },
    {
        "prompt": "Docker 컨테이너에서 GPU를 사용하려면 어떻게 해야 해?",
        "chosen": "Docker에서 GPU 사용: 1) NVIDIA Container Toolkit 설치 2) docker run --gpus all 옵션 사용 3) nvidia-smi로 GPU 인식 확인 4) CUDA 버전과 PyTorch 호환성 확인",
        "rejected": "Docker는 GPU를 지원하지 않습니다. 호스트에서 직접 실행하세요."
    },
    {
        "prompt": "Prometheus에서 GPU 메트릭을 수집하는 방법은?",
        "chosen": "DCGM Exporter를 사용합니다. 1) dcgm-exporter 컨테이너 실행 (포트 9400) 2) prometheus.yml에 scrape target 추가 3) DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_FB_USED 등 메트릭 수집 4) Grafana 대시보드로 시각화",
        "rejected": "nvidia-smi를 cron으로 돌려서 파일에 저장하세요."
    },
    {
        "prompt": "MLflow로 실험 추적하는 기본 코드를 보여줘",
        "chosen": "import mlflow\nmlflow.set_experiment('my-experiment')\nwith mlflow.start_run():\n    mlflow.log_param('lr', 0.001)\n    mlflow.log_param('batch_size', 32)\n    mlflow.log_metric('loss', 0.5)\n    mlflow.log_metric('accuracy', 0.92)\n    mlflow.pytorch.log_model(model, 'model')",
        "rejected": "실험 추적은 필요 없습니다. 터미널 출력을 복사해서 메모장에 저장하세요."
    },
    {
        "prompt": "Terraform으로 AWS VPC를 생성하는 기본 코드는?",
        "chosen": "resource \"aws_vpc\" \"main\" {\n  cidr_block = \"10.0.0.0/16\"\n  enable_dns_support = true\n  enable_dns_hostnames = true\n  tags = { Name = \"my-vpc\" }\n}\nresource \"aws_subnet\" \"public\" {\n  vpc_id = aws_vpc.main.id\n  cidr_block = \"10.0.1.0/24\"\n  map_public_ip_on_launch = true\n}",
        "rejected": "AWS 콘솔에서 클릭으로 만드세요. IaC는 복잡하기만 합니다."
    },
    {
        "prompt": "Ray를 사용한 분산 학습 기본 코드를 보여줘",
        "chosen": "import ray\nfrom ray import train\nfrom ray.train.torch import TorchTrainer\n\ndef train_func():\n    model = ...\n    optimizer = ...\n    for epoch in range(10):\n        loss = train_step(model, optimizer)\n        train.report({'loss': loss})\n\ntrainer = TorchTrainer(train_func, scaling_config=train.ScalingConfig(num_workers=4, use_gpu=True))\nresult = trainer.fit()",
        "rejected": "분산 학습은 어렵습니다. 그냥 GPU 1장으로 오래 돌리세요."
    },
    {
        "prompt": "Slurm에서 멀티노드 GPU 학습 작업을 제출하는 sbatch 스크립트를 보여줘",
        "chosen": "#!/bin/bash\n#SBATCH --job-name=train\n#SBATCH --nodes=4\n#SBATCH --ntasks-per-node=8\n#SBATCH --gpus-per-node=8\n#SBATCH --cpus-per-task=12\n#SBATCH --exclusive\n\nsrun --gpu-bind=closest torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1):29400 train.py",
        "rejected": "각 노드에 SSH로 접속해서 수동으로 python train.py를 실행하세요."
    },
]

# Chat 형식 변환
def format_dpo(example):
    system = "You are a helpful DevOps and ML engineering assistant."
    prompt_msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": example["prompt"]},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True),
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = Dataset.from_list(dpo_data)
dataset = dataset.map(format_dpo)

# ============================================================
# 4. DPO 학습
# ============================================================
training_args = DPOConfig(
    output_dir="./qwen-devops-dpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="adamw_torch",
    beta=0.1,
    max_length=2048,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# ============================================================
# 5. 저장
# ============================================================
model.save_pretrained("./qwen-devops-dpo")
tokenizer.save_pretrained("./qwen-devops-dpo")
print("Done!")
