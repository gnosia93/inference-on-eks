
### 모델 컴파일 ###

```bash
# S3 버킷 생성
aws s3 mb s3://your-bucket-name --region ap-northeast-2

# ServiceAccount에 S3 접근 권한 부여
eksctl create iamserviceaccount \
  --name s3-access-sa \
  --namespace default \
  --cluster <cluster-name> \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
  --approve

curl -o trtllm-engine-build.yaml \
  https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/yaml/trtllm-engine-build.yaml
kubectl apply -f trtllm-engine-build.yaml

kubectl wait --for=condition=complete job/trtllm-engine-build --timeout=60m
kubectl logs job/trtllm-engine-build
```

tensorrt 인퍼런스 서버를 배포한다.
```
curl -o https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/yaml/trtllm-qwen.yaml

kubectl apply -f trtllm-deployment.yaml
```


## 레퍼런스 ##
* https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html


