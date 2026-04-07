
```bash
kubectl apply -f https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/yaml/trtllm-engine-build.yaml

kubectl wait --for=condition=complete job/trtllm-engine-build --timeout=60m

kubectl logs job/trtllm-engine-build

kubectl apply -f trtllm-deployment.yaml


https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/yaml/trtllm-qwen.yaml
```

```
# ServiceAccount에 S3 접근 권한 부여
eksctl create iamserviceaccount \
  --name s3-access-sa \
  --namespace default \
  --cluster <cluster-name> \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
  --approve

```
