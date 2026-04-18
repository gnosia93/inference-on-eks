output "vscode" {
  value = "http://${aws_instance.vscode.public_ip}:9090"
}

output "eks_cluster_name" {
  value = aws_eks_cluster.main.name
}

output "milvus_role_arn" {
  value = aws_iam_role.milvus.arn
}

output "vectordb_bucket_name" {
  value       = aws_s3_bucket.milvus.id
  description = "Milvus S3 bucket name"
}


#output "eks_cluster_endpoint" {
#  value = aws_eks_cluster.main.endpoint
#}

#output "karpenter_role_arn" {
#  value = aws_iam_role.karpenter_controller.arn
#}
