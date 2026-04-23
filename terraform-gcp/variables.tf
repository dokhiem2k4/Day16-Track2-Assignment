variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "hf_token" {
  description = "Hugging Face Token for gated models (like Gemma)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "model_id" {
  description = "Hugging Face Model ID to serve"
  type        = string
  default     = "google/gemma-4-E2B-it"
}

variable "machine_type" {
  description = "GCE Machine Type for the compute node"
  type        = string
  default     = "n2-standard-8"  # CPU fallback: 8 vCPU, 32 GB RAM
}

variable "gpu_type" {
  description = "GPU accelerator type (unused in CPU fallback)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach (0 = CPU only)"
  type        = number
  default     = 0
}
