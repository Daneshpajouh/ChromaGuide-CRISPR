#!/bin/bash
# Submit all 45 ChromaGuide v4 experiments on fir

sbatch cg4_cnn_gru_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_cnn_gru_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_caduceus_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_dnabert2_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_evo_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg4_nucleotide_transformer_sC_f0_r456.sh
sleep 1  # Avoid rate limiting

echo 'Submitted 45 jobs'
