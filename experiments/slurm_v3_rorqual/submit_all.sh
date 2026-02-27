#!/bin/bash
# Submit all 45 ChromaGuide v3 experiments

sbatch cg3_cnn_gru_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_cnn_gru_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_caduceus_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_dnabert2_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_evo_sC_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sA_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sA_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sA_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sB_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sB_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sB_f0_r456.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sC_f0_r42.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sC_f0_r123.sh
sleep 1  # Avoid rate limiting
sbatch cg3_nucleotide_transformer_sC_f0_r456.sh
sleep 1  # Avoid rate limiting

echo 'Submitted 45 jobs'
