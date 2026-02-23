#!/bin/bash
# MANUAL FILE TRANSFER SCRIPT
# If git pull fails, run these scp commands individually

echo "Transferring deployment files to Narval..."
echo ""

# Transfer DNABERT-2 fix
echo "1. Transferring DNABERT-2 fix..."
scp /Users/studio/Desktop/PhD/Proposal/deploy_package/sequence_encoder.py amird@narval.alliancecan.ca:~/chromaguide_experiments/src/chromaguide/
echo "   ✅ sequence_encoder.py transferred"
echo ""

# Transfer focal loss training script
echo "2. Transferring focal loss implementation..."
scp /Users/studio/Desktop/PhD/Proposal/deploy_package/train_off_target_focal.py amird@narval.alliancecan.ca:~/chromaguide_experiments/scripts/
echo "   ✅ train_off_target_focal.py transferred"
echo ""

# Transfer SLURM job scripts
echo "3. Transferring SLURM job scripts..."
scp /Users/studio/Desktop/PhD/Proposal/deploy_package/slurm_multimodal_dnabert2_fixed.sh amird@narval.alliancecan.ca:~/chromaguide_experiments/scripts/
echo "   ✅ slurm_multimodal_dnabert2_fixed.sh transferred"

scp /Users/studio/Desktop/PhD/Proposal/deploy_package/slurm_off_target_focal.sh amird@narval.alliancecan.ca:~/chromaguide_experiments/scripts/
echo "   ✅ slurm_off_target_focal.sh transferred"
echo ""

# Transfer validation script
echo "4. Transferring validation script..."
scp /Users/studio/Desktop/PhD/Proposal/deploy_package/test_dnabert2_fix.py amird@narval.alliancecan.ca:~/chromaguide_experiments/
echo "   ✅ test_dnabert2_fix.py transferred"
echo ""

# Transfer multimodal training script
echo "5. Transferring multimodal training script..."
scp /Users/studio/Desktop/PhD/Proposal/deploy_package/train_on_real_data_v2.py amird@narval.alliancecan.ca:~/chromaguide_experiments/scripts/
echo "   ✅ train_on_real_data_v2.py transferred"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "✅ All files transferred successfully!"
echo ""
echo "Now on Narval, run:"
echo "  cd ~/chromaguide_experiments"
echo "  source ~/env_chromaguide/bin/activate"
echo "  export PYTHONPATH=~/chromaguide_experiments/src:\$PYTHONPATH"
echo "  python test_dnabert2_fix.py"
echo "  sbatch scripts/slurm_multimodal_dnabert2_fixed.sh"
echo "  sbatch scripts/slurm_off_target_focal.sh"
