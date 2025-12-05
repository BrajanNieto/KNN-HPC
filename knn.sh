#!/bin/bash
#SBATCH --job-name=knn-sweep
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/knn_%A_%a.out
#SBATCH --error=logs/knn_%A_%a.err

set -euo pipefail
mkdir -p logs

# ================== ==================
# Tamaño de entrenamiento (constante en el sweep)
TRAIN_SIZE=${TRAIN_SIZE:-640000}

# Tamaños del set de TEST (Q):
TEST_SIZES=(1000 2000 40000 8000)

N_JOBS_OPTS=(1 2 4 8 16 32)

# (Opcional) k y d (num_features) por si quieres sobreescribir:
K_VAL=${K_VAL:-3}
D_VAL=${D_VAL:-10}
# =====================================================

# ---- Módulos ----
module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6
module load python3/3.10.2

# Activar el entorno virtual con mpi4py, numpy, sklearn, etc.
source ~/venvs/py310_mpi/bin/activate

# ---- Mapeo del SLURM_ARRAY ----
NS=${#TEST_SIZES[@]}
NP=${#N_JOBS_OPTS[@]}
TOTAL=$(( NS * NP ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] Lanza con --array=0-$((TOTAL-1))"
  echo "Ej: sbatch --array=0-$((TOTAL-1)) knn.sh"
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} fuera de rango (0..$((TOTAL-1)))"
  exit 1
fi

i=$(( SLURM_ARRAY_TASK_ID / NP ))   # índice de TEST_SIZES
j=$(( SLURM_ARRAY_TASK_ID % NP ))   # índice de N_JOBS_OPTS

Q=${TEST_SIZES[$i]}
NPROCS=${N_JOBS_OPTS[$j]}

echo "[INFO] JobID=${SLURM_JOB_ID} ArrayIdx=${SLURM_ARRAY_TASK_ID}  -> Q=${Q}  NPROCS=${NPROCS}"
echo "[INFO] Nodes=${SLURM_NODELIST}  Reservado ntasks=${SLURM_NTASKS}  (usaremos mpirun -np ${NPROCS})"

# Seguridad: no pedir más ranks de los reservados
if (( NPROCS > SLURM_NTASKS )); then
  echo "[ERROR] NPROCS=${NPROCS} excede ntasks reservados (${SLURM_NTASKS}). Ajusta --ntasks en #SBATCH."
  exit 1
fi

LOGFILE="logs/knn_Q${Q}_N${NPROCS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

# Pasamos: train_size, test_size, k, num_features
mpirun -np "${NPROCS}" python3 "knn_mpi.py" "${TRAIN_SIZE}" "${Q}" "${K_VAL}" "${D_VAL}" 2>&1 | tee "${LOGFILE}"

echo "[INFO] Finalizado: Q=${Q}, NPROCS=${NPROCS} | Log: ${LOGFILE}"