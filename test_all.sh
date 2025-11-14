#!/bin/bash

# Script para testar todos os 7 casos de teste (2-8) e comparar GPU vs CPU
# Autor: Script de teste automatizado
# Data: 2025-11-14

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Arquivos
ENTRADA="entrada.txt"
SAIDA_GPU="saida_gpu.txt"
SAIDA_CPU="saida_cpu.txt"
RESULTADOS="resultados_comparacao2.txt"

# Verifica se os executáveis existem
if [ ! -f "medaocu" ]; then
    echo -e "${RED}Erro: Executável 'medaocu' (GPU) não encontrado!${NC}"
    echo "Compile com: nvcc -o medaocu gpu.cu"
    exit 1
fi

if [ ! -f "cpu" ]; then
    echo -e "${YELLOW}Aviso: Executável 'cpu' não encontrado. Tentando compilar...${NC}"
    gcc -o cpu cpu.c -lm
    if [ $? -ne 0 ]; then
        echo -e "${RED}Erro ao compilar cpu.c${NC}"
        exit 1
    fi
    echo -e "${GREEN}CPU compilado com sucesso!${NC}"
fi

if [ ! -f "$ENTRADA" ]; then
    echo -e "${RED}Erro: Arquivo de entrada '$ENTRADA' não encontrado!${NC}"
    exit 1
fi

# Limpa arquivo de resultados anterior
> "$RESULTADOS"

# Header do relatório
cat << EOF | tee "$RESULTADOS"
╔════════════════════════════════════════════════════════════════════════╗
║                TESTE COMPARATIVO: GPU vs CPU                           ║
║                  Simulação de Epidemia                                 ║
╚════════════════════════════════════════════════════════════════════════╝

Data/Hora: $(date '+%Y-%m-%d %H:%M:%S')
Arquivo de Entrada: $ENTRADA

EOF

# Extrai N e M do arquivo de entrada
read N M < "$ENTRADA"
echo -e "${CYAN}Dimensões da Grade: ${N} x ${M} = $((N*M)) células${NC}\n" | tee -a "$RESULTADOS"

# Arrays para armazenar tempos
declare -a tempos_gpu
declare -a tempos_cpu
declare -a speedups
declare -a nomes_testes

# Descrições dos testes
declare -A desc_testes
desc_testes[2]="1 bloco, 1 thread"
desc_testes[3]="1 bloco, 16x16 threads"
desc_testes[4]="2x1 blocos, 16x16 threads"
desc_testes[5]="2x2 blocos, 16x16 threads"
desc_testes[6]="4x2 blocos, 16x16 threads"
desc_testes[7]="MxN blocos, 1 thread/bloco"
desc_testes[8]="Grade completa, 16x16 threads/bloco"

echo -e "═══════════════════════════════════════════════════════════════════════" | tee -a "$RESULTADOS"
echo -e "                         EXECUÇÃO DOS TESTES                            " | tee -a "$RESULTADOS"
echo -e "═══════════════════════════════════════════════════════════════════════\n" | tee -a "$RESULTADOS"

# Primeiro, executa o teste na CPU (referência)
echo -e "${BOLD}${BLUE}► Executando teste na CPU (referência)...${NC}" | tee -a "$RESULTADOS"
echo "" | tee -a "$RESULTADOS"

# Captura a saída e extrai o tempo
cpu_output=$(./cpu "$ENTRADA" "$SAIDA_CPU" 2>&1)
tempo_cpu=$(echo "$cpu_output" | grep -oP "Tempo de execução \(CPU\): \K[0-9.]+")

if [ -z "$tempo_cpu" ]; then
    echo -e "${RED}Erro ao executar teste CPU${NC}" | tee -a "$RESULTADOS"
    exit 1
fi

tempo_cpu_ms=$(echo "$tempo_cpu * 1000" | bc)
echo "$cpu_output" | tee -a "$RESULTADOS"
echo "" | tee -a "$RESULTADOS"
echo -e "${GREEN}✓ CPU completado: ${tempo_cpu}s (${tempo_cpu_ms}ms)${NC}\n" | tee -a "$RESULTADOS"

# Loop através dos testes 2-8
for teste_id in {2..8}; do
    echo -e "───────────────────────────────────────────────────────────────────────" | tee -a "$RESULTADOS"
    echo -e "${BOLD}${CYAN}► Teste ${teste_id}: ${desc_testes[$teste_id]}${NC}" | tee -a "$RESULTADOS"
    echo "" | tee -a "$RESULTADOS"
    
    # Executa o teste GPU
    gpu_output=$(./medaocu "$teste_id" "$ENTRADA" "$SAIDA_GPU" 2>&1)
    
    # Extrai o tempo em ms do output da GPU
    tempo_gpu_ms=$(echo "$gpu_output" | grep -oP "Tempo de Execução: \K[0-9.]+(?= ms)")
    
    if [ -z "$tempo_gpu_ms" ]; then
        echo -e "${RED}Erro ao executar teste GPU ${teste_id}${NC}" | tee -a "$RESULTADOS"
        continue
    fi
    
    tempo_gpu_s=$(echo "scale=4; $tempo_gpu_ms / 1000" | bc)
    
    # Calcula speedup
    speedup=$(echo "scale=2; $tempo_cpu_ms / $tempo_gpu_ms" | bc)
    
    # Armazena resultados
    tempos_gpu+=("$tempo_gpu_ms")
    tempos_cpu+=("$tempo_cpu_ms")
    speedups+=("$speedup")
    nomes_testes+=("Teste $teste_id")
    
    # Mostra output resumido
    echo "$gpu_output" | grep -A 20 -- "--- Configuração da GPU ---" | tee -a "$RESULTADOS"
    echo "" | tee -a "$RESULTADOS"
    
    # Resultado comparativo
    if (( $(echo "$speedup > 1" | bc -l) )); then
        cor=$GREEN
        simbolo="⚡"
    else
        cor=$RED
        simbolo="⚠"
    fi
    
    echo -e "${BOLD}Resultados:${NC}" | tee -a "$RESULTADOS"
    echo -e "  GPU: ${tempo_gpu_ms} ms (${tempo_gpu_s}s)" | tee -a "$RESULTADOS"
    echo -e "  CPU: ${tempo_cpu_ms} ms (${tempo_cpu}s)" | tee -a "$RESULTADOS"
    echo -e "  ${cor}${simbolo} Speedup: ${speedup}x${NC}" | tee -a "$RESULTADOS"
    echo "" | tee -a "$RESULTADOS"
done

# Tabela resumo
echo -e "\n═══════════════════════════════════════════════════════════════════════" | tee -a "$RESULTADOS"
echo -e "                         TABELA COMPARATIVA                             " | tee -a "$RESULTADOS"
echo -e "═══════════════════════════════════════════════════════════════════════\n" | tee -a "$RESULTADOS"

printf "%-15s | %-15s | %-15s | %-10s\n" "Teste" "GPU (ms)" "CPU (ms)" "Speedup" | tee -a "$RESULTADOS"
echo "────────────────────────────────────────────────────────────────────────" | tee -a "$RESULTADOS"

for i in {0..6}; do
    teste_id=$((i + 2))
    printf "%-15s | %15.2f | %15.2f | %9.2fx\n" \
        "Teste $teste_id" \
        "${tempos_gpu[$i]}" \
        "$tempo_cpu_ms" \
        "${speedups[$i]}" | tee -a "$RESULTADOS"
done

echo "────────────────────────────────────────────────────────────────────────" | tee -a "$RESULTADOS"

# Calcula médias
soma_gpu=0
soma_speedup=0
for i in {0..6}; do
    soma_gpu=$(echo "$soma_gpu + ${tempos_gpu[$i]}" | bc)
    soma_speedup=$(echo "$soma_speedup + ${speedups[$i]}" | bc)
done

media_gpu=$(echo "scale=2; $soma_gpu / 7" | bc)
media_speedup=$(echo "scale=2; $soma_speedup / 7" | bc)

printf "%-15s | %15.2f | %15.2f | %9.2fx\n" \
    "MÉDIA" \
    "$media_gpu" \
    "$tempo_cpu_ms" \
    "$media_speedup" | tee -a "$RESULTADOS"

echo "" | tee -a "$RESULTADOS"

# Encontra melhor e pior teste
melhor_idx=0
pior_idx=0
melhor_speedup=${speedups[0]}
pior_speedup=${speedups[0]}

for i in {1..6}; do
    if (( $(echo "${speedups[$i]} > $melhor_speedup" | bc -l) )); then
        melhor_speedup=${speedups[$i]}
        melhor_idx=$i
    fi
    if (( $(echo "${speedups[$i]} < $pior_speedup" | bc -l) )); then
        pior_speedup=${speedups[$i]}
        pior_idx=$i
    fi
done

melhor_teste=$((melhor_idx + 2))
pior_teste=$((pior_idx + 2))

# Análise final
cat << EOF | tee -a "$RESULTADOS"

═══════════════════════════════════════════════════════════════════════
                              ANÁLISE                                   
═══════════════════════════════════════════════════════════════════════

📊 Estatísticas:
   • Tempo médio GPU: ${media_gpu} ms
   • Tempo CPU (ref): ${tempo_cpu_ms} ms
   • Speedup médio: ${media_speedup}x

🏆 Melhor desempenho:
   • Teste ${melhor_teste}: ${desc_testes[$melhor_teste]}
   • Speedup: ${melhor_speedup}x
   • Tempo: ${tempos_gpu[$melhor_idx]} ms

⚠️  Pior desempenho:
   • Teste ${pior_teste}: ${desc_testes[$pior_teste]}
   • Speedup: ${pior_speedup}x
   • Tempo: ${tempos_gpu[$pior_idx]} ms

📝 Observações:
   • Todos os resultados foram salvos em: $RESULTADOS
   • Saída GPU detalhada: $SAIDA_GPU
   • Saída CPU detalhada: $SAIDA_CPU

═══════════════════════════════════════════════════════════════════════

EOF

# Cria gráfico ASCII simples
echo "Gráfico de Speedup:" | tee -a "$RESULTADOS"
echo "" | tee -a "$RESULTADOS"

for i in {0..6}; do
    teste_id=$((i + 2))
    speedup=${speedups[$i]}
    # Converte speedup para barras (cada # = 0.5x)
    bars=$(echo "scale=0; $speedup * 2 / 1" | bc)
    bar_string=$(printf '█%.0s' $(seq 1 $bars))
    
    printf "Teste %d: %s %.2fx\n" "$teste_id" "$bar_string" "$speedup" | tee -a "$RESULTADOS"
done

echo "" | tee -a "$RESULTADOS"
echo -e "${GREEN}${BOLD}✅ Todos os testes concluídos!${NC}"
echo -e "Relatório completo salvo em: ${BOLD}$RESULTADOS${NC}\n"
