#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// --- 1. Definições ---
#define SAUDAVEL 1
#define CONTAMINADA -1
#define MORTA -2
#define VAZIO 0
#define P_CURA 1000
#define P_CONTINUA 4000

int N, M; // Globais

// --- 2. Macro de Verificação de Erro CUDA ---
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s em %s na linha %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))

// --- 3. Funções de Kernel (GPU) ---
__global__ void setup_kernel(curandState *states, int N, int M, unsigned long seed)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < M)
    {
        int id = i * M + j;
        curand_init(seed + id, 0, 0, &states[id]);
    }
}

// --- KERNEL CORRIGIDO ---
// (Agora aceita os dois novos contadores)
__global__ void simular_iteracao_kernel(const int *grid_atual, int *grid_proximo,
                                        curandState *states, int N, int M,
                                        int *d_mortos_acumulados, int *d_vivos_iteracao)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M)
    {
        int id = i * M + j;
        int estado_atual = grid_atual[id];
        grid_proximo[id] = estado_atual;

        switch (estado_atual)
        {
        case SAUDAVEL:
            if (i > 0 && (grid_atual[(i - 1) * M + j] == CONTAMINADA || grid_atual[(i - 1) * M + j] == MORTA))
                grid_proximo[id] = CONTAMINADA;
            else if (i < N - 1 && (grid_atual[(i + 1) * M + j] == CONTAMINADA || grid_atual[(i + 1) * M + j] == MORTA))
                grid_proximo[id] = CONTAMINADA;
            else if (j > 0 && (grid_atual[i * M + (j - 1)] == CONTAMINADA || grid_atual[i * M + (j - 1)] == MORTA))
                grid_proximo[id] = CONTAMINADA;
            else if (j < M - 1 && (grid_atual[i * M + (j + 1)] == CONTAMINADA || grid_atual[i * M + (j + 1)] == MORTA))
                grid_proximo[id] = CONTAMINADA;
            break;

        case CONTAMINADA:
            unsigned int r = curand(&states[id]) % 10000;
            if (r < P_CURA)
                grid_proximo[id] = SAUDAVEL;
            else if (r < P_CONTINUA)
                grid_proximo[id] = CONTAMINADA;
            else
            {
                grid_proximo[id] = MORTA;
                // --- CORREÇÃO 1: Contagem de Mortos Atômica ---
                atomicAdd(d_mortos_acumulados, 1);
            }
            break;

        case MORTA:
            grid_proximo[id] = VAZIO;
            break;
        case VAZIO:
            break;
        }

        // --- CORREÇÃO 2: Contagem de Vivos Atômica (para parada antecipada) ---
        // Contamos *depois* da lógica, com base no estado da *próxima* iteração
        if (grid_proximo[id] == SAUDAVEL || grid_proximo[id] == CONTAMINADA)
        {
            atomicAdd(d_vivos_iteracao, 1);
        }
    }
}

// --- 4. Funções Auxiliares (Host) ---
void ler_entrada_gpu(const char *nome_arquivo, int *grid)
{
    FILE *f = fopen(nome_arquivo, "r");
    if (f == NULL)
    {
        exit(1);
    }
    fscanf(f, "%d %d", &N, &M); // Lê N e M
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            fscanf(f, "%d", &grid[i * M + j]);
        }
    }
    fclose(f);
}

void escrever_saida(const char *nome_arquivo, int total_mortos, int sobreviventes)
{
    FILE *f = fopen(nome_arquivo, "w");
    if (f == NULL)
    {
        exit(1);
    }
    fprintf(f, "Total de Mortos: %d\n", total_mortos);
    fprintf(f, "Sobreviventes (Saudáveis + Contaminados): %d\n", sobreviventes);
    fclose(f);
}

// --- 5. Main (GPU com Seletor de Teste) ---
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Uso: %s <id_teste 2-8> <arquivo_entrada> <arquivo_saida>\n", argv[0]);
        return 1;
    }

    int teste_id = atoi(argv[1]);
    const char *arq_entrada = argv[2];
    const char *arq_saida = argv[3];

    if (teste_id < 2 || teste_id > 8)
    {
        printf("Erro: ID de teste inválido (%d). Use um número de 2 a 8.\n", teste_id);
        return 1;
    }

    printf("Executando Teste %d (GPU)...\n", teste_id);

    FILE *f_temp = fopen(arq_entrada, "r");
    if (f_temp == NULL)
    {
        exit(1);
    }
    fscanf(f_temp, "%d %d", &N, &M);
    fclose(f_temp);

    size_t grid_bytes = N * M * sizeof(int);

    int *h_grid_atual = (int *)malloc(grid_bytes);
    ler_entrada_gpu(arq_entrada, h_grid_atual);

    // --- Alocação Device (CORRIGIDO) ---
    int *d_grid_atual, *d_grid_proximo;
    curandState *d_rand_states;
    int *d_mortos_acumulados; // Nosso contador de mortos
    int *d_vivos_iteracao;    // Nosso contador de vivos por iteração

    CUDA_CHECK(cudaMalloc(&d_grid_atual, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_grid_proximo, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_rand_states, N * M * sizeof(curandState)));
    // Aloca memória para os contadores
    CUDA_CHECK(cudaMalloc(&d_mortos_acumulados, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vivos_iteracao, sizeof(int)));

    // Inicializa o contador de mortos em 0
    CUDA_CHECK(cudaMemset(d_mortos_acumulados, 0, sizeof(int)));

    // --- Switch de Configuração de Teste (Sem mudanças) ---
    dim3 threadsPerBlock;
    dim3 numBlocks;
    dim3 nThreads(16, 16);

    switch (teste_id)
    {
    case 2:
        threadsPerBlock = dim3(1, 1);
        numBlocks = dim3(1, 1);
        break;
    case 3:
        threadsPerBlock = nThreads;
        numBlocks = dim3(1, 1);
        break;
    case 4:
        threadsPerBlock = nThreads;
        numBlocks = dim3(2, 1);
        break;
    case 5:
        threadsPerBlock = nThreads;
        numBlocks = dim3(2, 2);
        break;
    case 6:
        threadsPerBlock = nThreads;
        numBlocks = dim3(4, 2);
        break;
    case 7:
        threadsPerBlock = dim3(1, 1);
        numBlocks = dim3(M, N);
        break;
    case 8:
        threadsPerBlock = nThreads;
        numBlocks = dim3((M + nThreads.x - 1) / nThreads.x, (N + nThreads.y - 1) / nThreads.y);
        break;
    }
    printf("Configuração: Blocos(%d, %d), Threads(%d, %d)\n",
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // --- Simulação (CORRIGIDA) ---
    setup_kernel<<<numBlocks, threadsPerBlock>>>(d_rand_states, N, M, (unsigned long)time(NULL));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_grid_atual, h_grid_atual, grid_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t inicio, fim;
    CUDA_CHECK(cudaEventCreate(&inicio));
    CUDA_CHECK(cudaEventCreate(&fim));
    CUDA_CHECK(cudaEventRecord(inicio));

    int max_iteracoes = N * M;
    int h_vivos_iteracao = 0; // Cópia local (host) do contador de vivos

    // Feedback de progresso
    int throttle_update = (max_iteracoes >= 100) ? max_iteracoes / 100 : 1;
    if (throttle_update == 0)
        throttle_update = 1;

    for (int i = 0; i < max_iteracoes; i++)
    {
        // Reseta o contador de *vivos* para 0 *a cada* iteração
        CUDA_CHECK(cudaMemset(d_vivos_iteracao, 0, sizeof(int)));

        // Lança o kernel, passando os ponteiros dos contadores
        simular_iteracao_kernel<<<numBlocks, threadsPerBlock>>>(
            d_grid_atual, d_grid_proximo, d_rand_states, N, M,
            d_mortos_acumulados, d_vivos_iteracao);

        CUDA_CHECK(cudaGetLastError());

        // Copia o resultado (apenas 1 int) de vivos de volta para a CPU
        CUDA_CHECK(cudaMemcpy(&h_vivos_iteracao, d_vivos_iteracao, sizeof(int), cudaMemcpyDeviceToHost));

        int *temp = d_grid_atual;
        d_grid_atual = d_grid_proximo;
        d_grid_proximo = temp;

        // --- Feedback e Parada Antecipada ---
        if ((i + 1) % throttle_update == 0 || (i + 1) == max_iteracoes)
        {
            int percent = (int)(((double)(i + 1) / max_iteracoes) * 100);
            printf("\rProgresso: %d%% (%d / %d iterações) | Vivos: %d", percent, i + 1, max_iteracoes, h_vivos_iteracao);
            fflush(stdout);
        }

        if (h_vivos_iteracao == 0)
        {
            printf("\rProgresso: 100%% (parada antecipada na iteração %d)\n", i + 1);
            break;
        }
    }
    printf("\n"); // Nova linha após o progresso

    CUDA_CHECK(cudaEventRecord(fim));
    CUDA_CHECK(cudaEventSynchronize(fim));
    float tempo_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tempo_ms, inicio, fim));
    printf("Tempo de execução (GPU): %f ms (%f segundos)\n", tempo_ms, tempo_ms / 1000.0);

    // --- Coleta de Resultados (CORRIGIDA) ---
    CUDA_CHECK(cudaMemcpy(h_grid_atual, d_grid_atual, grid_bytes, cudaMemcpyDeviceToHost));

    int h_total_mortos; // Cópia local (host) do contador de mortos
    // Copia o contador de mortos (só 1 int) de volta para a CPU
    CUDA_CHECK(cudaMemcpy(&h_total_mortos, d_mortos_acumulados, sizeof(int), cudaMemcpyDeviceToHost));

    int sobreviventes = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            int estado = h_grid_atual[i * M + j];
            if (estado == SAUDAVEL || estado == CONTAMINADA)
                sobreviventes++;
            // Não contamos mais os mortos aqui
        }
    }

    // Usa a contagem atômica da GPU
    escrever_saida(arq_saida, h_total_mortos, sobreviventes);

    // --- Limpeza (CORRIGIDA) ---
    free(h_grid_atual);
    CUDA_CHECK(cudaFree(d_grid_atual));
    CUDA_CHECK(cudaFree(d_grid_proximo));
    CUDA_CHECK(cudaFree(d_rand_states));
    CUDA_CHECK(cudaFree(d_mortos_acumulados)); // Libera os novos contadores
    CUDA_CHECK(cudaFree(d_vivos_iteracao));    // Libera os novos contadores
    CUDA_CHECK(cudaEventDestroy(inicio));
    CUDA_CHECK(cudaEventDestroy(fim));

    printf("Simulação GPU concluída. Resultados em '%s'.\n", arq_saida);
    return 0;
}