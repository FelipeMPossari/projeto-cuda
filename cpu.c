#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- 1. Definições ---
#define SAUDAVEL 1
#define CONTAMINADA -1
#define MORTA -2
#define VAZIO 0
#define P_CURA 1000
#define P_CONTINUA 4000

int N, M;
int g_total_mortos_acumulados = 0;

// --- 2. Funções Auxiliares ---

int **alocar_grid()
{
    int **grid;
    grid = (int **)malloc(N * sizeof(int *));
    if (grid == NULL)
    {
        printf("Erro: Falha na alocação de memória para as linhas.\n");
        exit(1);
    }
    for (int i = 0; i < N; i++)
    {
        grid[i] = (int *)malloc(M * sizeof(int));
        if (grid[i] == NULL)
        {
            printf("Erro: Falha na alocação de memória para as colunas.\n");
            exit(1);
        }
    }
    return grid;
}

void liberar_grid(int **grid)
{
    for (int i = 0; i < N; i++)
    {
        free(grid[i]);
    }
    free(grid);
}

void ler_entrada(const char *nome_arquivo, int **grid)
{
    FILE *f = fopen(nome_arquivo, "r");
    if (f == NULL)
    {
        printf("Erro: Não foi possível abrir o arquivo de entrada '%s'.\n", nome_arquivo);
        exit(1);
    }

    // N e M já foram lidos no main, então pulamos a primeira linha
    fscanf(f, "%d %d", &N, &M);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            fscanf(f, "%d", &grid[i][j]);
        }
    }
    fclose(f);
}

void escrever_saida(const char *nome_arquivo, int total_mortos, int sobreviventes)
{
    FILE *f = fopen(nome_arquivo, "w");
    if (f == NULL)
    {
        printf("Erro: Não foi possível criar o arquivo de saída '%s'.\n", nome_arquivo);
        exit(1);
    }
    fprintf(f, "Total de Mortos: %d\n", total_mortos);
    fprintf(f, "Sobreviventes (Saudáveis + Contaminados): %d\n", sobreviventes);
    fclose(f);
}

// --- 3. Lógica da Simulação (CORRIGIDA) ---

int simular_iteracao(int **grid_atual, int **grid_proximo)
{
    int vivos = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            int estado_atual = grid_atual[i][j];
            grid_proximo[i][j] = estado_atual;

            switch (estado_atual)
            {
            case SAUDAVEL:
                if (i > 0 && (grid_atual[i - 1][j] == CONTAMINADA || grid_atual[i - 1][j] == MORTA))
                    grid_proximo[i][j] = CONTAMINADA;
                else if (i < N - 1 && (grid_atual[i + 1][j] == CONTAMINADA || grid_atual[i + 1][j] == MORTA))
                    grid_proximo[i][j] = CONTAMINADA;
                else if (j > 0 && (grid_atual[i][j - 1] == CONTAMINADA || grid_atual[i][j - 1] == MORTA))
                    grid_proximo[i][j] = CONTAMINADA;
                else if (j < M - 1 && (grid_atual[i][j + 1] == CONTAMINADA || grid_atual[i][j + 1] == MORTA))
                    grid_proximo[i][j] = CONTAMINADA;

                vivos++;
                break;

            case CONTAMINADA:
                int r = rand() % 10000;
                if (r < P_CURA)
                    grid_proximo[i][j] = SAUDAVEL;
                else if (r < P_CONTINUA)
                    grid_proximo[i][j] = CONTAMINADA;
                else
                {
                    grid_proximo[i][j] = MORTA;
                    g_total_mortos_acumulados++;
                }
                vivos++;
                break;

            case MORTA:
                grid_proximo[i][j] = VAZIO;
                break;

            case VAZIO:
                break;
            }
        }
    }
    if (vivos == 0)
        return 1;
    return 0;
}

// --- 4. Função Principal (CORRIGIDA) ---

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Uso: %s <arquivo_entrada> <arquivo_saida>\n", argv[0]);
        return 1;
    }

    const char *arq_entrada = argv[1];
    const char *arq_saida = argv[2];

    srand(time(NULL));

    // --- Leitura Inicial (Apenas N e M) ---
    FILE *f_temp = fopen(arq_entrada, "r");
    if (f_temp == NULL)
    {
        printf("Erro: Não foi possível abrir %s para ler N e M.\n", arq_entrada);
        return 1;
    }
    fscanf(f_temp, "%d %d", &N, &M);
    fclose(f_temp);

    if (N <= 0 || M <= 0)
    {
        printf("Erro: N e M devem ser maiores que 0.\n");
        return 1;
    }

    // --- Alocação ---
    int **grid_atual = alocar_grid();
    int **grid_proximo = alocar_grid();

    // --- Leitura Completa ---
    ler_entrada(arq_entrada, grid_atual);

    // --- Simulação ---
    int max_iteracoes = N * M;
    int parar = 0;

    clock_t inicio = clock();

    // Imprime uma mensagem inicial
    printf("Iniciando simulação (CPU) para %d iterações (mapa %dx%d).\n", max_iteracoes, N, M);

    int throttle_update = (max_iteracoes >= 100) ? max_iteracoes / 100 : 1;
    if (throttle_update == 0)
        throttle_update = 1;

    for (int i = 0; i < max_iteracoes; i++)
    {
        parar = simular_iteracao(grid_atual, grid_proximo);

        int **temp = grid_atual;
        grid_atual = grid_proximo;
        grid_proximo = temp;

        if ((i + 1) % throttle_update == 0 || (i + 1) == max_iteracoes)
        {
            int percent = (int)(((double)(i + 1) / max_iteracoes) * 100);
            printf("\rProgresso: %d%% (%d / %d iterações)", percent, i + 1, max_iteracoes);
            fflush(stdout);
        }

        if (parar)
        {
            printf("\rProgresso: 100%% (parada antecipada na iteração %d)\n", i + 1);
            break;
        }
    }

    printf("\n");

    clock_t fim = clock();
    double tempo_cpu = ((double)(fim - inicio)) / CLOCKS_PER_SEC;
    printf("Tempo de execução (CPU): %f segundos\n", tempo_cpu);

    int sobreviventes = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (grid_atual[i][j] == SAUDAVEL || grid_atual[i][j] == CONTAMINADA)
                sobreviventes++;
        }
    }

    // --- CORREÇÃO APLICADA ---
    // Usa o contador global de mortos acumulados
    escrever_saida(arq_saida, g_total_mortos_acumulados, sobreviventes);

    // --- Limpeza ---
    liberar_grid(grid_atual);
    liberar_grid(grid_proximo);

    printf("Simulação concluída. Resultados em '%s'.\n", arq_saida);
    return 0;
}