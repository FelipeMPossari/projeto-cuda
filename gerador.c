#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- Definições dos Estados Iniciais ---
// (Não podemos começar com MORTA)
#define SAUDAVEL 1     // Pessoa saudável [cite: 6]
#define CONTAMINADA -1 // Pessoa contaminada [cite: 5]
#define VAZIO 0        // Ninguém [cite: 8]

// --- Parâmetros de Geração (AJUSTE AQUI) ---
// Chance de uma célula ter uma pessoa (vs. ser VAZIO)
#define TAXA_POPULACAO 0.8
// Se for uma pessoa, chance de estar contaminada (vs. saudável)
#define TAXA_CONTAMINACAO 0.1

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Uso: %s <N_linhas> <M_colunas> <arquivo_saida>\n", argv[0]);
        printf("Exemplo: %s 100 100 entrada.txt\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    const char *nome_arquivo = argv[3];

    if (N <= 0 || M <= 0)
    {
        printf("Erro: N e M devem ser maiores que 0.\n");
        return 1;
    }

    // Inicializar o gerador aleatório
    srand(time(NULL));

    FILE *f = fopen(nome_arquivo, "w");
    if (f == NULL)
    {
        printf("Erro: Não foi possível criar o arquivo '%s'.\n", nome_arquivo);
        return 1;
    }

    // 1. Escrever N e M na primeira linha
    fprintf(f, "%d %d\n", N, M);

    // 2. Escrever as N linhas seguintes
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {

            int estado = VAZIO; // Padrão

            // Sorteia um número de 0.0 a 1.0
            double r_pop = (double)rand() / RAND_MAX;

            if (r_pop < TAXA_POPULACAO)
            {
                // Se é uma pessoa, decidimos o estado
                double r_cont = (double)rand() / RAND_MAX;

                if (r_cont < TAXA_CONTAMINACAO)
                {
                    estado = CONTAMINADA;
                }
                else
                {
                    estado = SAUDAVEL;
                }
            }

            fprintf(f, "%d ", estado);
        }
        fprintf(f, "\n"); // Próxima linha
    }

    fclose(f);

    printf("Arquivo '%s' (NxM: %dx%d) gerado com sucesso.\n", nome_arquivo, N, M);

    return 0;
}