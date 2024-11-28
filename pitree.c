

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>

// Structure for a tree node
typedef struct Node {
    int digit;             // Digit of π
    struct Node* left;     // Left child
    struct Node* right;    // Right child
    double partial_sum;    // Partial sum of BBP terms
} Node;

// Function to create a new node
Node* createNode(int digit, double partial_sum) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    newNode->digit = digit;
    newNode->left = NULL;
    newNode->right = NULL;
    newNode->partial_sum = partial_sum;
    return newNode;
}

// Function to free the tree
void freeTree(Node* root) {
    if (root == NULL) return;
    freeTree(root->left);
    freeTree(root->right);
    free(root);
}

// Function to multiply two large numbers using FFT
void multiplyUsingFFT(double* a, double* b, int n) {
    fftw_complex *inA, *inB, *outA, *outB, *outC;
    fftw_plan planA, planB, planC;

    inA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    inB = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    outA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    outB = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    outC = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    planA = fftw_plan_dft_1d(n, inA, outA, FFTW_FORWARD, FFTW_ESTIMATE);
    planB = fftw_plan_dft_1d(n, inB, outB, FFTW_FORWARD, FFTW_ESTIMATE);
    planC = fftw_plan_dft_1d(n, outC, inA, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int i = 0; i < n; i++) {
        inA[i][0] = a[i];
        inA[i][1] = 0.0;
        inB[i][0] = b[i];
        inB[i][1] = 0.0;
    }

    fftw_execute(planA);
    fftw_execute(planB);

    for (int i = 0; i < n; i++) {
        outC[i][0] = outA[i][0] * outB[i][0] - outA[i][1] * outB[i][1];
        outC[i][1] = outA[i][0] * outB[i][1] + outA[i][1] * outB[i][0];
    }

    fftw_execute(planC);

    for (int i = 0; i < n; i++) {
        a[i] = inA[i][0] / n;
    }

    fftw_destroy_plan(planA);
    fftw_destroy_plan(planB);
    fftw_destroy_plan(planC);
    fftw_free(inA);
    fftw_free(inB);
    fftw_free(outA);
    fftw_free(outB);
    fftw_free(outC);
}

// Function to calculate the n-th digit of π using the BBP formula
int calculatePiDigit(int n) {
    double sum = 0.0;
    for (int k = 0; k <= n; k++) {
        double term = (4.0 / (8 * k + 1) - 2.0 / (8 * k + 4) - 1.0 / (8 * k + 5) - 1.0 / (8 * k + 6)) / pow(16, k);
        sum += term;
    }
    return (int)(sum * pow(16, n)) % 16;
}

// Helper function to create a node with FFT multiplication
Node* createNodeWithFFT(int digit, double term, int n) {
    double* a = (double*)malloc(sizeof(double) * (n + 1));
    double* b = (double*)malloc(sizeof(double) * (n + 1));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j <= n; j++) {
        a[j] = (j == 0) ? 1.0 : 0.0;
        b[j] = (j == n) ? term : 0.0;
    }
    multiplyUsingFFT(a, b, n + 1);
    double partial_sum = 0.0;
    for (int j = 0; j <= n; j++) {
        partial_sum += a[j];
    }
    Node* newNode = createNode(digit, partial_sum);
    free(a);
    free(b);
    return newNode;
}

// Function to build the tree up to a certain level with parallelization and FFT
Node* buildPiTree(int level) {
    if (level == 0) {
        return createNode(calculatePiDigit(0), 0.0);
    }

    Node* root = createNode(calculatePiDigit(0), 0.0);
    Node* current = root;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 1; i <= level; i++) {
                #pragma omp task
                {
                    double term = calculatePiDigit(i) / pow(16, i);
                    current->left = createNodeWithFFT(calculatePiDigit(i), term, i);
                }
                #pragma omp task
                {
                    double term = calculatePiDigit(i + 1) / pow(16, i + 1);
                    current->right = createNodeWithFFT(calculatePiDigit(i + 1), term, i + 1);
                }
                #pragma omp taskwait
                current = current->left;
            }
        }
    }

    return root;
}

// Function to find the n-th digit of π in the tree
int findPiDigit(Node* root, int n) {
    Node* current = root;
    for (int i = 0; i < n; i++) {
        if (current->left == NULL || current->right == NULL) {
            // If the tree is not deep enough, extend the tree
            #pragma omp parallel
            {
                #pragma omp single
                {
                    #pragma omp task
                    {
                        double term = calculatePiDigit(i + 1) / pow(16, i + 1);
                        current->left = createNodeWithFFT(calculatePiDigit(i + 1), term, i + 1);
                    }
                    #pragma omp task
                    {
                        double term = calculatePiDigit(i + 2) / pow(16, i + 2);
                        current->right = createNodeWithFFT(calculatePiDigit(i + 2), term, i + 2);
                    }
                    #pragma omp taskwait
                }
            }
        }
        current = current->left;
    }
    return current->digit;
}

int main() {
    int level = 5; // Depth level of the tree
    Node* root = buildPiTree(level);

    int n = 10; // Digit of π we want to find
    int digit = findPiDigit(root, n);

    printf("The %d-th digit of π is: %d\n", n, digit);

    // Free memory
    freeTree(root);

    return 0;
}
