// How to compile ?
// gcc -o pi_tree_fft pi_tree_fft.c -lfftw3 -lm -fopenmp
// gcc -g -fsanitize=address -o pi_tree_fft pi_tree_fft.c -lfftw3 -lm -fopenmp
// Author : Junior ADI

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

    // Check that the size is a power of 2 for FFT
    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "FFT size must be a power of 2.\n");
        exit(EXIT_FAILURE);
    }

    inA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    inB = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    outA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    outB = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    outC = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    if (!inA || !inB || !outA || !outB || !outC) {
        fprintf(stderr, "Memory allocation failed for FFT buffers.\n");
        exit(EXIT_FAILURE);
    }

    planA = fftw_plan_dft_1d(n, inA, outA, FFTW_FORWARD, FFTW_ESTIMATE);
    planB = fftw_plan_dft_1d(n, inB, outB, FFTW_FORWARD, FFTW_ESTIMATE);
    planC = fftw_plan_dft_1d(n, outC, inA, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Initialize the inputs
    for (int i = 0; i < n; i++) {
        inA[i][0] = a[i];
        inA[i][1] = 0.0;
        inB[i][0] = b[i];
        inB[i][1] = 0.0;
    }

    // Execute the FFT
    fftw_execute(planA);
    fftw_execute(planB);

    // Multiply the FFT results
    for (int i = 0; i < n; i++) {
        outC[i][0] = outA[i][0] * outB[i][0] - outA[i][1] * outB[i][1];
        outC[i][1] = outA[i][0] * outB[i][1] + outA[i][1] * outB[i][0];
    }

    // Inverse FFT
    fftw_execute(planC);

    // Normalize and copy the result into `a`
    for (int i = 0; i < n; i++) {
        a[i] = inA[i][0] / n;
    }

    // Clean up FFT allocations
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

// Function to build the tree up to a certain level
Node* buildPiTree(int level) {
    if (level < 0) return NULL;

    Node* root = createNode(calculatePiDigit(0), 0.0);
    if (level == 0) return root;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 1; i <= level; i++) {
                #pragma omp task
                {
                    root->left = createNode(calculatePiDigit(i), 0.0);
                    root->right = createNode(calculatePiDigit(i + 1), 0.0);
                }
                #pragma omp taskwait
            }
        }
    }

    return root;
}

// Function to find the n-th digit of π in the tree
int findPiDigit(Node* root, int n) {
    Node* current = root;
    for (int i = 0; i < n && current != NULL; i++) {
        current = (i % 2 == 0) ? current->left : current->right;
    }
    return current ? current->digit : -1;
}

int main() {
    int level = 5; // Example depth of the tree
    Node* root = buildPiTree(level);

    int n = 11; // Example digit to find
    int digit = findPiDigit(root, n);
    if (digit != -1) {
        printf("The %d-th digit of π is: %d\n", n, digit);
    } else {
        printf("Digit not found in the tree.\n");
    }

    freeTree(root);
    return 0;
}
