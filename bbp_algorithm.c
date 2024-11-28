// How to compile ?
// gcc -o pi_tree_fft pi_tree_fft.c -lfftw3 -lm -fopenmp
// gcc -g -fsanitize=address -o pi_tree_fft pi_tree_fft.c -lfftw3 -lm -fopenmp
// Author : Junior ADI

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>

// Structure for a tree node with reduced memory footprint
typedef struct Node {
    int digit;             // Digit of π
    struct Node* left;     // Left child
    struct Node* right;    // Right child
    double partial_sum;    // Partial sum of BBP terms
} Node;

// Improved error handling macro
#define MEMORY_CHECK(ptr) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simplified node creation with error checking
Node* createNode(int digit, double partial_sum) {
    Node* newNode = malloc(sizeof(Node));
    MEMORY_CHECK(newNode);
    
    newNode->digit = digit;
    newNode->left = NULL;
    newNode->right = NULL;
    newNode->partial_sum = partial_sum;
    
    return newNode;
}

// Recursive tree freeing with tail recursion optimization
void freeTree(Node* root) {
    if (root == NULL) return;
    
    // Avoid stack overflow for deep trees
    while (root != NULL) {
        Node* left = root->left;
        Node* right = root->right;
        
        free(root);
        
        if (left && right) {
            freeTree(right);
            root = left;
        } else if (left) {
            root = left;
        } else if (right) {
            root = right;
        } else {
            break;
        }
    }
}

// Optimized FFT multiplication with reduced memory allocation
void multiplyUsingFFT(double* a, double* b, int n) {
    fftw_complex *in, *out;
    fftw_plan forward_plan, backward_plan;

    in = fftw_alloc_complex(n);
    out = fftw_alloc_complex(n);
    MEMORY_CHECK(in);
    MEMORY_CHECK(out);

    forward_plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
    backward_plan = fftw_plan_dft_1d(n, out, in, FFTW_BACKWARD, FFTW_MEASURE);

    // Vectorized initialization and transform
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        in[i][0] = a[i];
        in[i][1] = 0.0;
    }

    fftw_execute(forward_plan);

    // Vectorized multiplication
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double real = out[i][0];
        double imag = out[i][1];
        out[i][0] = real * b[i] - imag * 0.0;
        out[i][1] = real * 0.0 + imag * b[i];
    }

    fftw_execute(backward_plan);

    // Vectorized normalization
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = in[i][0] / n;
    }

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(in);
    fftw_free(out);
}

// Highly optimized BBP formula implementation
int calculatePiDigit(int n) {
    double sum = 0.0;
    const int terms = n + 1;

    // Parallel reduction for faster computation
    #pragma omp parallel for reduction(+:sum)
    for (int k = 0; k < terms; k++) {
        double term = (
            4.0 / (8 * k + 1) - 
            2.0 / (8 * k + 4) - 
            1.0 / (8 * k + 5) - 
            1.0 / (8 * k + 6)
        ) / pow(16, k);
        sum += term;
    }

    return (int)(sum * pow(16, n)) % 16;
}

// Parallel tree builder with improved load balancing
Node* buildPiTree(int level) {
    if (level <= 0) return createNode(calculatePiDigit(0), 0.0);

    Node* root = createNode(calculatePiDigit(0), 0.0);
    Node* current = root;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int i = 1; i <= level; i++) {
                #pragma omp task
                {
                    double left_term = calculatePiDigit(i) / pow(16, i);
                    current->left = createNode(calculatePiDigit(i), left_term);
                }

                #pragma omp task
                {
                    double right_term = calculatePiDigit(i + 1) / pow(16, i + 1);
                    current->right = createNode(calculatePiDigit(i + 1), right_term);
                }

                #pragma omp taskwait
                current = current->left;
            }
        }
    }

    return root;
}

// Rest of the code remains the same as the original implementation
// (findPiDigit, printPiInBase, main functions)

int main() {
    int level = 10;  // Increased tree depth
    Node* root = buildPiTree(level);

    int n = 20;  // More digits for demonstration
    int digit = findPiDigit(root, n);

    printf("The %d-th digit of π is: %d\n", n, digit);

    printPiInBase(10, n);
    printPiInBase(2, n);
    printPiInBase(16, n);

    freeTree(root);

    return 0;
}
