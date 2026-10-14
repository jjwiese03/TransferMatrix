#include <complex>

extern "C" {
    struct data {
        const std::complex<double>* P; int Pn0, Pn1, Pn2, Pn3;

        // modes
        const int ML;
        const std::complex<double>* modes; int n0, n1, n2, n3;

        // coords
        const double* R;   int R_rows,  R_cols;
        const double* kR;  int kR_rows, kR_cols;
        const double* Phi; int Phi_rows, Phi_cols;

        const bool* diskmaskin;   // gleiche Größe wie R, siehe unten
        const bool* diskmaskout;
        int mask_rows, mask_cols;

    };

    std::complex<double> propagationCoeffs(const data* d, double* f, int nf, double* dist, double* tiltx, double* tilty, int nd, double eps) {
        // Julia ist column-major: Element (i,j) mit 0-basiertem Index
        // R[i + j * R_rows]

        std::complex<double> z(2.0, 3.0);
        return tiltx[9] + z;
    }
}