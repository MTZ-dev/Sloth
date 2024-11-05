# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

from cython cimport boundscheck, wraparound

cimport numpy as cnp
from scipy.linalg.cython_blas cimport zaxpy, zgemm, zhemm, zdotc, caxpy, cgemm, chemm, cdotc
from scipy.linalg.cython_lapack cimport zheevr, cheevr

# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

##################
# Double precision
##################


@boundscheck(False)
@wraparound(False)
def _zheevr_lwork(int n, jobz ='N', range='A', int il=1, int iu=0, cnp.float64_t vl=-1e308, cnp.float64_t vu=1e308):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int lda = n
        int ldz = n
        int m
        int info
        cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] a
        cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] z
        cnp.ndarray[cnp.float64_t, ndim=1] w
        cnp.ndarray[cnp.complex128_t, ndim=1] work
        int lwork
        cnp.ndarray[cnp.float64_t, ndim=1] rwork
        int lrwork
        cnp.ndarray[int, ndim=1] iwork
        int liwork
        cnp.ndarray[int, ndim=1] isuppz
        cnp.npy_intp *dims = [n, n]
        cnp.float64_t abs_tol = -1

    a = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX128, 1)
    w = cnp.PyArray_EMPTY(1, [n], cnp.NPY_FLOAT64, 1)
    if range == 'A':
        isuppz = cnp.PyArray_EMPTY(1, [2*n], cnp.NPY_INT32, 1)
    elif range == 'I':
        if iu == 0:
            iu = n
        isuppz = cnp.PyArray_EMPTY(1, [2*(iu-il+1)], cnp.NPY_INT32, 1)
    else:
        isuppz = cnp.PyArray_EMPTY(1, [1], cnp.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            if iu == 0:
                iu = n
            z = cnp.PyArray_EMPTY(2, [n, iu-il+1], cnp.NPY_COMPLEX128, 1)
        else:
            z = cnp.PyArray_EMPTY(2, [n, n], cnp.NPY_COMPLEX128, 1)
    else: 
        z = cnp.PyArray_EMPTY(2, [1, 1], cnp.NPY_COMPLEX128, 1)

    lwork = -1
    lrwork = -1
    liwork = -1
    work = cnp.PyArray_EMPTY(1, [1], cnp.NPY_COMPLEX128, 1)
    rwork = cnp.PyArray_EMPTY(1, [1], cnp.NPY_FLOAT64, 1)
    iwork = cnp.PyArray_EMPTY(1, [1], cnp.NPY_INT32, 1)
    
    with nogil:
        zheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)
 
    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")

    lwork = int(work[0].real)
    lrwork = int(rwork[0])
    liwork = int(iwork[0])
    
    return lwork, lrwork, liwork


@boundscheck(False)
@wraparound(False)
def _zheevr(cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] a, int lwork, int lrwork, int liwork, jobz='N', range='A', int il=1, int iu=0, cnp.float64_t vl=-1e30, cnp.float64_t vu=0):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int n = a.shape[0]
        int lda = n
        int ldz = n
        int m
        int info
        cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] z
        cnp.ndarray[cnp.float64_t, ndim=1] w
        cnp.ndarray[cnp.complex128_t, ndim=1] work
        cnp.ndarray[cnp.float64_t, ndim=1] rwork
        cnp.ndarray[int, ndim=1] iwork
        cnp.ndarray[int, ndim=1] isuppz
        cnp.float64_t abs_tol = -1

    if a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix must be square")

    w = cnp.PyArray_EMPTY(1, [n], cnp.NPY_FLOAT64, 1)
    if range == 'A':
        isuppz = cnp.PyArray_EMPTY(1, [2*n], cnp.NPY_INT32, 1)
    elif range == 'I':
        if iu == 0:
            iu = n
        isuppz = cnp.PyArray_EMPTY(1, [2*(iu-il+1)], cnp.NPY_INT32, 1)
    else:
        isuppz = cnp.PyArray_EMPTY(1, [1], cnp.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            if iu == 0:
                iu = n
            z = cnp.PyArray_EMPTY(2, [n, iu-il+1], cnp.NPY_COMPLEX128, 1)
        else:
            z = cnp.PyArray_EMPTY(2, [n, n], cnp.NPY_COMPLEX128, 1)
    else:
        z = cnp.PyArray_EMPTY(2, [1, 1], cnp.NPY_COMPLEX128, 1)
    
    work = cnp.PyArray_EMPTY(1, [lwork], cnp.NPY_COMPLEX128, 1)
    rwork = cnp.PyArray_EMPTY(1, [lrwork], cnp.NPY_FLOAT64, 1)
    iwork = cnp.PyArray_EMPTY(1, [liwork], cnp.NPY_INT32, 1)

    with nogil:
        zheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)

    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")
    
    if jobz == 'V':
        return w[:m], z[:, :m]
    else:
        return w[:m]


@boundscheck(False)
@wraparound(False)
def _zutmu(cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] U,
                          cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] M) -> cnp.ndarray[cnp.complex128_t]:
    cdef:
        int m = U.shape[0]
        int n = U.shape[1]
        cnp.npy_intp *dims = [m, n]
        cnp.npy_intp *dim = [n, n]
        cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] C = cnp.PyArray_EMPTY(2, dim, cnp.NPY_COMPLEX128, 1)
        cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] B = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX128, 1)
        cnp.complex128_t alpha = 1.0 + 0.0j
        cnp.complex128_t beta = 0.0 + 0.0j
        char side = b'L'[0]
        char uplo = b'L'[0]
        char transa = b'C'[0]
        char transb = b'N'[0]

    with nogil:
        zhemm(&side, &uplo, &m, &n, &alpha, &M[0, 0], &m, &U[0, 0], &m, &beta, &B[0, 0], &m)
        zgemm(&transa, &transb, &n, &n, &m, &alpha, &U[0, 0], &m, &B[0, 0], &m, &beta, &C[0, 0], &n)

    return C


@boundscheck(False)
@wraparound(False)
def _zutmud(cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] U,
                          cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] M) -> cnp.ndarray[cnp.float64_t]:
    cdef:
        int m = U.shape[0]
        int n = U.shape[1]
        cnp.npy_intp *dims = [m, n]
        cnp.npy_intp *dim = [n]
        cnp.ndarray[cnp.float64_t, ndim=1, mode="fortran"] C_diag = cnp.PyArray_EMPTY(1, dim, cnp.NPY_FLOAT64, 1)
        cnp.ndarray[cnp.complex128_t, ndim=2, mode="fortran"] B = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX128, 1)
        int i
        cnp.complex128_t alpha = 1.0 + 0.0j
        cnp.complex128_t beta = 0.0 + 0.0j
        int one = 1
        char side = b'L'[0]
        char uplo = b'L'[0]

    with nogil:
        zhemm(&side, &uplo, &m, &n, &alpha, &M[0, 0], &m, &U[0, 0], &m, &beta, &B[0, 0], &m)
        for i in range(n):
                C_diag[i] = zdotc(&n, &U[0, i], &one, &B[0, i], &one).real

    return C_diag


@boundscheck(False)
@wraparound(False)
def _zdot3d(cnp.ndarray[cnp.complex128_t, ndim=3] M, cnp.ndarray[cnp.float64_t, ndim=1] V) -> cnp.ndarray[cnp.complex128_t]:
    cdef:
        int m = M.shape[1]
        int n = M.shape[2]
        cnp.npy_intp *dims = [m, n]
        cnp.ndarray[cnp.complex128_t, ndim=2] result = cnp.PyArray_ZEROS(2, dims, cnp.NPY_COMPLEX128, 0)
        int one = 1
        cnp.complex128_t alpha0 = V[0] + 0.0j
        cnp.complex128_t alpha1 = V[1] + 0.0j
        cnp.complex128_t alpha2 = V[2] + 0.0j
        int size = m * n

    with nogil:
        zaxpy(&size, &alpha0, &M[0, 0, 0], &one, &result[0, 0], &one)
        zaxpy(&size, &alpha1, &M[1, 0, 0], &one, &result[0, 0], &one)
        zaxpy(&size, &alpha2, &M[2, 0, 0], &one, &result[0, 0], &one)

    return result


@boundscheck(False)
@wraparound(False)
def _zgemmt(cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] M, cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] N) -> cnp.ndarray[cnp.complex128_t]:
    cdef:
        int m = M.shape[1]
        int n = N.shape[1]
        int k = M.shape[0]
        cnp.npy_intp *dims = [m, n]
        cnp.ndarray[cnp.complex128_t, ndim=2, mode='fortran'] A = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX128, 1)
        cnp.complex128_t alpha = 1.0 + 0.0j
        cnp.complex128_t beta = 0.0 + 0.0j
        char transa = b'T'[0]
        char transb = b'N'[0]
        int lda = M.shape[0]
        int ldb = N.shape[0]
        int ldc = A.shape[0]

    with nogil:
        zgemm(&transa, &transb, &m, &n, &k, &alpha, &M[0, 0], &lda, &N[0, 0], &ldb, &beta, &A[0, 0], &ldc)

    return A


##################
# Single precision
##################


@boundscheck(False)
@wraparound(False)
def _cheevr_lwork(int n, jobz='N', range='A', int il=1, int iu=0, cnp.float32_t vl=-1e38, cnp.float32_t vu=1e38):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int lda = n
        int ldz = n
        int m
        int info
        cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] a
        cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] z
        cnp.ndarray[cnp.float32_t, ndim=1] w
        cnp.ndarray[cnp.complex64_t, ndim=1] work
        int lwork
        cnp.ndarray[cnp.float32_t, ndim=1] rwork
        int lrwork
        cnp.ndarray[int, ndim=1] iwork
        int liwork
        cnp.ndarray[int, ndim=1] isuppz
        cnp.npy_intp *dims = [n, n]
        cnp.float32_t abs_tol = -1

    a = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX64, 1)
    w = cnp.PyArray_EMPTY(1, [n], cnp.NPY_FLOAT32, 1)
    if range == 'A':
        isuppz = cnp.PyArray_EMPTY(1, [2*n], cnp.NPY_INT32, 1)
    elif range == 'I':
        if iu == 0:
            iu = n
        isuppz = cnp.PyArray_EMPTY(1, [2*(iu-il+1)], cnp.NPY_INT32, 1)
    else:
        isuppz = cnp.PyArray_EMPTY(1, [1], cnp.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            if iu == 0:
                iu = n
            z = cnp.PyArray_EMPTY(2, [n, iu-il+1], cnp.NPY_COMPLEX64, 1)
        else:
            z = cnp.PyArray_EMPTY(2, [n, n], cnp.NPY_COMPLEX64, 1)
    else:
        z = cnp.PyArray_EMPTY(2, [1, 1], cnp.NPY_COMPLEX64, 1)
    
    lwork = -1
    lrwork = -1
    liwork = -1
    work = cnp.PyArray_EMPTY(1, [1], cnp.NPY_COMPLEX64, 1)
    rwork = cnp.PyArray_EMPTY(1, [1], cnp.NPY_FLOAT32, 1)
    iwork = cnp.PyArray_EMPTY(1, [1], cnp.NPY_INT32, 1)
    
    with nogil:
        cheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)
       
    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")
    
    lwork = int(work[0].real)
    lrwork = int(rwork[0])
    liwork = int(iwork[0])
    
    return lwork, lrwork, liwork


@boundscheck(False)
@wraparound(False)
def _cheevr(cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] a, int lwork, int lrwork, int liwork, jobz='N', range='A', int il=1, int iu=0, cnp.float32_t vl=-1e30, cnp.float32_t vu=0):
    cdef:
        char jobz_char = jobz.encode('ascii')[0]
        char range_char = range.encode('ascii')[0]
        char uplo = b'L'[0]
        int n = a.shape[0]
        int lda = n
        int ldz = n
        int m
        int info
        cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] z
        cnp.ndarray[cnp.float32_t, ndim=1] w
        cnp.ndarray[cnp.complex64_t, ndim=1] work
        cnp.ndarray[cnp.float32_t, ndim=1] rwork
        cnp.ndarray[int, ndim=1] iwork
        cnp.ndarray[int, ndim=1] isuppz
        cnp.float32_t abs_tol = -1

    if a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix must be square")


    w = cnp.PyArray_EMPTY(1, [n], cnp.NPY_FLOAT32, 1)
    if range == 'A':
        isuppz = cnp.PyArray_EMPTY(1, [2*n], cnp.NPY_INT32, 1)
    elif range == 'I':
        if iu == 0:
            iu = n
        isuppz = cnp.PyArray_EMPTY(1, [2*(iu-il+1)], cnp.NPY_INT32, 1)
    else:
        isuppz = cnp.PyArray_EMPTY(1, [1], cnp.NPY_INT32, 1)
    if jobz == 'V':
        if range == 'I':
            if iu == 0:
                iu = n
            z = cnp.PyArray_EMPTY(2, [n, iu-il+1], cnp.NPY_COMPLEX64, 1)
        else:
            z = cnp.PyArray_EMPTY(2, [n, n], cnp.NPY_COMPLEX64, 1)
    else:
        z = cnp.PyArray_EMPTY(2, [1, 1], cnp.NPY_COMPLEX64, 1)
    
    work = cnp.PyArray_EMPTY(1, [lwork], cnp.NPY_COMPLEX64, 1)
    rwork = cnp.PyArray_EMPTY(1, [lrwork], cnp.NPY_FLOAT32, 1)
    iwork = cnp.PyArray_EMPTY(1, [liwork], cnp.NPY_INT32, 1)

    with nogil:
        cheevr(&jobz_char, &range_char, &uplo, &n, &a[0,0], &lda, &vl, &vu, &il, &iu, &abs_tol, &m, &w[0], &z[0,0], &ldz, &isuppz[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info)

    if info != 0:
        raise ValueError(f"Error in LAPACK zheevr: info={info}")
        
    if jobz == 'V':
        return w[:m], z[:, :m]
    else:
        return w[:m]


@boundscheck(False)
@wraparound(False)
def _cutmu(cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] U,
                          cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] M) -> cnp.ndarray[cnp.complex64_t]:
    cdef:
        int m = U.shape[0]
        int n = U.shape[1]
        cnp.npy_intp *dims = [m, n]
        cnp.npy_intp *dim = [n, n]
        cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] C = cnp.PyArray_EMPTY(2, dim, cnp.NPY_COMPLEX64, 1)
        cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] B = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX64, 1)
        cnp.complex64_t alpha = 1.0 + 0.0j
        cnp.complex64_t beta = 0.0 + 0.0j
        char side = b'L'[0]
        char uplo = b'L'[0]
        char transa = b'C'[0]
        char transb = b'N'[0]

    with nogil:
        chemm(&side, &uplo, &m, &n, &alpha, &M[0, 0], &m, &U[0, 0], &m, &beta, &B[0, 0], &m)
        cgemm(&transa, &transb, &n, &n, &m, &alpha, &U[0, 0], &m, &B[0, 0], &m, &beta, &C[0, 0], &n)

    return C


@boundscheck(False)
@wraparound(False)
def _cutmud(cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] U,
                          cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] M) -> cnp.ndarray[cnp.float32_t]:
    cdef:
        int m = U.shape[0]
        int n = U.shape[1]
        cnp.npy_intp *dims = [m, n]
        cnp.npy_intp *dim = [n]
        cnp.ndarray[cnp.float32_t, ndim=1, mode="fortran"] C_diag = cnp.PyArray_EMPTY(1, dim, cnp.NPY_FLOAT32, 1)
        cnp.ndarray[cnp.complex64_t, ndim=2, mode="fortran"] B = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX64, 1)
        int i
        cnp.complex64_t alpha = 1.0 + 0.0j
        cnp.complex64_t beta = 0.0 + 0.0j
        int one = 1
        char side = b'L'[0]
        char uplo = b'L'[0]

    with nogil:
        chemm(&side, &uplo, &m, &n, &alpha, &M[0, 0], &m, &U[0, 0], &m, &beta, &B[0, 0], &m)
        for i in range(n):
                C_diag[i] = cdotc(&n, &U[0, i], &one, &B[0, i], &one).real

    return C_diag


@boundscheck(False)
@wraparound(False)
def _cdot3d(cnp.ndarray[cnp.complex64_t, ndim=3] M, cnp.ndarray[cnp.float32_t, ndim=1] V) -> cnp.ndarray[cnp.complex64_t]:
    cdef:
        int m = M.shape[1]
        int n = M.shape[2]
        cnp.npy_intp *dims = [m, n]
        cnp.ndarray[cnp.complex64_t, ndim=2] result = cnp.PyArray_ZEROS(2, dims, cnp.NPY_COMPLEX64, 0)
        int one = 1
        cnp.complex64_t alpha0 = V[0] + 0.0j
        cnp.complex64_t alpha1 = V[1] + 0.0j
        cnp.complex64_t alpha2 = V[2] + 0.0j
        int size = m * n

    with nogil:
        caxpy(&size, &alpha0, &M[0, 0, 0], &one, &result[0, 0], &one)
        caxpy(&size, &alpha1, &M[1, 0, 0], &one, &result[0, 0], &one)
        caxpy(&size, &alpha2, &M[2, 0, 0], &one, &result[0, 0], &one)

    return result


@boundscheck(False)
@wraparound(False)
def _cgemmt(cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] M, cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] N) -> cnp.ndarray[cnp.complex64_t]:
    cdef:
        int m = M.shape[1]
        int n = N.shape[1]
        int k = M.shape[0]
        cnp.npy_intp *dims = [m, n]
        cnp.ndarray[cnp.complex64_t, ndim=2, mode='fortran'] A = cnp.PyArray_EMPTY(2, dims, cnp.NPY_COMPLEX64, 1)
        cnp.complex64_t alpha = 1.0 + 0.0j
        cnp.complex64_t beta = 0.0 + 0.0j
        char transa = b'T'[0]
        char transb = b'N'[0]
        int lda = M.shape[0]
        int ldb = N.shape[0]
        int ldc = A.shape[0]

    with nogil:
        cgemm(&transa, &transb, &m, &n, &k, &alpha, &M[0, 0], &lda, &N[0, 0], &ldb, &beta, &A[0, 0], &ldc)

    return A