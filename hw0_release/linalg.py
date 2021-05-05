import numpy as np

M = np.zeros(shape=(4, 3))
k = 1
for i in range(4):
    for j in range(3):
        M[i][j] = k
        k += 1
a = np.array([[1, 1, 0]])
b = np.array([[-1, 2, 5]]).T
M_EIGEN = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test0():
    print("M = \n", M)
    print("The size of M is: ", M.shape)
    print()
    print("a = ", a)
    print("The size of a is: ", a.shape)
    print()
    print("b = ", b)
    print("The size of b is: ", b.shape)


def test1():
    aDotB = dot_product(a, b)
    print(aDotB)
    print("The size is: ", aDotB.shape)


def test2():
    # Your answer should be $[[3], [9], [15], [21]]$ of shape(4, 1).
    ans = complicated_matrix_function(M, a, b)
    print(ans)
    print()
    print("The size is: ", ans.shape)
    print("-----Test 2.1----")
    M_2 = np.array(range(4)).reshape((2, 2))
    a_2 = np.array([[1, 1]])
    b_2 = np.array([[10, 10]]).T
    ans = complicated_matrix_function(M_2, a_2, b_2)
    print(ans)
    print()
    print("The size is: ", ans.shape)


def test3():
    only_first_singular_value = get_singular_values(M, 1)
    print(only_first_singular_value)
    first_two_singular_values = get_singular_values(M, 2)
    print(first_two_singular_values)
    assert only_first_singular_value[0] == first_two_singular_values[0]


def test4(M):
    val, vec = get_eigen_values_and_vectors(M[:, :3], 1)
    print("First eigenvalue =", val[0])
    print()
    print("First eigenvector =", vec[0])
    print()
    assert len(vec) == 1
    val, vec = get_eigen_values_and_vectors(M[:,:3], 2)
    print("Eigenvalues =", val)
    print()
    print("Eigenvectors =", vec)
    assert len(vec) == 2

def myFun():
    # test0()
    # test1()
    # test2()
    # test3()
    test4(M_EIGEN)

def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    out = None
    ### YOUR CODE HERE
    out = np.dot(a, b)
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    out = None
    aDotb = dot_product(a, b)
    mDota = dot_product(M, a.T)
    out = dot_product(mDota, aDotb)

    return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u = None
    s = None
    v = None
    u, s, v = np.linalg.svd(M)
    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = None
    ### YOUR CODE HERE
    u, s, v = svd(M)
    ### END YOUR CODE
    return s[:k]


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    w, v = np.linalg.eig(M)
    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    w, v = eigen_decomp(M)
    eigenvalues = w[:k]
    eigenvectors = v[:k]
    return eigenvalues, eigenvectors


def main():
    myFun()


if __name__ == "__main__":
    main()
