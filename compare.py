"""
Parts:
  A - function that solves Ax=b with scipy.optimize.root
  B - test it on random inputs and compare to numpy
  C - compare runtimes on sizes 1..1000 and plot
"""

import numpy as np
from scipy.optimize import root
from time import perf_counter
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Part A
# solve Ax=b using scipy.optimize.root
# -------------------------------------------------------

def solve_with_root(a, b):
    """
    Finds x such that Ax = b using scipy.optimize.root instead of numpy.
    Defines f(x) = Ax - b and finds x where f(x) = 0.
    Raises ValueError if the system has no solution.

    a - the matrix A (n x n)
    b - the right-hand side vector (length n)

    returns x such that Ax = b

    Examples
    --------
    >>> import numpy as np

    # diagonal system: 2*x=4, 3*y=9 -> x=2, y=3
    >>> a = np.array([[2.0, 0.0], [0.0, 3.0]])
    >>> b = np.array([4.0, 9.0])
    >>> np.allclose(solve_with_root(a, b), [2.0, 3.0])
    True

    # identity matrix - x should just equal b
    >>> a = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> b = np.array([5.0, 7.0])
    >>> np.allclose(solve_with_root(a, b), [5.0, 7.0])
    True

    # 1x1 case: 3*x = 9 -> x = 3
    >>> a = np.array([[3.0]])
    >>> b = np.array([9.0])
    >>> np.allclose(solve_with_root(a, b), [3.0])
    True

    # 10*x=1, 10*y=1 -> x=y=0.1
    >>> a = np.array([[10.0, 0.0], [0.0, 10.0]])
    >>> b = np.array([1.0, 1.0])
    >>> np.allclose(solve_with_root(a, b), [0.1, 0.1])
    True

    # 2x2 system: x + 2y = 5,  3x + 4y = 6
    # solving manually:
    #   from eq1:  x = 5 - 2y
    #   sub into eq2:  3(5-2y) + 4y = 6  =>  15 - 6y + 4y = 6  =>  -2y = -9  =>  y = 4.5
    #   x = 5 - 2*4.5 = -4
    #   solution: x=-4, y=4.5
    >>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = np.array([5.0, 6.0])
    >>> x = solve_with_root(a, b)
    >>> np.allclose(x, [-4.0, 4.5])
    True

    # 3x3 system: 4x + y = 1,  x + 3y + z = 2,  y + 5z = 3
    # solving manually:
    #   from eq1:  y = 1 - 4x
    #   sub into eq2:  x + 3(1-4x) + z = 2  =>  -11x + z = -1
    #   sub into eq3:  (1-4x) + 5z = 3      =>  -4x + 5z = 2
    #   from -11x + z = -1:  z = 11x - 1
    #   sub z into -4x + 5z = 2:  -4x + 5(11x-1) = 2  =>  51x = 7  =>  x = 7/51
    #   y = 1 - 4*(7/51) = 23/51
    #   z = 11*(7/51) - 1 = 26/51
    #   solution: x=7/51, y=23/51, z=26/51
    >>> a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
    >>> b = np.array([1.0, 2.0, 3.0])
    >>> x = solve_with_root(a, b)
    >>> np.allclose(x, [7/51, 23/51, 26/51])
    True

    # size 0 - no variables, no equations, solution is an empty array
    >>> a = np.array([]).reshape(0, 0)
    >>> b = np.array([])
    >>> list(solve_with_root(a, b))
    []

    # singular matrix - no solution exists
    # row 2 is exactly 2 * row 1, so the two equations are contradictory:
    #   x + 2y = 3  and  2x + 4y = 7  (7 != 2*3, so no solution)
    >>> a = np.array([[1.0, 2.0], [2.0, 4.0]])
    >>> b = np.array([3.0, 7.0])
    >>> solve_with_root(a, b)
    Traceback (most recent call last):
        ...
    ValueError: no solution found - matrix is singular or system is inconsistent
    """

    # start the search from the zero vector as our initial guess
    x0 = np.zeros(len(b))

    # find x such that Ax - b = 0, which means Ax = b
    result = root(lambda x: a @ x - b, x0=x0)

    # verify the result by plugging x back in - if Ax != b the system has no solution
    if not np.allclose(a @ result.x, b):
        raise ValueError("no solution found - matrix is singular or system is inconsistent")

    return result.x


# -------------------------------------------------------
# Part B
# test our function against numpy on random inputs
# -------------------------------------------------------

def check_correctness_vs_numpy(num_tests=100, max_size=100):
    """
    Runs random tests to check that solve_with_root agrees with numpy.linalg.solve.
    Each test passes if both solvers find matching solutions, or both fail on a singular matrix.

    num_tests - how many random systems to test
    max_size - maximum matrix size to use
    """

    np.random.seed(100)  # fix seed

    for _ in range(num_tests):
        n = np.random.randint(1, max_size + 1)

        # purely random matrix - might be singular
        a = np.random.randn(n, n)
        b = np.random.randn(n)

        # try solving with numpy - raises LinAlgError if singular
        try:
            x_numpy = np.linalg.solve(a, b)
            numpy_failed = False
        except np.linalg.LinAlgError:
            numpy_failed = True

        # try solving with solve_with_root - raises ValueError if singular
        try:
            x_scipy = solve_with_root(a, b)
            scipy_failed = False
        except ValueError:
            scipy_failed = True

        # case 1: both found a solution -> results must match
        if not numpy_failed and not scipy_failed:
            assert np.allclose(x_numpy, x_scipy, atol=1e-5), \
                f"solutions differ for size={n}"

        # case 2: both failed -> both agree the matrix is singular -> ok
        elif numpy_failed and scipy_failed:
            pass

        # case 3: only one failed -> they disagree -> something is wrong
        else:
            assert False, f"solvers disagree on singular/non-singular for size={n}"


# -------------------------------------------------------
# Part C
# compare runtimes over sizes 1..1000 and plot
# -------------------------------------------------------

def compare_performance(sizes, num_runs=1):
    """
    Times numpy.linalg.solve and solve_with_root on random systems of each given size.
    Returns the average runtimes as two lists (one per solver).

    sizes - list of matrix sizes to test
    num_runs - how many times to repeat each measurement (results are averaged)

    returns (numpy_times, scipy_times) - average runtime in seconds per size
    """

    numpy_times = []  # will hold one average time per size for numpy
    scipy_times = []  # same for scipy

    for n in sizes:
        np_total = 0.0  # accumulate total time over num_runs for numpy
        sp_total = 0.0  # accumulate total time over num_runs for scipy

        for _ in range(num_runs):
            # keep trying until a non-singular matrix is found for this size
            while True:
                a = np.random.randn(n, n)
                b = np.random.randn(n)
                try:
                    t = perf_counter()
                    np.linalg.solve(a, b)
                    np_total += perf_counter() - t

                    t = perf_counter()
                    solve_with_root(a, b)
                    sp_total += perf_counter() - t
                    break  # success - move on to next run
                except (np.linalg.LinAlgError, ValueError):
                    pass  # singular matrix - try again with a new one

        # divide by num_runs to get the average time for this size
        numpy_times.append(np_total / num_runs)
        scipy_times.append(sp_total / num_runs)

    return numpy_times, scipy_times


def plot_and_save(sizes, numpy_times, scipy_times, filename="comparison.png"):
    """
    Plots the runtime curves for both solvers and saves the figure to a PNG file.

    sizes - list of matrix sizes (x-axis)
    numpy_times - runtimes for numpy.linalg.solve
    scipy_times - runtimes for scipy.optimize.root
    filename - output file path
    """

    # create a figure with a wide aspect ratio so the x-axis has room to breathe
    plt.figure(figsize=(11, 6))

    # plot scipy line
    plt.plot(sizes, scipy_times, label="scipy.optimize.root", color="firebrick", linewidth=3)

    # plot numpy line
    plt.plot(sizes, numpy_times, label="numpy.linalg.solve", color="teal", linewidth=3)

    # label the x-axis so the reader knows what n means
    plt.xlabel("Matrix size n")

    # label the y-axis - units are seconds
    plt.ylabel("Average runtime (seconds)")

    # title at the top of the graph describing what we're comparing
    plt.title("numpy.linalg.solve vs scipy.optimize.root - runtime comparison")

    # show a legend so the reader knows which line is which
    plt.legend()

    # add a light grid to make it easier to read values off the chart
    plt.grid(True, alpha=0.4)

    # remove extra whitespace around the plot before saving
    plt.tight_layout()

    # save the figure to disk as a PNG file
    plt.savefig(filename, dpi=150)

    print(f"saved graph to {filename}")


# -------------------------------------------------------
# pytest entry points
# -------------------------------------------------------

def test_solve_with_root_examples():
    """
    Runs the doctests in solve_with_root to verify the examples are correct.
    """

    import doctest
    import sys

    results = doctest.testmod(sys.modules[__name__], verbose=False)
    assert results.failed == 0, f"{results.failed} doctest(s) failed"


def test_solve_with_root_random_inputs():
    """Checks that solve_with_root matches numpy.linalg.solve on 100 random matrices."""
    check_correctness_vs_numpy(num_tests=100, max_size=100)


# -------------------------------------------------------
# Main - run part B then part C
# -------------------------------------------------------

if __name__ == "__main__":

    # part B - run the random correctness tests
    print("=" * 50)
    print("Part B - correctness tests")
    print("=" * 50)
    try:
        check_correctness_vs_numpy(num_tests=100, max_size=100)
        print("all tests passed!")
    except AssertionError as e:
        print(f"tests FAILED: {e}")

    # part C - time both solvers on every size from 1 to 1000
    print("\n" + "=" * 50)
    print("Part C - performance comparison (sizes 1 to 1000)")
    print("=" * 50)

    # dense at the start (every size), then gradually sparser as n grows
    # small sizes are cheap so we can afford more points there,
    # large sizes take long with scipy so we measure fewer of them
    sizes = (
        list(range(1, 51, 1))        # 1..50   - every single size
        + list(range(51, 201, 5))    # 51..200 - every 5
        + list(range(201, 501, 20))  # 201..500 - every 20
        + list(range(501, 1001, 50)) # 501..950 - every 50
        + [1000]                     # always include 1000 exactly
    )

    np.random.seed(100)  # fix seed
    numpy_times, scipy_times = compare_performance(sizes, num_runs=1)

    plot_and_save(sizes, numpy_times, scipy_times, filename="comparison.png")
