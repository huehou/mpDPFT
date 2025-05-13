import numpy as np
from scipy.interpolate import RBFInterpolator
import argparse
import sys # Added for explicit flushing

def rbf_interpolate_to_grid(input_filename, output_filename, n):
    """
    Reads 2D scattered data, performs RBF interpolation on a regular nxn grid,
    and writes the interpolated values to a space-delimited text file.

    Args:
        input_filename (str): Path to the input text file containing scattered data.
                              Each row should be "x y f" (space-delimited).
        output_filename (str): Path to the output text file where interpolated
                               grid data will be written. Each row will be
                               "x_grid y_grid f_interpolated" (space-delimited).
        n (int): The size of the regular grid (nxn).
    """
    try:
        # 1. Read scattered data (x, y, f) from the input file
        try:
            scattered_data = np.loadtxt(input_filename)
        except Exception as e:
            print(f"---mpScript_rbf.py: Error: Could not read or parse input file '{input_filename}'. Ensure it's a valid text file. Details: {e}", flush=True, file=sys.stderr)
            return False # Indicate failure

        if scattered_data.ndim == 0: # Handles empty file or non-numeric data leading to empty array
             print(f"---mpScript_rbf.py: Error: Input file '{input_filename}' is empty or contains no valid numeric data.", flush=True, file=sys.stderr)
             return False
        if scattered_data.ndim == 1: # Handle case with only one data point or malformed single line
            if scattered_data.shape[0] == 3:
                scattered_data = scattered_data.reshape(1, 3)
            else:
                print(f"---mpScript_rbf.py: Error: Input file '{input_filename}' has a single line that does not contain 3 columns (x, y, f). Shape is {scattered_data.shape}.", flush=True, file=sys.stderr)
                return False
        elif scattered_data.shape[1] != 3:
            print(f"---mpScript_rbf.py: Error: Input file '{input_filename}' must have 3 columns (x, y, f). Detected shape: {scattered_data.shape}.", flush=True, file=sys.stderr)
            return False

        x_scatter = scattered_data[:, 0]
        y_scatter = scattered_data[:, 1]
        f_scatter = scattered_data[:, 2]

        if x_scatter.size == 0: # Should be caught by ndim checks, but as a safeguard
            print(f"---mpScript_rbf.py: Error: No data points found in the input file '{input_filename}'.", flush=True, file=sys.stderr)
            return False

        # RBFInterpolator typically needs at least D+1 points for a D-dimensional space for some kernels,
        # or more for stability depending on the kernel and data.
        # For 2D, this means at least 3 points for some configurations.
        if x_scatter.size < 3 and n > 1 :
            print(f"---mpScript_rbf.py: Warning: Performing RBF interpolation with only {x_scatter.size} scattered points. Results might be unreliable or fail depending on the RBF kernel and parameters.", flush=True, file=sys.stderr)


        # 2. Create the RBF interpolator
        points_scatter = np.column_stack((x_scatter, y_scatter))
        try:
            # Using 'thin_plate_spline' as it's a common and robust choice.
            # Epsilon is not directly used by thin_plate_spline in the same way as for Gaussian.
            # Smoothing can be added if data is noisy: interpolator = RBFInterpolator(points_scatter, f_scatter, kernel='thin_plate_spline', smoothing=0.0)
            rbf_interpolator = RBFInterpolator(points_scatter, f_scatter, kernel='thin_plate_spline')
        except Exception as e:
            print(f"---mpScript_rbf.py: Error: Failed to create RBFInterpolator. This can happen with too few points or collinear/coincident points. Details: {e}", flush=True, file=sys.stderr)
            return False

        # 3. Define the regular grid
        if x_scatter.size == 1: # If only one point, create a small grid around it.
             x_min, x_max = x_scatter[0] - 0.5, x_scatter[0] + 0.5
             y_min, y_max = y_scatter[0] - 0.5, y_scatter[0] + 0.5
        else:
            x_min, x_max = np.min(x_scatter), np.max(x_scatter)
            y_min, y_max = np.min(y_scatter), np.max(y_scatter)

        # Handle cases where min and max are the same (e.g., all points on a line or a single unique x/y)
        # This prevents linspace from creating an array with a single value if n > 1
        if x_min == x_max:
            if n > 1:
                print(f"---mpScript_rbf.py: Warning: All scattered X coordinates are identical ({x_min}). Grid will have constant X.", flush=True, file=sys.stderr)
            # For linspace to work as expected across a "range", ensure max > min if n > 1
            # If n=1, it doesn't matter. If n > 1, we need a small range.
            x_max = x_min + 1e-6 if n > 1 else x_min # Add a tiny offset if multiple grid points needed
        if y_min == y_max:
            if n > 1:
                print(f"---mpScript_rbf.py: Warning: All scattered Y coordinates are identical ({y_min}). Grid will have constant Y.", flush=True, file=sys.stderr)
            y_max = y_min + 1e-6 if n > 1 else y_min

        grid_x_coords = np.linspace(x_min, x_max, n)
        grid_y_coords = np.linspace(y_min, y_max, n)
        X_grid, Y_grid = np.meshgrid(grid_x_coords, grid_y_coords)

        # Prepare grid points for interpolation
        points_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

        # 4. Interpolate on the grid
        try:
            F_interpolated_flat = rbf_interpolator(points_grid)
        except Exception as e:
            print(f"---mpScript_rbf.py: Error: RBF interpolation failed for grid points. Details: {e}", flush=True, file=sys.stderr)
            return False

        # 5. Prepare the output data - Fortran-style flatten
        Xf = X_grid.ravel(order='F')
        Yf = Y_grid.ravel(order='F')
        Ff = F_interpolated_flat.reshape((ny,nx), order='F').ravel(order='F')
        output_data = np.column_stack((Xf, Yf, Ff))

        # 6. Write to the output file
        try:
            np.savetxt(output_filename, output_data, fmt='%g %g %g', delimiter=' ')
            print(f"---mpScript_rbf.py: Interpolated data successfully written to {output_filename}", flush=True)
            return True # Indicate success
        except Exception as e:
            print(f"---mpScript_rbf.py: Error: Could not write output to file '{output_filename}'. Details: {e}", flush=True, file=sys.stderr)
            return False # Indicate failure

    except FileNotFoundError: # This specific exception for input_filename should be caught by np.loadtxt or earlier os.path checks if added.
        print(f"---mpScript_rbf.py: Error: Input file '{input_filename}' not found.", flush=True, file=sys.stderr)
        return False
    except ValueError as ve: # Catches potential errors from np operations if not handled above
        print(f"---mpScript_rbf.py: ValueError during script execution: {ve}", flush=True, file=sys.stderr)
        return False
    except Exception as e: # General catch-all for unexpected errors
        print(f"---mpScript_rbf.py: An unexpected error occurred: {e}", flush=True, file=sys.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform RBF interpolation on 2D scattered data and output to a regular grid.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""The script will print status messages to standard output and errors to standard error. Exit code will be 0 on success, 1 on failure."""
    )
    parser.add_argument("input_file", help="Path to the input text file containing scattered data (x y f).")
    parser.add_argument("output_file", help="Path to the output text file for interpolated grid data.")
    parser.add_argument("grid_size", type=int, help="Integer size N for the N x N regular grid.")

    args = parser.parse_args()

    if args.grid_size <= 0:
        print("---mpScript_rbf.py: Error: Grid size (grid_size) must be a positive integer.", flush=True, file=sys.stderr)
        sys.exit(1) # Exit with error code

    # Execute the main function and exit with appropriate code
    if rbf_interpolate_to_grid(args.input_file, args.output_file, args.grid_size):
        sys.exit(0) # Success
    else:
        sys.exit(1) # Failure
