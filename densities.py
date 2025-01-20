import numpy as np
def compute_volume(region):
    """
            compute measure of a given set in R^n
    """
    volume = 1
    for interval in region:
        length = interval[1] - interval[0]
        volume *= length
    return volume

def uniform_rect_regions(x, regions):
    """
        Uniform over rectengular regions
    """
    # Initialize a boolean array to check if each vector is in any of the regions
    in_region = np.zeros(x.shape[0], dtype=int)  # Store which region the vector belongs to
    
    # Iterate through each region
    for i, region in enumerate(regions):
        # Check if x is within the bounds of this region (per dimension)
        condition = np.all((x >= region[:, 0]) & (x <= region[:, 1]), axis=1)
        
        # Mark the regions where the vector satisfies the condition
        in_region |= condition * (i + 1)  # Mark with a positive region index (1-based)
    
    # Map the region index to the region value and return it, 0 if not in any region
    result = np.zeros(x.shape[0])
    for i in range(len(regions)):
        result[in_region == (i + 1)] = compute_volume(regions[i])**(-1)
    return result

if __name__ == "__main__":
    x_values = np.array([[0.1, 0.2],
                         [0.2, 0.25],  
                         [0.5, 0.7], 
                         [0.1, 0.8],
                         [0.8, 0.8]])  

    # Define regions: shape (N_Regions, dimension, 2)
    # Region 1: (0, 0.3) for both dimensions, Region 2: (0.6, 0.9) for both dimensions
    regions = np.array([[[0, 0.3], [0, 0.3]],  # Region 1
                        [[0.6, 0.9], [0.7, 0.9]]])  # Region 2
    pdf_values = uniform_rect_regions(x_values, regions)

    print("Input values:\n", x_values)
    print("PDF values:", pdf_values)
