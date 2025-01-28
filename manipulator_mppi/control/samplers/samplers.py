#TODO: Implement Halton spline, and Sergery Levine's smoother

def generate_Gaussian(self, size, noise_sigma):
    """
    Generate noise for sampling actions.

    Args:
        size (tuple): Shape of the noise array.

    Returns:
        np.ndarray: Generated noise scaled by `noise_sigma`.
    """
    return self.random_generator.normal(size=size) * noise_sigma